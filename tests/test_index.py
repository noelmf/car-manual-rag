"""An index that does not match its chunks cites the wrong page in silence."""

import json

import numpy as np
import pytest

from car_manual_rag import index


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "clave-de-prueba")
    monkeypatch.setenv("GEMINI_EMBEDDING", "modelo-de-prueba")


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    """A manual with its chunks on disk and a matching index built from fakes."""
    chunk_dir, index_dir = tmp_path / "chunks", tmp_path / "index"
    chunk_dir.mkdir()
    records = [
        {
            "chunk_id": "M:00000",
            "section": "Uno",
            "pages": [1],
            "printed": ["9"],
            "text": "primero",
        },
        {
            "chunk_id": "M:00001",
            "section": "Dos",
            "pages": [2, 3],
            "printed": ["10", "11"],
            "text": "segundo",
        },
    ]
    (chunk_dir / "M.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        index,
        "embed",
        lambda texts, kind, note=None: [[float(i + 1), 0.0] for i in range(len(texts))],
    )
    index.build("M", index_dir=index_dir, chunk_dir=chunk_dir)
    return chunk_dir, index_dir, records


class TestDigest:
    def test_the_same_text_gives_the_same_digest(self):
        a = [{"text": "uno"}, {"text": "dos"}]
        assert index.digest_of(a) == index.digest_of([dict(c) for c in a])

    def test_one_changed_character_changes_it(self):
        assert index.digest_of([{"text": "uno"}]) != index.digest_of([{"text": "uno."}])

    def test_moving_text_between_chunks_changes_it(self):
        # Without the separator, ['ab','c'] and ['a','bc'] would hash alike.
        assert index.digest_of([{"text": "ab"}, {"text": "c"}]) != index.digest_of(
            [{"text": "a"}, {"text": "bc"}]
        )


class TestBuild:
    """A half-written index is a file that cites pages it never read."""

    @pytest.fixture
    def one_chunk(self, monkeypatch):
        monkeypatch.setattr(index, "embed", lambda texts, kind, note=None: [[1.0, 0.0]])
        return [{"text": "uno"}]

    def test_a_write_that_fails_halfway_leaves_nothing_behind(
        self, tmp_path, one_chunk, monkeypatch
    ):
        def half_written(path, **arrays):
            path.write_bytes(b"a medias")
            raise OSError("no queda espacio")

        monkeypatch.setattr(index.np, "savez", half_written)
        with pytest.raises(OSError):
            index.build("M", chunks=one_chunk, index_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_even_an_interrupt_cleans_up(self, tmp_path, one_chunk, monkeypatch):
        # 'except BaseException', not 'except Exception': a Ctrl-C partway
        # through a forty-second embed must not leave a truncated index that
        # the next run would happily load.
        def interrupted(path, **arrays):
            path.write_bytes(b"a medias")
            raise KeyboardInterrupt

        monkeypatch.setattr(index.np, "savez", interrupted)
        with pytest.raises(KeyboardInterrupt):
            index.build("M", chunks=one_chunk, index_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_a_finished_build_leaves_only_the_index(self, tmp_path, one_chunk):
        index.build("M", chunks=one_chunk, index_dir=tmp_path)
        assert [p.name for p in tmp_path.iterdir()] == ["M.npz"]


class TestLoad:
    def test_a_matching_index_is_reused(self, indexed):
        chunk_dir, index_dir, records = indexed
        vectors, _ = index.load("M", index_dir=index_dir, chunk_dir=chunk_dir)
        assert len(vectors) == len(records)

    def test_a_missing_index_names_the_command_that_creates_it(self, tmp_path, indexed):
        chunk_dir, _, _ = indexed
        with pytest.raises(LookupError, match="crag-index M"):
            index.load("M", index_dir=tmp_path / "vacio", chunk_dir=chunk_dir)

    def test_changed_chunks_are_refused_even_with_the_same_ids(self, indexed, monkeypatch):
        # The reason the digest exists: chunk ids are positional, so re-chunking
        # can change every text while leaving the ids identical.
        chunk_dir, index_dir, records = indexed
        records[0]["text"] += " y algo mas"
        (chunk_dir / "M.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        with pytest.raises(LookupError, match="different chunks"):
            index.load("M", index_dir=index_dir, chunk_dir=chunk_dir)

    def test_another_embedding_model_is_refused(self, indexed, monkeypatch):
        chunk_dir, index_dir, _ = indexed
        monkeypatch.setenv("GEMINI_EMBEDDING", "otro-modelo")
        with pytest.raises(LookupError, match="was indexed with"):
            index.load("M", index_dir=index_dir, chunk_dir=chunk_dir)

    def test_a_file_this_version_did_not_write_is_refused(self, indexed):
        chunk_dir, index_dir, _ = indexed
        np.savez(index_dir / "M.npz", algo=np.zeros(3))
        with pytest.raises(LookupError, match="not an index"):
            index.load("M", index_dir=index_dir, chunk_dir=chunk_dir)

    def test_missing_chunks_point_at_the_chunker(self, tmp_path):
        with pytest.raises(LookupError, match="crag-chunk"):
            index.load("NO_EXISTE", index_dir=tmp_path, chunk_dir=tmp_path)


class TestVectors:
    def test_vectors_are_stored_normalised(self, indexed):
        chunk_dir, index_dir, _ = indexed
        vectors, _ = index.load("M", index_dir=index_dir, chunk_dir=chunk_dir)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_a_zero_vector_does_not_divide_by_zero(self, monkeypatch):
        monkeypatch.setattr(index, "embed", lambda texts, kind, note=None: [[0.0, 0.0]])
        assert np.isfinite(index.vectors_of(["x"], "passage")).all()


class TestSearch:
    """The ranking itself. load() is covered above; this is what it feeds."""

    @pytest.fixture
    def corpus(self, monkeypatch):
        # Unit vectors at known angles to the query [1, 0], so the expected
        # order is arithmetic rather than a guess: 1.0, 0.6, 0.0, -1.0.
        vectors = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        chunks = [{"chunk_id": f"M:{i}", "text": t} for i, t in enumerate("abcd")]
        monkeypatch.setattr(index, "load", lambda manual_id: (vectors, chunks))
        monkeypatch.setattr(index, "embed", lambda texts, kind, note=None: [[1.0, 0.0]])
        return chunks

    def test_the_closest_chunk_comes_first_and_the_opposite_one_last(self, corpus):
        assert [h["chunk_id"] for h in index.search("M", "pregunta")] == [
            "M:0",
            "M:1",
            "M:2",
            "M:3",
        ]

    def test_k_caps_how_many_come_back(self, corpus):
        assert len(index.search("M", "pregunta", k=2)) == 2

    def test_every_hit_carries_the_score_it_was_ranked_by(self, corpus):
        hits = index.search("M", "pregunta", k=2)
        assert [h["score"] for h in hits] == [pytest.approx(1.0), pytest.approx(0.6)]

    def test_a_hit_keeps_the_chunk_it_came_from(self, corpus):
        assert index.search("M", "pregunta", k=1)[0]["text"] == "a"

    def test_the_stored_chunks_are_not_given_a_score(self, corpus):
        # The hit is a copy: scoring the cached chunks in place would leak the
        # previous question's scores into the next search.
        index.search("M", "pregunta")
        assert all("score" not in c for c in corpus)


class TestLabel:
    def test_is_the_section_and_carries_no_page(self):
        out = index.label({"section": "Frenos", "printed": ["10", "11"], "pages": [2, 3]})
        assert out == "Frenos"
        assert "10" not in out and "pag" not in out

    def test_a_chunk_with_no_section_still_has_a_label(self):
        assert index.label({"section": None, "printed": ["10"], "pages": [2]}) == "sin seccion"


class TestPagesOf:
    def test_uses_the_printed_page_the_reader_can_see(self):
        assert index.pages_of({"printed": ["10", "11"], "pages": [2, 3]}) == "10-11"

    def test_falls_back_to_the_pdf_page_when_none_was_printed(self):
        assert index.pages_of({"printed": [], "pages": [2, 3]}) == "2-3"
