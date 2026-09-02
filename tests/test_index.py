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


class TestCite:
    def test_uses_the_printed_page_the_reader_can_see(self):
        assert (
            index.cite({"section": "Frenos", "printed": ["10", "11"], "pages": [2, 3]})
            == "Frenos, pag. 10-11"
        )

    def test_falls_back_to_the_pdf_page_when_none_was_printed(self):
        assert "2-3" in index.cite({"section": "Frenos", "printed": [], "pages": [2, 3]})

    def test_a_chunk_with_no_section_still_cites(self):
        assert (
            index.cite({"section": None, "printed": ["10"], "pages": [2]}) == "sin seccion, pag. 10"
        )
