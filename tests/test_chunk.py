"""The chunker decides the quality of every answer, so it carries the most tests."""

import json

from car_manual_rag.ingest import chunk


def lines(pages):
    return [chunk.lines_of(p["text"]) for p in pages]


class TestReflow:
    def test_joins_a_word_split_across_lines(self):
        assert chunk.reflow(["habita-", "culo."]) == ["habitaculo."]

    def test_keeps_the_hyphen_of_a_compound(self):
        # 'Start-\nStop' is one word with a hyphen, not a wrapped one.
        assert chunk.reflow(["Start-", "Stop activo"]) == ["Start-Stop activo"]

    def test_joins_wrapped_lines_into_one_paragraph(self):
        assert chunk.reflow(["El aire acondicionado enfria", "y deshumedece el aire"]) == [
            "El aire acondicionado enfria y deshumedece el aire"
        ]

    def test_a_bullet_starts_a_new_paragraph(self):
        out = chunk.reflow(["Texto previo", "●Primer paso", "●Segundo paso"])
        assert len(out) == 3 and out[1].startswith("- Primer")

    def test_a_figure_caption_starts_a_new_paragraph(self):
        out = chunk.reflow(["Texto previo", "Fig. 81  En la consola"])
        assert len(out) == 2


class TestChrome:
    def test_finds_the_head_repeated_across_pages(self, pages):
        assert "Climatizacion" in chunk.find_chrome(lines(pages * 2))

    def test_head_becomes_the_section_and_leaves_the_body(self, pages):
        page = chunk.lines_of(pages[0]["text"])
        body, section, printed = chunk.strip_chrome(page, {"Climatizacion"})
        assert section == "Climatizacion"
        assert printed == "117"
        assert "Climatizacion" not in body and "117" not in body

    def test_a_decorative_glyph_is_stripped_but_is_not_a_title(self):
        # The bug found in 72 manuals: '>>' repeated at a page edge was taken
        # for a chapter title, and citations showed it instead of a section.
        page = ["»", "El aire acondicionado enfria.", "42"]
        body, section, _ = chunk.strip_chrome(page, {"»"})
        assert section is None
        assert "»" not in body

    def test_control_characters_never_survive(self):
        # A section of '\x02' rendered as an invisible character in a citation.
        out = chunk.lines_of("\x02\nTexto legible\n")
        assert out == ["Texto legible"]

    def test_icon_font_glyphs_are_stripped(self):
        # Private-use characters are what the symbol fonts leave behind.
        assert chunk.lines_of("\ue003Pulse la tecla") == ["Pulse la tecla"]


class TestNavPages:
    def test_a_contents_page_is_recognised(self):
        from tests.conftest import NAV_PAGE

        assert chunk.is_nav(chunk.lines_of(NAV_PAGE["text"]))

    def test_a_body_page_is_not(self, pages):
        assert not chunk.is_nav(chunk.lines_of(pages[0]["text"]))

    def test_an_empty_page_is_not(self):
        assert not chunk.is_nav([])


class TestGrouping:
    def unit(self, text, section="A", page=1):
        return {"text": text, "page": page, "printed": str(page), "section": section}

    def test_a_section_change_always_breaks_a_chunk(self):
        units = [self.unit("uno", "A"), self.unit("dos", "B")]
        assert len(chunk.group(units, target=10_000, overlap=0)) == 2

    def test_overlap_repeats_the_tail_of_the_previous_chunk(self):
        units = [self.unit("a" * 40) for _ in range(3)]
        groups = chunk.group(units, target=80, overlap=40)
        assert groups[1][0] is groups[0][-1]

    def test_overlap_never_crosses_a_section(self):
        units = [self.unit("a" * 40, "A"), self.unit("b" * 40, "B")]
        groups = chunk.group(units, target=80, overlap=40)
        assert len(groups[1]) == 1

    def test_a_paragraph_longer_than_a_chunk_is_split(self):
        pieces = chunk.split_long("palabra " * 500, 1200)
        assert len(pieces) > 1 and all(len(p) <= 1200 for p in pieces)

    def test_a_short_paragraph_is_left_alone(self):
        assert chunk.split_long("corto", 1200) == ["corto"]


class TestUnits:
    def test_nav_pages_are_dropped_and_sections_carry_forward(self, pages):
        from tests.conftest import NAV_PAGE

        units, dropped = chunk.units_of(pages * 2 + [NAV_PAGE])
        assert dropped == 1
        assert {u["section"] for u in units} == {"Climatizacion"}

    def test_the_printed_page_is_kept_not_the_pdf_page(self, pages):
        units, _ = chunk.units_of(pages * 2)
        assert {u["printed"] for u in units} == {"117", "118", "119"}


def text_file(tmp_path, pages, manual_id="M"):
    """A page-per-line JSONL like the one crag-extract writes."""
    path = tmp_path / f"{manual_id}.jsonl"
    path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pages) + "\n", encoding="utf-8"
    )
    return path


class TestChunkFile:
    def test_writes_chunks_with_their_page_and_section(self, tmp_path, pages):
        out = tmp_path / "out"
        out.mkdir()
        stats = chunk.chunk(str(text_file(tmp_path, pages * 2)), str(out), 1200, 150)
        written = [
            json.loads(x) for x in (out / "M.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert stats["chunks"] == len(written) > 0
        assert all(c["section"] == "Climatizacion" for c in written)
        assert all(c["manual_id"] == "M" for c in written)

    def test_chunk_ids_are_sequential_within_the_manual(self, tmp_path, pages):
        out = tmp_path / "out"
        out.mkdir()
        chunk.chunk(str(text_file(tmp_path, pages * 3)), str(out), 400, 0)
        ids = [json.loads(x)["chunk_id"] for x in (out / "M.jsonl").read_text().splitlines()]
        assert ids == [f"M:{i:05d}" for i in range(len(ids))]

    def test_a_broken_source_is_reported_not_raised(self, tmp_path):
        bad = tmp_path / "roto.jsonl"
        bad.write_text("esto no es json\n", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()
        assert "error" in chunk.chunk(str(bad), str(out), 1200, 150)

    def test_a_failure_leaves_no_partial_file(self, tmp_path):
        bad = tmp_path / "roto.jsonl"
        bad.write_text("esto no es json\n", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()
        chunk.chunk(str(bad), str(out), 1200, 150)
        assert list(out.iterdir()) == []

    def test_nav_pages_are_counted_as_dropped(self, tmp_path, pages):
        from tests.conftest import NAV_PAGE

        out = tmp_path / "out"
        out.mkdir()
        stats = chunk.chunk(str(text_file(tmp_path, pages * 2 + [NAV_PAGE])), str(out), 1200, 150)
        assert stats["nav"] == 1


def chunk_stat(name="M.jsonl", pages=100, chunks=300, chars=330_000, unsectioned=0):
    return {
        "name": name,
        "pages": pages,
        "nav": 0,
        "chunks": chunks,
        "chars": chars,
        "median": 1100,
        "unsectioned": unsectioned,
    }


class TestReport:
    def test_a_healthy_run_raises_no_flag(self):
        assert chunk.report([chunk_stat()]) is False

    def test_a_failure_is_flagged(self):
        assert chunk.report([{"name": "M.jsonl", "error": "roto"}]) is True

    def test_a_manual_with_no_chunks_is_flagged(self):
        assert chunk.report([chunk_stat(chunks=0)]) is True

    def test_fewer_chunks_than_pages_is_flagged(self):
        # Fewer than one chunk per page means the text mostly vanished.
        assert chunk.report([chunk_stat(pages=100, chunks=99)]) is True

    def test_too_many_chunks_without_a_section_is_flagged(self):
        # The running-head detector failed on that manual's layout.
        assert chunk.report([chunk_stat(chunks=300, unsectioned=31)]) is True

    def test_an_empty_run_does_not_divide_by_zero(self):
        assert chunk.report([]) is False
