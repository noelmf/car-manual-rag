"""Extraction is judged by its own report: a PDF can open fine and yield junk.

A scanned manual has no text layer, so it extracts to almost nothing without
raising anything. The heuristics in report() are the only thing standing
between that and 300 pages of empty JSONL, so they carry the tests.
"""

import json

import pymupdf

from car_manual_rag.ingest import text


def make_pdf(path, pages):
    """A real PDF with the given text on each page."""
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        if body:
            page.insert_text((72, 72), body, fontsize=11)
    doc.save(path)
    doc.close()
    return path


class TestExtract:
    def test_writes_one_line_per_page_numbered_from_one(self, tmp_path):
        pdf = make_pdf(tmp_path / "M.pdf", ["primera pagina", "segunda pagina"])
        stats = text.extract(str(pdf), str(tmp_path))
        lines = [
            json.loads(x) for x in (tmp_path / "M.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [d["page"] for d in lines] == [1, 2]
        assert "primera" in lines[0]["text"]
        assert stats["pages"] == 2

    def test_counts_accents_so_a_broken_text_layer_shows_up(self, tmp_path):
        pdf = make_pdf(tmp_path / "M.pdf", ["revisión periódica del vehículo"])
        assert text.extract(str(pdf), str(tmp_path))["accents"] > 0

    def test_counts_pages_with_no_readable_text_as_empty(self, tmp_path):
        pdf = make_pdf(tmp_path / "M.pdf", ["una pagina con bastante texto legible aqui", ""])
        assert text.extract(str(pdf), str(tmp_path))["empty"] == 1

    def test_a_file_that_is_not_a_pdf_is_reported_not_raised(self, tmp_path):
        # One bad file must not stop a run over 274 of them.
        bad = tmp_path / "roto.pdf"
        bad.write_bytes(b"esto no es un PDF")
        stats = text.extract(str(bad), str(tmp_path))
        assert "error" in stats and stats["name"] == "roto.pdf"

    def test_a_failure_leaves_no_partial_file(self, tmp_path):
        bad = tmp_path / "roto.pdf"
        bad.write_bytes(b"esto no es un PDF")
        text.extract(str(bad), str(tmp_path))
        assert not list(tmp_path.glob("*.part"))
        assert not list(tmp_path.glob("*.jsonl"))

    def test_the_output_is_named_after_the_pdf(self, tmp_path):
        pdf = make_pdf(tmp_path / "SEAT_Ibiza_11.25.pdf", ["texto"])
        stats = text.extract(str(pdf), str(tmp_path))
        assert stats["name"] == "SEAT_Ibiza_11.25.jsonl"


def stat(name="M.jsonl", pages=100, chars=100_000, empty=0, accents=1000):
    return {"name": name, "pages": pages, "chars": chars, "empty": empty, "accents": accents}


class TestReport:
    def test_a_healthy_manual_raises_no_flag(self):
        assert text.report([stat()]) is False

    def test_a_file_that_failed_to_open_is_flagged(self):
        assert text.report([{"name": "M.pdf", "error": "cannot open"}]) is True

    def test_a_manual_with_no_text_at_all_is_flagged(self):
        assert text.report([stat(chars=0)]) is True

    def test_too_little_text_per_page_is_flagged(self):
        # A scanned manual: it opens, it has pages, and almost no characters.
        assert text.report([stat(chars=100 * 199)]) is True

    def test_spanish_with_no_accents_at_all_is_flagged(self):
        # A text layer built from a bad encoding loses them first.
        assert text.report([stat(accents=0)]) is True

    def test_mostly_blank_pages_are_flagged(self):
        assert text.report([stat(empty=21)]) is True

    def test_a_healthy_manual_next_to_a_broken_one_still_flags(self):
        assert text.report([stat(), stat(name="roto.jsonl", chars=0)]) is True

    def test_an_empty_run_does_not_divide_by_zero(self):
        assert text.report([]) is False
