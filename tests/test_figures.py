"""Figures are cut by rendering the page, which is what keeps the callouts.

The fixtures build real PDFs rather than mocking pymupdf: what this stage has
to get right -- which images count, where they sit, what the page draws over
them -- only exists in a real document.
"""

import json

import pymupdf
import pytest

from car_manual_rag.ingest import figures


def block(width, height, colour=(1, 0, 0)):
    """A solid image of a given size in pixels."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pix.set_rect(pix.irect, tuple(int(c * 255) for c in colour))
    return pix


@pytest.fixture
def manual(tmp_path):
    """A two-page PDF: one real figure, one image too small to be one."""

    def write(name="SEAT_Test_01.25", small=True):
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=300)
        page.insert_image(pymupdf.Rect(20, 20, 220, 170), pixmap=block(200, 150))
        if small:
            page.insert_image(pymupdf.Rect(300, 20, 330, 50), pixmap=block(40, 40))
        doc.new_page(width=400, height=300)  # a page with nothing on it
        path = tmp_path / f"{name}.pdf"
        doc.save(path)
        doc.close()
        return path

    return write


class TestPlacements:
    def test_a_real_figure_is_found_with_its_rectangle(self, manual):
        with pymupdf.open(manual()) as doc:
            found = figures.placements(doc[0])
        assert len(found) == 1
        bbox, width, height = found[0]
        assert (width, height) == (200, 150)
        assert round(bbox.x0) == 20 and round(bbox.y0) == 20

    def test_an_image_too_small_to_be_a_figure_is_dropped(self, manual):
        # 40x40 is 1,600 square pixels, far under the 10,000 threshold.
        with pymupdf.open(manual()) as doc:
            assert len(figures.placements(doc[0])) == 1

    def test_a_page_with_no_images_yields_none(self, manual):
        with pymupdf.open(manual()) as doc:
            assert figures.placements(doc[1]) == []

    def test_the_threshold_is_the_gap_between_rules_and_figures(self):
        assert 5_000 <= figures.MIN_AREA <= 20_000


class StubPage:
    """A page handing back exactly the images and rectangle a test wants.

    A PDF whose image rectangle cannot be resolved is not something the corpus
    lets us build on demand, so the two ways that can go wrong are staged here.
    """

    def __init__(self, bbox, images=((1, 0, 200, 150),)):
        self._bbox, self._images = bbox, images

    def get_images(self, full=True):
        return list(self._images)

    def get_image_bbox(self, image):
        if isinstance(self._bbox, Exception):
            raise self._bbox
        return self._bbox


class TestUnresolvableRectangles:
    def test_a_rectangle_the_renderer_refuses_is_skipped(self):
        assert figures.placements(StubPage(ValueError("no matrix"))) == []

    def test_an_infinite_rectangle_is_not_a_figure(self):
        assert figures.placements(StubPage(pymupdf.Rect(1, 1, -1, -1))) == []

    def test_an_empty_rectangle_is_not_a_figure(self):
        assert figures.placements(StubPage(pymupdf.Rect(5, 5, 5, 5))) == []


class TestCut:
    def test_writes_one_jpeg_per_figure_and_an_index(self, manual, tmp_path):
        out = tmp_path / "out"
        stats = figures.cut(str(manual()), out)
        assert stats["figures"] == 1
        folder = out / "SEAT_Test_01.25"
        assert (folder / "p0001_1.jpg").is_file()
        assert (folder / figures.INDEX).is_file()

    def test_the_index_records_where_the_figure_came_from(self, manual, tmp_path):
        out = tmp_path / "out"
        figures.cut(str(manual()), out)
        record = json.loads(
            (out / "SEAT_Test_01.25" / figures.INDEX).read_text(encoding="utf-8").splitlines()[0]
        )
        assert record["figure_id"] == "SEAT_Test_01.25:0001:1"
        assert record["page"] == 1 and record["file"] == "p0001_1.jpg"
        assert record["source_px"] == [200, 150]

    def test_the_file_is_a_jpeg_the_browser_can_show(self, manual, tmp_path):
        out = tmp_path / "out"
        figures.cut(str(manual()), out)
        data = (out / "SEAT_Test_01.25" / "p0001_1.jpg").read_bytes()
        assert data[:3] == b"\xff\xd8\xff"  # JPEG, not the PDF's CMYK stream

    def test_a_broken_source_is_reported_not_raised(self, tmp_path):
        bad = tmp_path / "SEAT_Roto_01.25.pdf"
        bad.write_text("not a pdf", encoding="utf-8")
        assert "error" in figures.cut(str(bad), tmp_path / "out")

    def test_a_failure_leaves_no_index_so_the_manual_is_done_again(self, tmp_path):
        bad = tmp_path / "SEAT_Roto_01.25.pdf"
        bad.write_text("not a pdf", encoding="utf-8")
        figures.cut(str(bad), tmp_path / "out")
        assert not (tmp_path / "out" / "SEAT_Roto_01.25" / figures.INDEX).exists()


class TestUndecodableFigures:
    """A JPEG the decoder chokes on is one figure lost, not a whole manual.

    163 of the 274 manuals carry at least one image whose height arrives in a
    DNL marker, which MuPDF refuses. Aborting the manual on the first of them
    cost 59% of the corpus.
    """

    def test_the_manual_still_finishes_and_records_the_loss(self, manual, tmp_path, monkeypatch):
        monkeypatch.setattr(
            figures, "render", lambda *a: (_ for _ in ()).throw(RuntimeError("DNL not supported"))
        )
        stats = figures.cut(str(manual()), tmp_path / "out")
        assert "error" not in stats
        assert stats["figures"] == 0 and stats["skipped"] == 1

    def test_an_index_is_still_written_so_the_manual_is_not_redone(
        self, manual, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            figures, "render", lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        out = tmp_path / "out"
        figures.cut(str(manual()), out)
        assert (out / "SEAT_Test_01.25" / figures.INDEX).is_file()

    def test_a_manual_losing_most_of_its_figures_is_flagged(self):
        assert figures.report([{"name": "m", "figures": 1, "skipped": 9, "pages": 1, "bytes": 10}])

    def test_losing_the_odd_one_is_not_worth_flagging(self):
        assert not figures.report(
            [{"name": "m", "figures": 99, "skipped": 1, "pages": 40, "bytes": 1000}]
        )


class TestLoading:
    def test_a_manual_without_figures_names_the_command(self, tmp_path):
        with pytest.raises(LookupError, match="crag-figures"):
            figures.load("SEAT_Nada_01.25", tmp_path)

    def test_on_pages_returns_only_the_figures_of_those_pages(self, manual, tmp_path):
        out = tmp_path / "out"
        figures.cut(str(manual()), out)
        assert len(figures.on_pages("SEAT_Test_01.25", [1], out)) == 1
        assert figures.on_pages("SEAT_Test_01.25", [2], out) == []

    def test_on_pages_accepts_the_pages_a_chunk_spans(self, manual, tmp_path):
        out = tmp_path / "out"
        figures.cut(str(manual()), out)
        assert len(figures.on_pages("SEAT_Test_01.25", [1, 2], out)) == 1


class TestByPage:
    def test_groups_the_figures_by_the_page_they_are_printed_on(self, manual, tmp_path):
        out = tmp_path / "out"
        figures.cut(str(manual()), out)
        grouped = figures.by_page("SEAT_Test_01.25", out)
        assert list(grouped) == [1]
        assert grouped[1][0]["file"] == "p0001_1.jpg"

    def test_a_manual_without_figures_is_empty_not_an_error(self, tmp_path):
        # Answering is not held up by a stage nobody has run: a missing drawing
        # leaves the answer poorer, not wrong.
        assert figures.by_page("SEAT_Nada_01.25", tmp_path) == {}


class TestReport:
    def test_a_healthy_run_raises_no_flag(self):
        assert not figures.report(
            [{"name": "m", "figures": 8, "skipped": 0, "pages": 6, "bytes": 4000}]
        )

    def test_a_failure_is_flagged(self):
        assert figures.report([{"name": "m", "error": "boom"}])

    def test_a_manual_with_no_figures_is_flagged(self):
        assert figures.report([{"name": "m", "figures": 0, "skipped": 0, "pages": 0, "bytes": 0}])

    def test_a_long_list_of_failures_is_truncated(self, capsys):
        figures.report([{"name": f"m{i}", "error": "boom"} for i in range(12)])
        assert "... and 2 more" in capsys.readouterr().out

    def test_an_empty_run_does_not_divide_by_zero(self):
        assert not figures.report([])
