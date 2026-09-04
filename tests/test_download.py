"""The downloader's job is to refuse anything that is not a manual.

A server that answers an expired URL with an HTML error page returns 200 and a
few KB of markup. Saved as a .pdf, the next run would skip it as done and the
whole pipeline would carry a hole nobody notices until a question comes back
empty. These tests are about that refusal.
"""

import urllib.error

import pytest

from car_manual_rag.ingest import download

PDF = b"%PDF-1.7\n" + b"contenido" * 100


class FakeResponse:
    def __init__(self, body, content_type):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def server(monkeypatch):
    """Make urlopen replay the given outcomes, one per attempt."""

    def serve(*outcomes):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(request)
            result = outcomes[min(len(calls), len(outcomes)) - 1]
            if isinstance(result, Exception):
                raise result
            return FakeResponse(*result)

        monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(download.time, "sleep", lambda s: None)
        return calls

    return serve


class TestValidation:
    def test_writes_a_real_pdf_and_reports_its_size(self, server, tmp_path):
        server((PDF, "application/pdf"))
        path = tmp_path / "manual.pdf"
        assert download.download("http://x", path) == len(PDF)
        assert path.read_bytes() == PDF

    def test_rejects_a_wrong_content_type(self, server, tmp_path):
        server((PDF, "text/html; charset=utf-8"))
        with pytest.raises(ValueError, match="not a PDF"):
            download.download("http://x", tmp_path / "manual.pdf")

    def test_rejects_a_body_that_is_not_a_pdf(self, server, tmp_path):
        # The dangerous case: the right header on an error page.
        server((b"<html>404</html>", "application/pdf"))
        with pytest.raises(ValueError, match="does not start with"):
            download.download("http://x", tmp_path / "manual.pdf")

    def test_a_rejected_download_leaves_nothing_behind(self, server, tmp_path):
        server((b"<html>404</html>", "application/pdf"))
        path = tmp_path / "manual.pdf"
        with pytest.raises(ValueError):
            download.download("http://x", path)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_an_uppercase_content_type_is_accepted(self, server, tmp_path):
        server((PDF, "Application/PDF"))
        assert download.download("http://x", tmp_path / "manual.pdf") == len(PDF)


class TestRetries:
    def test_retries_a_rate_limit_then_succeeds(self, server, tmp_path):
        calls = server(
            urllib.error.HTTPError("http://x", 429, "", {}, None),
            (PDF, "application/pdf"),
        )
        download.download("http://x", tmp_path / "manual.pdf")
        assert len(calls) == 2

    def test_does_not_retry_a_not_found(self, server, tmp_path):
        calls = server(urllib.error.HTTPError("http://x", 404, "", {}, None))
        with pytest.raises(urllib.error.HTTPError):
            download.download("http://x", tmp_path / "manual.pdf")
        assert len(calls) == 1

    def test_gives_up_after_the_last_attempt(self, server, tmp_path):
        calls = server(urllib.error.HTTPError("http://x", 503, "", {}, None))
        with pytest.raises(urllib.error.HTTPError):
            download.download("http://x", tmp_path / "manual.pdf")
        assert len(calls) == download.RETRIES

    def test_retries_a_dropped_connection(self, server, tmp_path):
        calls = server(urllib.error.URLError("connection reset"), (PDF, "application/pdf"))
        download.download("http://x", tmp_path / "manual.pdf")
        assert len(calls) == 2
