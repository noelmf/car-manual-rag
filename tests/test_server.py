"""The API, exercised over a real socket.

The routing, the status codes and the JSON envelope only exist as HTTP, so the
tests speak HTTP: a server on an ephemeral port and http.client against it.
That also keeps the path handling honest -- a raw request line reaches the
handler exactly as a browser would send it, traversal attempts included.
"""

import contextlib
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from car_manual_rag import server

HITS = [
    {
        "section": "Frenos",
        "printed": ["10"],
        "pages": [2],
        "text": "El liquido de frenos.",
        "chunk_id": "M:0",
        "score": 0.75,
    }
]

FIGURE = {
    "figure_id": "M:0002:1",
    "file": "p0002_1.jpg",
    "page": 2,
    "width": 277,
    "height": 217,
}

ANSWERED = {
    "answer": "Cada dos anos.",
    "hits": HITS,
    "model": "modelo",
    "usage": {"totalTokenCount": 42},
}


class Quiet(server.Handler):
    """The handler as it is, minus the request log that would flood the run."""

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def client():
    """A running server, and a call() returning its status, body and type.

    One server for the whole module: shutdown() polls before it returns, so a
    server per test would spend most of the run waiting to stop. Tests still
    get to redirect it, because what they replace -- server.ask, server.DIST --
    are module globals it reads on each request.
    """
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def call(method, path, body=None, timeout=5):
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=timeout)
        payload = None if body is None else json.dumps(body)
        conn.request(method, path, payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        kind = response.getheader("Content-Type") or ""
        return SimpleNamespace(
            status=response.status,
            raw=raw,
            type=kind,
            body=json.loads(raw) if kind.startswith("application/json") else None,
            events=[json.loads(line) for line in raw.splitlines() if line]
            if kind.startswith("application/x-ndjson")
            else None,
        )

    call.port = httpd.server_port
    yield call
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def answers(monkeypatch):
    """Stand in for ask(), so the API is tested without calling Gemini."""

    def use(result=ANSWERED):
        monkeypatch.setattr(server, "ask", lambda *a, **k: result)

    return use


class TestCatalog:
    def test_serves_the_picker_nested_brand_model_year_edition(self, client):
        out = client("GET", "/api/catalog")
        assert out.status == 200
        editions = out.body["SEAT"]["Ibiza"]["2026"]
        assert all(isinstance(manual_id, str) for manual_id in editions.values())

    def test_is_parsed_once_and_reused(self):
        assert server.catalog() is server.catalog()


class TestAsk:
    def test_returns_the_answer_with_its_citations(self, client, answers):
        answers()
        out = client("POST", "/api/ask", {"manual": "M", "question": "¿cuando?"})
        assert out.status == 200
        assert out.body["answer"] == "Cada dos anos."
        assert out.body["hits"][0]["section"] == "Frenos"
        assert out.body["hits"][0]["pages"] == "10"
        assert out.body["tokens"] == 42

    def test_a_body_without_a_question_is_the_callers_mistake(self, client):
        out = client("POST", "/api/ask", {"manual": "M"})
        assert out.status == 400

    def test_a_body_too_large_to_be_a_question_is_refused(self, client):
        out = client(
            "POST",
            "/api/ask",
            {"manual": "M", "question": "x" * (server.MAX_BODY + 1)},
        )
        assert out.status == 400

    def test_an_unindexed_manual_answers_with_the_command_that_fixes_it(self, client, monkeypatch):
        def missing(*a, **k):
            raise LookupError("M is not indexed -- run: crag-index M")

        monkeypatch.setattr(server, "ask", missing)
        out = client("POST", "/api/ask", {"manual": "M", "question": "¿cuando?"})
        assert out.status == 404
        assert "crag-index M" in out.body["error"]

    def test_a_failure_at_gemini_is_reported_as_an_upstream_failure(self, client, monkeypatch):
        def broken(*a, **k):
            raise RuntimeError("503 from generativelanguage")

        monkeypatch.setattr(server, "ask", broken)
        out = client("POST", "/api/ask", {"manual": "M", "question": "¿cuando?"})
        assert out.status == 502

    def test_posting_anywhere_else_is_not_a_route(self, client):
        assert client("POST", "/api/other", {}).status == 404

    def test_getting_an_unknown_api_path_is_not_a_route(self, client):
        assert client("GET", "/api/other").status == 404

    def test_a_refused_post_leaves_the_connection_fit_for_the_next_request(self, client):
        # A body answered but never read stays in the socket, and keep-alive
        # then reads it as the next request line. Two requests on one
        # connection is the only way to see that.
        conn = http.client.HTTPConnection("127.0.0.1", client.port, timeout=5)
        conn.request("POST", "/api/other", "{}", {"Content-Type": "application/json"})
        assert conn.getresponse().read()
        conn.request("GET", "/api/catalog")
        assert conn.getresponse().status == 200
        conn.close()


class TestStreamingAsk:
    """The fragments leave in half a second; the prose takes seconds more."""

    def stream(self, monkeypatch, events):
        monkeypatch.setattr(server, "ask_stream", lambda *a, **k: iter(events))

    def hits_event(self):
        return {"event": "hits", "hits": HITS}

    def test_answers_newline_delimited_json(self, client, monkeypatch):
        self.stream(monkeypatch, [self.hits_event(), {"event": "done", "model": "m"}])
        out = client("POST", "/api/ask/stream", {"manual": "M", "question": "¿q?"})
        assert out.status == 200
        assert out.type.startswith("application/x-ndjson")

    def test_the_fragments_arrive_before_the_prose(self, client, monkeypatch):
        self.stream(
            monkeypatch,
            [self.hits_event(), {"event": "token", "text": "Cada"}, {"event": "done"}],
        )
        out = client("POST", "/api/ask/stream", {"manual": "M", "question": "¿q?"})
        assert [e["event"] for e in out.events] == ["hits", "token", "done"]

    def test_the_fragments_carry_their_figures(self, client, monkeypatch):
        self.stream(monkeypatch, [self.hits_event(), {"event": "done"}])
        monkeypatch.setattr(server, "by_page", lambda *a: {2: [FIGURE]})
        out = client("POST", "/api/ask/stream", {"manual": "M", "question": "¿q?"})
        figures = out.events[0]["hits"][0]["figures"]
        assert figures[0]["url"] == "/api/figure/M/p0002_1.jpg"

    def test_an_unindexed_manual_is_refused_before_the_stream_starts(self, client, monkeypatch):
        def missing(*a, **k):
            raise LookupError("M is not indexed -- run: crag-index M")
            yield  # pragma: no cover - never reached

        monkeypatch.setattr(server, "ask_stream", missing)
        out = client("POST", "/api/ask/stream", {"manual": "M", "question": "¿q?"})
        assert out.status == 404 and "crag-index M" in out.body["error"]

    def test_gemini_refusing_up_front_is_an_upstream_failure(self, client, monkeypatch):
        def refuses(*a, **k):
            raise RuntimeError("503 from generativelanguage")
            yield  # pragma: no cover - never reached

        monkeypatch.setattr(server, "ask_stream", refuses)
        out = client("POST", "/api/ask/stream", {"manual": "M", "question": "¿q?"})
        assert out.status == 502

    def test_a_reader_that_leaves_is_not_an_error_to_report(self, client, monkeypatch):
        # Nobody is listening any more, so there is nobody to tell.
        self.stream(monkeypatch, [self.hits_event(), {"event": "done"}])
        original = server.Handler.send_event
        calls = []

        def hangs_up(self, event):
            calls.append(event)
            if len(calls) > 1:
                raise BrokenPipeError(32, "Broken pipe")
            return original(self, event)

        monkeypatch.setattr(server.Handler, "send_event", hangs_up)
        with contextlib.suppress(OSError, http.client.HTTPException):
            # The reply is cut off mid-stream, so reading it fails here too.
            # A short deadline: this reply never gets its terminating chunk,
            # so waiting the usual five seconds would only slow the suite.
            client("POST", "/api/ask/stream", {"manual": "M", "question": "¿q?"}, timeout=0.5)
        assert len(calls) == 2  # it stopped, and the handler did not blow up

    def test_a_failure_after_the_first_line_is_reported_inside_the_stream(
        self, client, monkeypatch
    ):
        def breaks():
            yield {"event": "hits", "hits": HITS}
            raise RuntimeError("gemini went away")

        monkeypatch.setattr(server, "ask_stream", lambda *a, **k: breaks())
        out = client("POST", "/api/ask/stream", {"manual": "M", "question": "¿q?"})
        assert out.status == 200  # the reply had already begun
        assert out.events[-1] == {"event": "error", "error": "gemini went away"}


class TestFigureFiles:
    def test_serves_the_jpeg(self, client, monkeypatch, tmp_path):
        folder = tmp_path / "SEAT_Test_01.25"
        folder.mkdir()
        (folder / "p0002_1.jpg").write_bytes(b"\xff\xd8\xffdatos")
        monkeypatch.setattr(server, "FIGURE_DIR", tmp_path)
        out = client("GET", "/api/figure/SEAT_Test_01.25/p0002_1.jpg")
        assert out.status == 200 and out.type == "image/jpeg"
        assert out.raw == b"\xff\xd8\xffdatos"

    def test_a_missing_figure_names_the_command(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(server, "FIGURE_DIR", tmp_path)
        out = client("GET", "/api/figure/SEAT_Test_01.25/nope.jpg")
        assert out.status == 404 and "crag-figures" in out.body["error"]

    def test_a_path_climbing_out_of_the_figures_is_refused(self, client, monkeypatch, tmp_path):
        store = tmp_path / "figures"
        store.mkdir()
        (tmp_path / "secreto.txt").write_text("no", encoding="utf-8")
        monkeypatch.setattr(server, "FIGURE_DIR", store)
        out = client("GET", "/api/figure/../secreto.txt")
        assert out.status == 404


class TestAnswerShape:
    def test_carries_only_what_the_ui_shows(self, monkeypatch, answers):
        answers()
        out = server.answer("M", "¿cuando?")
        assert set(out) == {"answer", "model", "tokens", "hits"}
        assert set(out["hits"][0]) == {"section", "pages", "score", "text", "chunk_id", "figures"}


class TestStaticUI:
    def test_serves_the_built_page(self, client, monkeypatch, tmp_path):
        (tmp_path / "index.html").write_text("<h1>hola</h1>", encoding="utf-8")
        monkeypatch.setattr(server, "DIST", tmp_path)
        out = client("GET", "/")
        assert out.status == 200 and b"hola" in out.raw
        assert out.type.startswith("text/html")

    def test_serves_an_asset_with_its_own_type(self, client, monkeypatch, tmp_path):
        (tmp_path / "index.html").write_text("<h1>hola</h1>", encoding="utf-8")
        (tmp_path / "app.js").write_text("export default 1;", encoding="utf-8")
        monkeypatch.setattr(server, "DIST", tmp_path)
        out = client("GET", "/app.js")
        assert out.status == 200 and out.type.startswith("text/javascript")

    def test_an_unknown_path_is_a_route_inside_the_single_page(self, client, monkeypatch, tmp_path):
        (tmp_path / "index.html").write_text("<h1>hola</h1>", encoding="utf-8")
        monkeypatch.setattr(server, "DIST", tmp_path)
        assert b"hola" in client("GET", "/cualquier/ruta").raw

    def test_a_path_climbing_out_of_dist_gets_the_page_not_the_file(
        self, client, monkeypatch, tmp_path
    ):
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<h1>hola</h1>", encoding="utf-8")
        (tmp_path / "secreto.txt").write_text("no", encoding="utf-8")
        monkeypatch.setattr(server, "DIST", dist)
        out = client("GET", "/../secreto.txt")
        assert out.status == 200 and b"hola" in out.raw and b"no" not in out.raw

    def test_without_a_build_it_says_how_to_build(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(server, "DIST", tmp_path / "missing")
        out = client("GET", "/")
        assert out.status == 404 and "pnpm --dir web run build" in out.body["error"]
