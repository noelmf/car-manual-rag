"""The HTTP API the web UI talks to, on the standard library alone.

    crag-serve            # http://127.0.0.1:8000

Two endpoints, because the picker and the question are the only two things the
UI does:

    GET  /api/catalog                     the picker, brand -> model -> year
    POST /api/ask                         {manual, question} -> the whole answer
    POST /api/ask/stream                  the same, as it is produced
    GET  /api/figure/<manual>/<file>      one figure, as JPEG

/api/ask/stream exists because of where the seconds go. The fragments are found
in half a second and the model needs several more before its first word, so the
buffered endpoint shows nothing at all for that whole time while the streaming
one has the evidence on screen almost at once. It answers newline-delimited
JSON rather than Server-Sent Events: SSE only buys EventSource, EventSource
cannot POST, and a request carrying a body has to be read with fetch() either
way.

The catalogue goes over the wire in one piece rather than as a cascade of
requests. It is 274 manuals of nested strings, so the client can walk it
without asking again -- four round trips to fill four dropdowns would be four
times the latency for data that fits in one.

Asking never indexes, for the same reason crag-ask never does: a question must
not spend money as a side effect. An unindexed manual comes back as a 404
carrying the command that fixes it, which is what index.load already writes.

This serves the built UI too when web/dist exists, so a checkout has one
command to run instead of two. In development Vite serves the UI itself and
proxies /api here, so that branch is dead until somebody runs `pnpm run build`.
"""

import argparse
import json
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from car_manual_rag.ask import ask, ask_stream
from car_manual_rag.config import FIGURE_DIR, ROOT
from car_manual_rag.index import TOP_K, label, pages_of
from car_manual_rag.ingest.catalog import load, tree
from car_manual_rag.ingest.figures import by_page

DIST = ROOT / "web" / "dist"
MAX_BODY = 64 * 1024  # a question, not an upload
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


@lru_cache(maxsize=1)
def catalog():
    """The picker, parsed once and shared by every request."""
    return tree(load())


def shaped(manual_id, hits):
    """The fragments as the interface reads them: evidence plus its drawings.

    A fragment's figures are the ones printed on the pages it came from. The
    figure travels as a URL rather than as bytes, so a long answer stays a
    small reply and the browser caches each drawing once.
    """
    figures = by_page(manual_id)
    return [
        {
            "section": label(hit),
            "pages": pages_of(hit),
            "score": hit["score"],
            "text": hit["text"],
            "chunk_id": hit["chunk_id"],
            "figures": [
                {
                    "figure_id": f["figure_id"],
                    "url": f"/api/figure/{manual_id}/{f['file']}",
                    "page": f["page"],
                    "width": f["width"],
                    "height": f["height"],
                }
                for page in hit["pages"]
                for f in figures.get(page, ())
            ],
        }
        for hit in hits
    ]


def answer(manual, question, k=TOP_K):
    """One question, shaped for the UI: the prose plus what it was built from.

    The answer sends the reader nowhere, so each fragment travels with its own
    text: that is the evidence, and the interface shows it under the answer.
    The page rides along as data for the interface to label it with, never as
    something the reader is told to go and look up.
    """
    result = ask(manual, question, k)
    return {
        "answer": result["answer"],
        "model": result["model"],
        "tokens": result["usage"].get("totalTokenCount"),
        "hits": shaped(manual, result["hits"]),
    }


class Handler(BaseHTTPRequestHandler):
    """Routes, and the translation of this project's errors into statuses."""

    protocol_version = "HTTP/1.1"
    server_version = "crag"

    def do_GET(self):
        path = self.path.partition("?")[0]
        if path == "/api/catalog":
            return self.send_json(HTTPStatus.OK, catalog())
        if path.startswith("/api/figure/"):
            return self.send_figure(path[len("/api/figure/") :])
        if path.startswith("/api/"):
            return self.send_json(HTTPStatus.NOT_FOUND, {"error": f"no route {path}"})
        return self.send_file(path)

    def do_POST(self):
        # The body is read before the route is checked. Answering without
        # reading it leaves it in the socket, where the next request on a
        # keep-alive connection starts -- and the JSON is then parsed as a
        # request line.
        try:
            body = self.read_json()
        except ValueError:
            return self.send_json(HTTPStatus.BAD_REQUEST, {"error": "expected a JSON body"})

        route = self.path.partition("?")[0]
        if route not in ("/api/ask", "/api/ask/stream"):
            return self.send_json(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})

        try:
            manual, question, k = body["manual"], body["question"], int(body.get("k") or TOP_K)
        except (KeyError, TypeError, ValueError):
            return self.send_json(
                HTTPStatus.BAD_REQUEST, {"error": "expected {manual, question} as JSON"}
            )

        if route == "/api/ask/stream":
            return self.send_stream(manual, question, k)

        try:
            payload = answer(manual, question, k)
        except LookupError as e:
            # A manual that is not indexed, or a setting that is not set: the
            # message names the command to run, so it is the whole answer.
            return self.send_json(HTTPStatus.NOT_FOUND, {"error": str(e)})
        except RuntimeError as e:
            return self.send_json(HTTPStatus.BAD_GATEWAY, {"error": str(e)})
        return self.send_json(HTTPStatus.OK, payload)

    def send_stream(self, manual, question, k):
        """Relay ask_stream as newline-delimited JSON, one event per line.

        The first event is pulled before any header goes out, and that ordering
        is the whole trick: search() raises there if the manual is not indexed,
        so the request can still be refused with a status code. After the first
        line the reply has begun and a failure can only be reported inside it,
        as an 'error' event.
        """
        events = ask_stream(manual, question, k)
        try:
            first = next(events)
        except LookupError as e:
            return self.send_json(HTTPStatus.NOT_FOUND, {"error": str(e)})
        except RuntimeError as e:
            return self.send_json(HTTPStatus.BAD_GATEWAY, {"error": str(e)})

        first["hits"] = shaped(manual, first["hits"])
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        # Nothing between here and the reader may hold a line back waiting for
        # more: buffering the stream would undo the reason it exists.
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            self.send_event(first)
            for event in events:
                self.send_event(event)
        except (LookupError, RuntimeError) as e:
            self.send_event({"event": "error", "error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            return  # the reader left; there is nobody to tell
        self.wfile.write(b"0\r\n\r\n")

    def send_event(self, event):
        """One NDJSON line, in its own chunk so it leaves immediately."""
        line = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
        self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
        self.wfile.flush()

    def send_figure(self, rest):
        """One extracted figure, addressed as <manual_id>/<file>."""
        manual_id, _, name = rest.partition("/")
        target = (FIGURE_DIR / manual_id / name).resolve()
        if not (target.is_file() and target.is_relative_to(FIGURE_DIR.resolve())):
            return self.send_json(
                HTTPStatus.NOT_FOUND, {"error": f"no figure {rest} -- run crag-figures"}
            )
        self.respond(HTTPStatus.OK, "image/jpeg", target.read_bytes())

    def read_json(self):
        """The request body, refusing anything too big to be a question."""
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            # Refusing means not reading it, which leaves the unread body in
            # the socket where the next request should start. On a keep-alive
            # connection that body would be parsed as the next request, so the
            # connection has to end with this response.
            self.close_connection = True
            raise ValueError("body too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def send_json(self, status, payload):
        self.respond(status, "application/json; charset=utf-8", json.dumps(payload).encode())

    def send_file(self, path):
        """A file from the built UI, with unknown paths falling back to index.

        The UI is one page, so anything that is not a file on disk is a route
        inside it and gets the same HTML.
        """
        target = (DIST / path.lstrip("/")).resolve()
        if not (target.is_file() and target.is_relative_to(DIST.resolve())):
            target = DIST / "index.html"
        if not target.is_file():
            return self.send_json(
                HTTPStatus.NOT_FOUND, {"error": "UI not built -- run: pnpm --dir web run build"}
            )
        kind = TYPES.get(target.suffix, "application/octet-stream")
        self.respond(HTTPStatus.OK, kind, target.read_bytes())

    def respond(self, status, kind, body):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host="127.0.0.1", port=8000):  # pragma: no cover - blocks
    """Run until interrupted. Threaded: one slow answer must not block the UI."""
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"crag-serve on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main():  # pragma: no cover - argparse
    """Serve the API, and the built UI if there is one."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
