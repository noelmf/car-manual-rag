"""Retries only run when something is already wrong, so nobody tests them by hand."""

import urllib.error

import pytest

from car_manual_rag import gemini, net


class FakeResponse:
    def __init__(self, body):
        self.headers, self._body = {}, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, headers=None, body=b""):
    return urllib.error.HTTPError("http://x", code, "", headers or {}, None)


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    slept = []
    monkeypatch.setattr(net.time, "sleep", slept.append)
    return slept


def responder(monkeypatch, outcomes):
    """Make urlopen return or raise each outcome in turn."""
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        result = outcomes[len(calls) - 1]
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)

    monkeypatch.setattr(net.urllib.request, "urlopen", fake_urlopen)
    return calls


class TestFetch:
    def test_returns_the_body_on_success(self, monkeypatch):
        responder(monkeypatch, [b"hola"])
        assert net.fetch("req", 10)[1] == b"hola"

    def test_retries_a_rate_limit_then_succeeds(self, monkeypatch, no_sleeping):
        calls = responder(monkeypatch, [http_error(429), b"ok"])
        assert net.fetch("req", 10)[1] == b"ok"
        assert len(calls) == 2

    def test_does_not_retry_a_bad_request(self, monkeypatch):
        calls = responder(monkeypatch, [http_error(400), b"nunca"])
        with pytest.raises(urllib.error.HTTPError):
            net.fetch("req", 10)
        assert len(calls) == 1

    def test_gives_up_after_the_last_attempt(self, monkeypatch, no_sleeping):
        calls = responder(monkeypatch, [http_error(503)] * 4)
        with pytest.raises(urllib.error.HTTPError):
            net.fetch("req", 10, retries=4)
        assert len(calls) == 4

    def test_honours_the_retry_after_header(self, monkeypatch, no_sleeping):
        responder(monkeypatch, [http_error(429, {"Retry-After": "30"}), b"ok"])
        net.fetch("req", 10)
        assert no_sleeping == [30.0]

    def test_backs_off_by_doubling_when_nobody_says_otherwise(self, monkeypatch, no_sleeping):
        responder(monkeypatch, [http_error(500), http_error(500), b"ok"])
        net.fetch("req", 10)
        assert no_sleeping == [2.0, 4.0]

    def test_the_body_survives_for_the_caller_to_report(self, monkeypatch):
        responder(monkeypatch, [http_error(400)])
        with pytest.raises(urllib.error.HTTPError) as e:
            net.fetch("req", 10)
        assert hasattr(e.value, "body")


class TestOpened:
    """The streaming way in: retried until the response exists, never after."""

    def test_hands_back_the_response_itself_unread(self, monkeypatch):
        responder(monkeypatch, [b"hola"])
        response = net.opened("req", 10)
        assert response.read() == b"hola"  # still the caller's to read

    def test_retries_a_rate_limit_then_succeeds(self, monkeypatch, no_sleeping):
        calls = responder(monkeypatch, [http_error(429), b"ok"])
        assert net.opened("req", 10).read() == b"ok"
        assert len(calls) == 2

    def test_does_not_retry_a_bad_request(self, monkeypatch):
        calls = responder(monkeypatch, [http_error(400)])
        with pytest.raises(urllib.error.HTTPError):
            net.opened("req", 10)
        assert len(calls) == 1

    def test_gives_up_after_the_last_attempt(self, monkeypatch, no_sleeping):
        calls = responder(monkeypatch, [http_error(503)] * net.RETRIES)
        with pytest.raises(urllib.error.HTTPError):
            net.opened("req", 10)
        assert len(calls) == net.RETRIES

    def test_retries_a_refused_connection(self, monkeypatch, no_sleeping):
        calls = responder(monkeypatch, [urllib.error.URLError("refused"), b"ok"])
        assert net.opened("req", 10).read() == b"ok"
        assert len(calls) == 2

    def test_a_stall_is_visible_rather_than_silent(self, monkeypatch, no_sleeping):
        lines = []
        responder(monkeypatch, [http_error(429), b"ok"])
        net.opened("req", 10, note=lines.append)
        assert "HTTP 429" in lines[0]

    def test_the_body_survives_for_the_caller_to_report(self, monkeypatch):
        error = http_error(400)
        error.read = lambda: b"por que"
        responder(monkeypatch, [error])
        with pytest.raises(urllib.error.HTTPError) as raised:
            net.opened("req", 10)
        assert raised.value.body == b"por que"


class TestFetchWhenTheNetworkGoesAway:
    """A 429 answers; a dropped connection does not, and takes a different path."""

    def test_retries_a_refused_connection(self, monkeypatch, no_sleeping):
        calls = responder(monkeypatch, [urllib.error.URLError("conexion rechazada"), b"ok"])
        assert net.fetch("req", 10)[1] == b"ok"
        assert len(calls) == 2

    def test_retries_a_timeout(self, monkeypatch, no_sleeping):
        responder(monkeypatch, [TimeoutError("se agoto el tiempo"), b"ok"])
        assert net.fetch("req", 10)[1] == b"ok"

    def test_gives_up_on_a_connection_that_never_comes_back(self, monkeypatch, no_sleeping):
        calls = responder(monkeypatch, [urllib.error.URLError("caida")] * 4)
        with pytest.raises(urllib.error.URLError):
            net.fetch("req", 10, retries=4)
        assert len(calls) == 4

    def test_backs_off_by_doubling(self, monkeypatch, no_sleeping):
        # Nobody sends a Retry-After with a dropped socket, so doubling is the
        # only policy left.
        responder(
            monkeypatch,
            [urllib.error.URLError("una"), urllib.error.URLError("otra"), b"ok"],
        )
        net.fetch("req", 10)
        assert no_sleeping == [2.0, 4.0]


class TestProgress:
    """A silent minute-long wait is indistinguishable from a hang."""

    def test_a_rate_limit_says_the_reason_and_the_wait(self, monkeypatch, no_sleeping):
        lines = []
        responder(monkeypatch, [http_error(429, {"Retry-After": "30"}), b"ok"])
        net.fetch("req", 10, note=lines.append)
        assert lines == ["    HTTP 429, retry 1 in 30s"]

    def test_a_network_failure_names_itself(self, monkeypatch, no_sleeping):
        lines = []
        responder(monkeypatch, [urllib.error.URLError("conexion rechazada"), b"ok"])
        net.fetch("req", 10, note=lines.append)
        assert "conexion rechazada" in lines[0]

    def test_nothing_is_printed_when_nothing_fails(self, monkeypatch):
        lines = []
        responder(monkeypatch, [b"ok"])
        net.fetch("req", 10, note=lines.append)
        assert lines == []


class TestGeminiStream:
    """Streaming asks the same endpoint a different way and reads SSE lines."""

    def sse(self, *payloads):
        class FakeStream(FakeResponse):
            def __init__(self, lines):
                super().__init__(b"")
                self._lines = lines

            def __iter__(self):
                return iter(self._lines)

        import json as _json

        return FakeStream(
            [b": comentario\n", b"\n"]
            + [b"data: " + _json.dumps(p).encode() + b"\n" for p in payloads]
        )

    def test_reads_only_the_data_lines(self):
        out = list(gemini.chunks(self.sse({"a": 1}, {"a": 2})))
        assert out == [{"a": 1}, {"a": 2}]

    def test_asks_for_server_sent_events_on_the_streaming_verb(self, monkeypatch):
        seen = {}

        def fake_opened(request, timeout, **kw):
            seen["url"] = request.full_url
            return self.sse({"a": 1})

        monkeypatch.setattr(gemini, "opened", fake_opened)
        monkeypatch.setenv("GEMINI_API_KEY", "clave")
        list(gemini.stream("modelo", "streamGenerateContent", {}))
        assert seen["url"].endswith("/models/modelo:streamGenerateContent?alt=sse")

    def test_the_connection_opens_before_a_single_chunk_is_asked_for(self, monkeypatch):
        # A caller relaying this has to learn of a refusal while it can still
        # answer with a status code, so the failure cannot wait for iteration.
        def refuses(request, timeout, **kw):
            error = http_error(400)
            error.body = b"malo"
            raise error

        monkeypatch.setattr(gemini, "opened", refuses)
        monkeypatch.setenv("GEMINI_API_KEY", "clave")
        with pytest.raises(RuntimeError, match="malo"):
            gemini.stream("modelo", "streamGenerateContent", {})

    def test_the_error_names_the_code_gemini_gave(self, monkeypatch):
        def refuses(request, timeout, **kw):
            error = http_error(429)
            error.body = b"despacio"
            raise error

        monkeypatch.setattr(gemini, "opened", refuses)
        monkeypatch.setenv("GEMINI_API_KEY", "clave")
        with pytest.raises(RuntimeError, match="429"):
            gemini.stream("modelo", "streamGenerateContent", {})


class TestGeminiRetryDelay:
    def test_reads_the_wait_gemini_asks_for(self):
        # Gemini puts it in the body, not in Retry-After; missing it meant
        # backing off 2s when the server had asked for 59.
        error = http_error(429)
        error.body = (
            b'{"error":{"details":[{"@type":"type.googleapis.com/'
            b'google.rpc.RetryInfo","retryDelay":"59s"}]}}'
        )
        assert gemini.asked_wait(error) == 60.0

    def test_a_body_without_retry_info_gives_nothing(self):
        error = http_error(429)
        error.body = b'{"error":{"message":"nope"}}'
        assert gemini.asked_wait(error) is None

    def test_a_body_that_is_not_json_gives_nothing(self):
        error = http_error(429)
        error.body = b"<html>502</html>"
        assert gemini.asked_wait(error) is None


class TestGeminiCall:
    @pytest.fixture(autouse=True)
    def key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "clave-secreta")

    def test_builds_the_url_from_the_model_and_the_verb(self, monkeypatch):
        seen = {}

        def fake_fetch(request, timeout, note=None, asked_wait=None):
            seen["url"] = request.full_url
            seen["headers"] = request.headers
            return {}, b'{"ok": true}'

        monkeypatch.setattr(gemini, "fetch", fake_fetch)
        assert gemini.call("un-modelo", "generateContent", {"a": 1}) == {"ok": True}
        assert seen["url"].endswith("/models/un-modelo:generateContent")

    def test_sends_the_key_in_gemini_s_own_header(self, monkeypatch):
        seen = {}

        def fake_fetch(request, timeout, note=None, asked_wait=None):
            seen.update(request.headers)
            return {}, b"{}"

        monkeypatch.setattr(gemini, "fetch", fake_fetch)
        gemini.call("m", "v", {})
        # urllib capitalises header names, so compare case-insensitively.
        assert any(k.lower() == "x-goog-api-key" for k in seen)

    def test_an_error_carries_the_reason_gemini_gave(self, monkeypatch):
        def fake_fetch(request, timeout, note=None, asked_wait=None):
            error = http_error(400)
            error.body = b'{"error":{"message":"model not found"}}'
            raise error

        monkeypatch.setattr(gemini, "fetch", fake_fetch)
        with pytest.raises(RuntimeError, match="model not found"):
            gemini.call("m", "v", {})
