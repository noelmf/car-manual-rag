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


class TestGeminiRetryDelay:
    def test_reads_the_wait_gemini_asks_for(self):
        # Gemini puts it in the body, not in Retry-After; missing it meant
        # backing off 2s when the server had asked for 59.
        error = http_error(429)
        error.body = (b'{"error":{"details":[{"@type":"type.googleapis.com/'
                      b'google.rpc.RetryInfo","retryDelay":"59s"}]}}')
        assert gemini.asked_wait(error) == 60.0

    def test_a_body_without_retry_info_gives_nothing(self):
        error = http_error(429)
        error.body = b'{"error":{"message":"nope"}}'
        assert gemini.asked_wait(error) is None

    def test_a_body_that_is_not_json_gives_nothing(self):
        error = http_error(429)
        error.body = b"<html>502</html>"
        assert gemini.asked_wait(error) is None
