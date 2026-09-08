"""Calling Gemini: the endpoint, the key header and the model in the path.

Embedding and generation are two verbs on the same API, so the shape of the
call lives here and both sit on it as siblings -- otherwise the answer path
would import the embedding module for its transport, and be tuned by constants
named after a batch of passages.
"""

import json
import urllib.error
import urllib.request

from car_manual_rag.config import API_KEY, required
from car_manual_rag.net import fetch, opened

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{verb}"
TIMEOUT = 120


def asked_wait(error):
    """The wait Gemini asks for, which it puts in RetryInfo, not Retry-After."""
    try:
        for detail in json.loads(error.body)["error"].get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(detail["retryDelay"].rstrip("s")) + 1
    except (ValueError, KeyError, TypeError):
        return None


def request_for(url, payload):
    """The POST every call makes: the key in Gemini's header, JSON in the body."""
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": required(API_KEY), "Content-Type": "application/json"},
    )


def call(model, verb, payload, note=None):
    """POST a request to one model's endpoint and return the parsed reply."""
    url = URL.format(model=model, verb=verb)
    request = request_for(url, payload)
    try:
        _, body = fetch(request, TIMEOUT, note=note, asked_wait=asked_wait)
    except urllib.error.HTTPError as e:
        # Gemini puts the reason in the body; a bare '400' is not actionable.
        raise RuntimeError(f"{e.code} from {url}: {e.body[:300].decode('utf-8', 'replace')}") from e
    return json.loads(body)


def stream(model, verb, payload, note=None):
    """Open a streaming reply and return its chunks, parsed, as they arrive.

    Asked with alt=sse, Gemini answers in Server-Sent Events: one 'data:' line
    per chunk, each a slice of the same reply shape call() returns whole.

    The connection is opened here rather than inside the generator, and that is
    the point of the split. A caller relaying this to somebody else has to know
    the request was refused -- a 400, an exhausted quota -- while it can still
    answer with an error instead of a half-written page. Once these chunks
    start arriving it is too late to change the subject.
    """
    url = URL.format(model=model, verb=verb) + "?alt=sse"
    try:
        response = opened(request_for(url, payload), TIMEOUT, note=note, asked_wait=asked_wait)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code} from {url}: {e.body[:300].decode('utf-8', 'replace')}") from e
    return chunks(response)


def chunks(response):
    """The parsed 'data:' lines of an SSE response, until it closes."""
    with response:
        for line in response:
            line = line.decode("utf-8").strip()
            if line.startswith("data:"):
                yield json.loads(line[5:])
