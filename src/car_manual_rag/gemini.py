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
from car_manual_rag.net import fetch

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


def call(model, verb, payload, note=None):
    """POST a request to one model's endpoint and return the parsed reply."""
    url = URL.format(model=model, verb=verb)
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": required(API_KEY), "Content-Type": "application/json"})
    try:
        _, body = fetch(request, TIMEOUT, note=note, asked_wait=asked_wait)
    except urllib.error.HTTPError as e:
        # Gemini puts the reason in the body; a bare '400' is not actionable.
        raise RuntimeError(f"{e.code} from {url}: "
                           f"{e.body[:300].decode('utf-8', 'replace')}") from e
    return json.loads(body)
