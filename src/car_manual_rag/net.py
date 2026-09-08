"""One HTTP retry policy, shared by everything that leaves the machine.

Downloading a PDF and calling an API are different jobs, but they fail the same
way -- rate limits, gateway errors, a connection that drops -- and there should
be one answer to that, not one per caller.

Two callers read differently, so there are two ways in. fetch() buffers the
whole body and can start over from nothing at any point, which is what a 6 MB
PDF wants. opened() hands back the response still open, for a reply consumed as
it arrives, and there retrying stops the moment the response exists: bytes
already shown to a reader cannot be unshown. What counts as retryable, and for
how long, is decided in one place for both.
"""

import time
import urllib.error
import urllib.request

RETRIES = 4
RETRY_CODES = (429, 500, 502, 503, 504)  # a 400 or a 401 will not improve


def retry_wait(error, attempt, delay, retries, asked_wait):
    """The pause before another attempt, or None when this failure is final.

    A rate-limited server says how long to wait, and honouring it matters:
    guessing a shorter wait just earns another 429. Not everyone says it in the
    Retry-After header -- Gemini puts it in the error body -- so 'asked_wait'
    lets the caller read it from wherever that API keeps it. Failing both, the
    wait doubles.
    """
    if attempt == retries:
        return None
    if isinstance(error, urllib.error.HTTPError):
        if error.code not in RETRY_CODES:
            return None
        return (asked_wait(error) if asked_wait else None) or float(
            error.headers.get("Retry-After") or delay
        )
    return delay


def opened(request, timeout, retries=RETRIES, note=None, asked_wait=None):
    """The response, still open, for a caller that reads it as it arrives.

    Only the attempt to open is retried. Once the response exists its first
    bytes may already have reached whoever asked, and starting over would
    replay them.
    """
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.URLError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError):
                e.body = e.read()  # read once: the raiser needs it too
            wait = retry_wait(e, attempt, delay, retries, asked_wait)
            if wait is None:
                raise
            if note:
                note(f"    {_reason(e)}, retry {attempt} in {wait:.0f}s")
            time.sleep(wait)
            delay *= 2


def _reason(error):
    """How a failure is named in the progress line."""
    return f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError) else str(error)


def fetch(request, timeout, retries=RETRIES, note=None, asked_wait=None):
    """Return (headers, body), retrying the failures worth retrying.

    Unlike opened(), this retries around the read as well: nothing has been
    handed to the caller until the whole body is in hand, so a connection that
    drops halfway simply starts again. 'note' is called with a line of
    progress, so a long stall is visible rather than silent.
    """
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.headers, response.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError):
                e.body = e.read()  # read once: the raiser needs it too
            wait = retry_wait(e, attempt, delay, retries, asked_wait)
            if wait is None:
                raise
            if note:
                note(f"    {_reason(e)}, retry {attempt} in {wait:.0f}s")
            time.sleep(wait)
            delay *= 2
