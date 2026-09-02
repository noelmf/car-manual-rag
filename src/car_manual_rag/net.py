"""One HTTP retry policy, shared by everything that leaves the machine.

Downloading a PDF and calling an API are different jobs, but they fail the same
way -- rate limits, gateway errors, a connection that drops -- and there should
be one answer to that, not one per caller.
"""
import time
import urllib.error
import urllib.request

RETRIES = 4
RETRY_CODES = (429, 500, 502, 503, 504)    # a 400 or a 401 will not improve


def fetch(request, timeout, retries=RETRIES, note=None, asked_wait=None):
    """Return (headers, body), retrying the failures worth retrying.

    A rate-limited server says how long to wait, and honouring it matters:
    guessing a shorter wait just earns another 429. Not everyone says it in the
    Retry-After header -- Gemini puts it in the error body -- so 'asked_wait'
    lets the caller read it from wherever that API keeps it. Failing both, the
    wait doubles. 'note' is called with a line of progress, so a long stall is
    visible rather than silent.
    """
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.headers, response.read()
        except urllib.error.HTTPError as e:
            e.body = e.read()          # read once: the raiser needs it too
            if e.code not in RETRY_CODES or attempt == retries:
                raise
            wait = ((asked_wait(e) if asked_wait else None)
                    or float(e.headers.get("Retry-After") or delay))
            reason = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                raise
            wait, reason = delay, str(e)
        if note:
            note(f"    {reason}, retry {attempt} in {wait:.0f}s")
        time.sleep(wait)
        delay *= 2
