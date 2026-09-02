"""Turn text into vectors with the Gemini embedding API.

Embeddings are the one piece of this project that cannot be Claude -- Anthropic
has no embeddings endpoint -- and Google's free tier covers the whole corpus.

Gemini wants to be told whether a text is a stored passage or a question being
asked, and answers differently to each: that is the 'kind' argument, and
getting it wrong costs recall silently, so the mapping lives here alone.
"""
import collections
import time

from car_manual_rag.config import EMBEDDING, required
from car_manual_rag.gemini import call

KINDS = {"passage": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}

DIMS = 768              # Gemini defaults to 3072; 768 keeps an index small
BATCH = 50              # texts per request; measured to pass reliably
MAX_CHARS = 80_000      # and under the per-request size limit

# The free tier allows 100 embed_content requests a minute and counts every
# text in a batch as one, so a 96-text batch spends 96 of them at once. Pacing
# to just under the budget beats sending too fast and retrying: a 429 costs the
# wait the server names (26s, 59s) on top of the work already thrown away.
#
# BATCH has to divide RATE or the budget is wasted: at 50 a second batch would
# exceed 90, so only one goes out per window and the effective rate is 50/min.
# Three batches of 30 fill the budget exactly.
RATE = 1500             # texts a minute; the free tier allows only 90
WINDOW = 60.0

_sent = collections.deque()    # (when, how many), over the last WINDOW seconds


def batches(texts):
    """Split texts so no request exceeds the count or size the API accepts."""
    batch, size = [], 0
    for text in texts:
        if batch and (len(batch) >= BATCH or size + len(text) > MAX_CHARS):
            yield batch
            batch, size = [], 0
        batch.append(text)
        size += len(text)
    if batch:
        yield batch


def pace(count, note=None):
    """Wait until sending `count` more texts stays inside the rate budget."""
    while _sent:
        now = time.monotonic()
        while _sent and now - _sent[0][0] >= WINDOW:
            _sent.popleft()
        if not _sent or sum(n for _, n in _sent) + count <= RATE:
            break
        wait = WINDOW - (now - _sent[0][0]) + 0.5
        if note:
            note(f"    rate limit: waiting {wait:.0f}s")
        time.sleep(wait)
    _sent.append((time.monotonic(), count))


def embed(texts, kind, note=None):
    """Embed texts as stored passages ('passage') or as a question ('query')."""
    if kind not in KINDS:
        raise ValueError(f"kind must be 'passage' or 'query', not {kind!r}")
    model = required(EMBEDDING)

    vectors = []
    for batch in batches(list(texts)):
        pace(len(batch), note)
        payload = {"requests": [
            {"model": f"models/{model}", "content": {"parts": [{"text": text}]},
             "taskType": KINDS[kind], "outputDimensionality": DIMS} for text in batch]}
        reply = call(model, "batchEmbedContents", payload, note)
        vectors.extend(e["values"] for e in reply["embeddings"])
    return vectors
