"""Answer a question about one manual, citing the page it came from.

    crag-ask SEAT_Ibiza_11.25 "cada cuanto se cambia el aceite"

The user has already narrowed the picker to a single manual, so this asks that
manual and nothing else: index.search() supplies the fragments and the model
only rewrites what they say.

Two things shape the prompt, and both come from what this is for. A driver acts
on the answer, so an invented figure is worse than no answer -- the model is
told to say when the fragments do not cover the question rather than fill the
gap. And a manual's warnings are the part a paraphrase most easily softens, so
they are to be carried over intact.

The answer does not send the reader to a page. It cannot: the fragments it was
built from are shown underneath it in full, so the evidence is already on the
screen. A page number would be an instruction to go and look somewhere that
nobody has open, and the manual's own cross-references are dropped at chunking
for the same reason. The pages stay in each fragment's record, for an interface
that wants to show where a passage came from.
"""

import argparse
import sys
import time

from car_manual_rag.config import MODEL, required
from car_manual_rag.gemini import call
from car_manual_rag.index import TOP_K, label, search

# The ceiling covers the model's thinking as well as its answer, and a
# reasoning model spends most of it before writing a word: at 1024 this task
# left 41 tokens for the answer and cut it off mid-procedure. Only what is
# generated is billed, so the ceiling is set where no answer reaches it.
MAX_TOKENS = 4096

SYSTEM = """You answer questions about one car's owner manual.

Rules:
- Answer only with what the given fragments say. Do not use general knowledge
  about cars or about other models.
- If the fragments do not contain the answer, say so plainly and do not invent
  one. Saying nothing beats saying something wrong: the reader is going to act
  on this.
- Do not send the reader anywhere. No page numbers, no "see the section on
  X": the fragments are printed under your answer, so say what they say.
- If a fragment carries a safety warning, reproduce it; do not summarise it or
  soften it.
- Write for someone standing next to the car: plain sentences, and a plain
  list when the manual gives steps. No headings, no bold, and no preamble about
  the manual or the fragments -- begin with the answer itself.
- Answer in the language of the question, briefly and directly.
"""


def prompt(question, hits):
    """The fragments and the question, as the model sees them."""
    fragments = "\n\n".join(
        f"--- Fragment {i} ({label(hit)}) ---\n{hit['text']}" for i, hit in enumerate(hits, 1)
    )
    return f"{fragments}\n\n--- Question ---\n{question}"


def seen(hits):
    """The hits with any paragraph already shown by an earlier one removed."""
    shown = set()
    for hit in hits:
        kept = [p for p in hit["text"].split("\n") if p not in shown]
        shown.update(kept)
        if kept:
            yield dict(hit, text="\n".join(kept))


def ask(manual_id, question, k=TOP_K):
    """Search the manual and answer from what comes back."""
    model = required(MODEL)
    hits = search(manual_id, question, k)  # never builds; see index.load
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt(question, hits)}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        # Zero temperature: the answer should be what the manual says, and the
        # same question twice should not give two different figures.
        "generationConfig": {"temperature": 0, "maxOutputTokens": MAX_TOKENS},
    }
    reply = call(model, "generateContent", payload)

    candidates = reply.get("candidates") or []
    if not candidates:
        # A blocked or empty reply is not an answer; say why instead of ''.
        raise RuntimeError(f"no answer from {model}: {reply.get('promptFeedback', reply)}")

    finish = candidates[0].get("finishReason")
    if finish and finish != "STOP":
        # Half a procedure reads exactly like a whole one, which is the failure
        # this project exists to avoid. An answer that stopped early is not an
        # answer, so it is refused rather than printed.
        raise RuntimeError(
            f"{model} stopped early ({finish}); the answer would be incomplete. "
            f"MAX_TOKENS is {MAX_TOKENS}, and it covers the model's thinking too"
        )

    parts = candidates[0].get("content", {}).get("parts") or []
    return {
        "answer": "".join(p.get("text", "") for p in parts).strip(),
        "hits": hits,
        "model": model,
        "usage": reply.get("usageMetadata", {}),
    }


def main():  # pragma: no cover - argparse and printing
    """Ask one manual a question; 0 = it answered."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("manual", help="manual id, from crag-catalog --resolve")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=TOP_K, help="fragments to retrieve")
    args = p.parse_args()

    try:
        started = time.time()
        result = ask(args.manual, args.question, args.k)
    except (LookupError, RuntimeError) as e:
        print(e)
        return 1

    print(result["answer"])
    print(
        f"\n-- {result['model']}, {len(result['hits'])} fragmentos, "
        f"{result['usage'].get('totalTokenCount', '?')} tokens, "
        f"{time.time() - started:.1f}s --"
    )
    # The fragments themselves, not a list of pages to go and look up. Printed
    # once each: the chunker repeats a short paragraph across the seam between
    # two chunks, and the reader should not read it twice.
    for hit in seen(result["hits"]):
        print(f"\n[{hit['score']:.3f}] {label(hit)}")
        print(hit["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
