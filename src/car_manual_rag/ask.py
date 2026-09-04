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

Every claim carries the page printed on the paper, not the page of the PDF, so
the reader can find it in their own copy.
"""

import argparse
import sys
import time

from car_manual_rag.config import MODEL, required
from car_manual_rag.gemini import call
from car_manual_rag.index import TOP_K, cite, search

MAX_TOKENS = 1024

SYSTEM = """You answer questions about one car's owner manual.

Rules:
- Answer only with what the given fragments say. Do not use general knowledge
  about cars or about other models.
- If the fragments do not contain the answer, say so plainly and do not invent
  one. Saying nothing beats saying something wrong: the reader is going to act
  on this.
- Always cite the page in brackets at the end of each claim, in the (pag. N)
  format that appears in each fragment.
- If a fragment carries a safety warning, reproduce it; do not summarise it or
  soften it.
- Answer in the language of the question, briefly and directly.
"""


def prompt(question, hits):
    """The fragments and the question, as the model sees them."""
    fragments = "\n\n".join(
        f"--- Fragment {i} ({cite(hit)}) ---\n{hit['text']}" for i, hit in enumerate(hits, 1)
    )
    return f"{fragments}\n\n--- Question ---\n{question}"


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
    for hit in result["hits"]:
        print(f"   [{hit['score']:.3f}] {cite(hit)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
