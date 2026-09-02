"""Embed one manual's chunks and search them, one .npz per manual.

    data/interim/chunks/SEAT_Ibiza_11.25.jsonl
    -> data/processed/index/SEAT_Ibiza_11.25.npz

Because the user filters down to a single manual before asking, a search never
spans manuals: the largest index in the corpus is 1,107 chunks, a 3 MB matrix.
That is small enough that the whole search is a dot product against an array
held in memory -- no vector database, no metadata filtering, no approximate
neighbours. Picking the manual *is* the filter.

It also means the corpus does not have to be embedded up front: only the
manuals somebody actually opens need an index. Building one is crag-index's
job and nobody else's -- asking a question never spends money on embedding a
whole manual as a side effect, so an unindexed manual is an error that names
the command to fix it.

Vectors are stored normalised, which makes cosine similarity a plain dot
product, and beside the model that produced them -- vectors from two models are
meaningless to compare, so a changed model forces a rebuild.

The freshness this guarantees stops at the chunks: it detects an index built
from different chunks, not chunks built from stale extracted text. The earlier
stages skip on a file existing, so re-running them needs --force.
"""
import argparse
import hashlib
import json
import sys
import time

import numpy as np

from car_manual_rag.config import CHUNK_DIR, EMBEDDING, INDEX_DIR, required
from car_manual_rag.embed import embed

TOP_K = 5


def vectors_of(texts, kind, note=None):
    """Embed and normalise, so that a dot product is the cosine similarity."""
    raw = np.array(embed(texts, kind, note), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-12)


def digest_of(chunks):
    """A fingerprint of the chunk texts.

    The chunk ids are positional, so re-chunking can change every text while
    leaving the ids identical -- an index checked by id alone would be reused
    against text it was not built from, and cite the wrong page. The digest
    changes whenever any character does.
    """
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk["text"].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def index_path(manual_id, index_dir=INDEX_DIR):
    """Where a manual's vectors live -- the naming rule, in one place."""
    return index_dir / f"{manual_id}.npz"


def chunks_of(manual_id, chunk_dir=CHUNK_DIR):
    """The manual's chunks, in file order -- the order the index rows follow."""
    path = chunk_dir / f"{manual_id}.jsonl"
    if not path.is_file():
        raise LookupError(f"no chunks for {manual_id} -- run crag-chunk")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build(manual_id, chunks=None, index_dir=INDEX_DIR, chunk_dir=CHUNK_DIR, note=None):
    """Embed a manual, cache the vectors, and return them with the model used.

    Takes the chunks when the caller already has them parsed, so a cache miss
    does not read and hash the same file twice.
    """
    chunks = chunks_of(manual_id, chunk_dir) if chunks is None else chunks
    model = required(EMBEDDING)
    vectors = vectors_of([c["text"] for c in chunks], "passage", note)

    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_path(manual_id, index_dir)
    tmp = path.with_suffix(".part.npz")
    try:
        np.savez(tmp, vectors=vectors, digest=np.array(digest_of(chunks)),
                 model=np.array(model))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)    # never leave a half-written index
        raise
    return vectors, model


def load(manual_id, index_dir=INDEX_DIR, chunk_dir=CHUNK_DIR):
    """The manual's vectors and chunks, or an error naming the fix.

    Never builds. An index that does not match the chunks or the model it
    claims is refused rather than used -- citing a page from vectors built off
    other text is the one failure this whole design exists to prevent -- and
    each refusal says which command repairs it.
    """
    path = index_path(manual_id, index_dir)
    chunks = chunks_of(manual_id, chunk_dir)
    rebuild = f"run: crag-index {manual_id} --force"

    if not path.is_file():
        raise LookupError(f"{manual_id} is not indexed -- run: crag-index {manual_id}")

    cached = np.load(path, allow_pickle=False)
    # A file that is not one of ours has no business being trusted, so the
    # fields are checked before they are read.
    if not {"vectors", "model", "digest"} <= set(cached.files):
        raise LookupError(f"{path.name} is not an index this version wrote -- {rebuild}")

    model = required(EMBEDDING)
    if str(cached["model"]) != model:
        raise LookupError(f"{manual_id} was indexed with {cached['model']}, "
                          f"not {model} -- {rebuild}")
    if str(cached["digest"]) != digest_of(chunks):
        raise LookupError(f"{manual_id} was indexed from different chunks -- {rebuild}")

    return cached["vectors"], chunks


def search(manual_id, question, k=TOP_K):
    """The k chunks of one manual closest to the question, best first."""
    vectors, chunks = load(manual_id)
    scores = vectors @ vectors_of([question], "query")[0]
    best = np.argsort(scores)[::-1][:k]
    return [dict(chunks[i], score=float(scores[i])) for i in best]


def cite(chunk):
    """How a chunk is referred to in an answer: the page the reader can see.

    ask.SYSTEM tells the model to reproduce the '(pag. N)' this produces, so
    the two have to change together.
    """
    pages = chunk["printed"] or [str(p) for p in chunk["pages"]]
    return f"{chunk['section'] or 'sin seccion'}, pag. {'-'.join(pages)}"


def main():
    """Build an index or try a search; 0 = nothing failed."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manual", help="manual id, from crag-catalog --resolve")
    p.add_argument("--search", metavar="QUESTION", help="ask the manual a question")
    p.add_argument("-k", type=int, default=TOP_K, help="chunks to return")
    p.add_argument("--force", action="store_true", help="rebuild an existing index")
    args = p.parse_args()

    try:
        if args.search:
            started = time.time()
            for hit in search(args.manual, args.search, args.k):
                print(f"\n[{hit['score']:.3f}] {cite(hit)}  {hit['chunk_id']}")
                print(f"  {hit['text'][:300]}...")
            print(f"\nsearched in {time.time() - started:.1f}s")
            return 0

        path = index_path(args.manual)
        if path.is_file() and not args.force:
            print(f"{args.manual} already indexed -> {path} (use --force to rebuild)")
            return 0
        started = time.time()
        vectors, model = build(args.manual, note=print)
        print(f"{args.manual}: {len(vectors)} chunks, {vectors.shape[1]} dims, "
              f"{model}, {time.time() - started:.1f}s")
    except (LookupError, RuntimeError) as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
