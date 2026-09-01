"""Extract the text layer of every downloaded PDF into JSONL, one page per line.

    data/raw/pdf/SEAT_Ibiza_11.25.pdf
    -> data/interim/text/SEAT_Ibiza_11.25.jsonl
       {"page": 1, "text": "..."}
       {"page": 2, "text": "..."}

The page number is kept from the start because the answers must cite it, and it
cannot be recovered later. The text is stored raw: joining hyphenated words and
dropping the dotted table-of-contents leaders belongs to the chunking stage, so
that cleanup can be reworked without re-reading 2 GB of PDFs.

Extraction is CPU-bound, so files are processed in parallel. Resumable: manuals
already extracted are skipped unless --force is given.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

from car_manual_rag.config import PDF_DIR, TEXT_DIR

EMPTY_PAGE_CHARS = 20    # below this a page is all figure, no readable text
ACCENTED = "áéíóúüñÁÉÍÓÚÜÑ¡¿"


def extract(pdf_path, out_dir):
    """Write one JSONL file for a PDF and return its stats.

    Stats: name, pages, chars, empty pages, accented chars. The caller uses
    them to spot manuals whose text layer came out wrong.
    """
    out_path = Path(out_dir) / (Path(pdf_path).stem + ".jsonl")
    tmp = out_path.with_suffix(".part")
    pages = chars = empty = accents = 0

    try:
        with pymupdf.open(pdf_path) as doc, tmp.open("w", encoding="utf-8") as fh:
            for number, page in enumerate(doc, 1):
                text = page.get_text()
                fh.write(json.dumps({"page": number, "text": text}, ensure_ascii=False) + "\n")
                pages += 1
                chars += len(text)
                empty += len(text.strip()) < EMPTY_PAGE_CHARS
                accents += sum(text.count(c) for c in ACCENTED)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return {"name": Path(pdf_path).name, "error": str(e)}

    tmp.rename(out_path)
    return {"name": out_path.name, "pages": pages, "chars": chars,
            "empty": empty, "accents": accents}


def report(stats):
    """Print a validation summary and return True if anything looks wrong."""
    failed = [s for s in stats if "error" in s]
    ok = [s for s in stats if "error" not in s]
    pages = sum(s["pages"] for s in ok)
    chars = sum(s["chars"] for s in ok)

    print(f"\n{len(ok)} manuals, {pages:,} pages, {chars / 1e6:.1f}M chars "
          f"({chars / max(pages, 1):.0f} chars/page)")

    # A manual whose text layer failed shows up as very little text per page,
    # or as no accented characters at all in a Spanish document.
    nothing = [s for s in ok if not s["pages"] or not s["chars"]]
    thin = [s for s in ok if s["pages"] and s["chars"] / s["pages"] < 200]
    flat = [s for s in ok if s["chars"] and s["accents"] / s["chars"] < 0.002]
    blank = [s for s in ok if s["pages"] and s["empty"] / s["pages"] > 0.2]

    for label, group in (("failed to open", failed),
                         ("no pages or no text at all", nothing),
                         ("under 200 chars/page", thin),
                         ("almost no accented characters", flat),
                         ("over 20% blank pages", blank)):
        print(f"  {label}: {len(group)}")
        shown = group[:10]
        for s in shown:
            print(f"      {s['name']}{' - ' + s['error'] if 'error' in s else ''}")
        if len(group) > len(shown):
            print(f"      ... and {len(group) - len(shown)} more")

    return bool(failed or nothing or thin or flat or blank)


def main():
    """Run the extraction; return a shell exit code (0 = nothing suspicious)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=PDF_DIR, help="directory of PDFs")
    p.add_argument("--dest", type=Path, default=TEXT_DIR, help="output directory")
    p.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    p.add_argument("--limit", type=int, help="extract only the first N manuals")
    p.add_argument("--force", action="store_true", help="re-extract manuals already done")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.source.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {args.source} -- run crag-download first")
        return 1

    pending = pdfs if args.force else [
        f for f in pdfs if not (args.dest / (f.stem + ".jsonl")).exists()
    ]
    skipped = len(pdfs) - len(pending)
    if args.limit:
        pending = pending[: args.limit]

    print(f"{len(pdfs)} PDFs in {args.source}, {skipped} already extracted, "
          f"{len(pending)} to process -> {args.dest}")

    stats = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract, str(f), str(args.dest)) for f in pending]
        for i, future in enumerate(futures, 1):
            s = future.result()
            stats.append(s)
            if "error" in s:
                print(f"[{i}/{len(pending)}] {s['name']}  FAILED: {s['error']}")
            else:
                print(f"[{i}/{len(pending)}] {s['name']}  {s['pages']} pages, "
                      f"{s['chars'] / 1000:.0f}k chars")

    elapsed = time.time() - started
    print(f"\nExtracted {len(stats)} manuals in {elapsed / 60:.1f} min")
    return 1 if report(stats) else 0


if __name__ == "__main__":
    sys.exit(main())
