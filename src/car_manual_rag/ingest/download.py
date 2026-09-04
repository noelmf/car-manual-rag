"""Download the catalogue PDFs (manuals.json) into data/raw/pdf/."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from car_manual_rag.config import CATALOG, PDF_DIR
from car_manual_rag.ingest.catalog import manual_id

PAUSE = 1.0  # seconds between downloads
RETRIES = 4
UA = "Mozilla/5.0 (compatible; car-manual-rag/0.1)"


def download(url, path):
    """Return the number of bytes written."""
    delay = 2.0
    for attempt in range(1, RETRIES + 1):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                content_type = r.headers.get("Content-Type", "")
                data = r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < RETRIES:
                wait = float(e.headers.get("Retry-After") or delay)
                print(f"    HTTP {e.code}, retry {attempt} in {wait:.0f}s")
                time.sleep(wait)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < RETRIES:
                print(f"    {e}, retry {attempt} in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            raise

        if "pdf" not in content_type.lower():
            raise ValueError(f"not a PDF (Content-Type: {content_type!r})")
        if not data.startswith(b"%PDF"):
            raise ValueError("content does not start with %PDF")

        # Atomic write: an interrupted run must not leave a truncated PDF
        # behind that the next run would happily skip.
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(path)
        return len(data)


def main():  # pragma: no cover - argparse and printing
    """Run the download; return a shell exit code (0 = every PDF fetched)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, help="download only the first N manuals")
    p.add_argument("--pause", type=float, default=PAUSE, help="seconds to wait between downloads")
    p.add_argument("--dest", type=Path, default=PDF_DIR, help="output directory")
    args = p.parse_args()

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    args.dest.mkdir(parents=True, exist_ok=True)

    with_pdf = [m for m in catalog if m.get("url")]
    without_pdf = len(catalog) - len(with_pdf)

    # The key must identify exactly one PDF, otherwise files would overwrite
    # each other silently.
    keys = {}
    for m in with_pdf:
        keys.setdefault(manual_id(m), []).append(m)
    clashes = {k: v for k, v in keys.items() if len(v) > 1}
    if clashes:
        print(f"ERROR: {len(clashes)} duplicate keys, they do not identify a single PDF:")
        for k, v in list(clashes.items())[:5]:
            print(f"  {k}")
            for m in v:
                print(f"     -> {m['url']}")
        return 1

    pending = with_pdf[: args.limit] if args.limit else with_pdf
    print(
        f"{len(catalog)} manuals in the catalogue, {without_pdf} without a PDF, "
        f"{len(pending)} to process -> {args.dest}"
    )

    downloaded = skipped = 0
    failures = []
    total_bytes = 0
    started = time.time()

    for i, m in enumerate(pending, 1):
        path = args.dest / f"{manual_id(m)}.pdf"
        label = f"[{i}/{len(pending)}] {path.name}"
        if path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        try:
            n = download(m["url"], path)
        except Exception as e:
            print(f"{label}  FAILED: {e}")
            failures.append((path.name, m["url"], str(e)))
        else:
            downloaded += 1
            total_bytes += n
            print(f"{label}  {n / 1e6:.1f} MB")
        time.sleep(args.pause)

    elapsed = time.time() - started
    print(
        f"\nDownloaded {downloaded} ({total_bytes / 1e9:.2f} GB) in {elapsed / 60:.1f} min, "
        f"{skipped} already present, {len(failures)} failed"
    )
    for name, url, err in failures:
        print(f"  {name}: {err}\n    {url}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
