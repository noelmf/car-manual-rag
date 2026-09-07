"""Cut every figure out of the manuals, one JPEG per figure.

    data/raw/pdf/SEAT_Ibiza_11.25.pdf
    -> data/interim/figures/SEAT_Ibiza_11.25/
       index.jsonl                {"figure_id": "...:0017:1", "page": 17, ...}
       p0017_1.jpg

A figure is an image the manual placed on a page. The page is rendered inside
that image's rectangle rather than the stored bitmap being pulled out, and both
halves of that sentence were learned the hard way:

  * the callout numbers are drawn over the picture as vectors, not baked into
    it. A seat diagram extracted straight from the PDF comes out with no 1, 2
    or 3 on it, which is the one thing a reader needs to match the drawing to
    the text;
  * the bitmap is stored in whatever orientation the typesetter had, and the
    page's matrix is what turns it upright. Extracted raw, a car appears lying
    on its side;
  * and 98% of the stored bitmaps are CMYK JPEGs, which browsers do not render.
    Rendering a page gives RGB for free.

Small images are dropped: rules, spacers and 1x1 pixels are images too. The
threshold sits in the gap between the two populations -- across the corpus a
cut anywhere from 5,000 to 20,000 square pixels discards the same 8% of images,
because almost nothing is that size. Real figures start around 190x190.

Extraction is CPU-bound, so manuals are processed in parallel. Resumable: the
index is written last and a manual with one is skipped, so a run killed halfway
leaves a manual without an index and it is simply done again.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

from car_manual_rag.config import FIGURE_DIR, PDF_DIR

MIN_AREA = 10_000  # square pixels; below this it is a rule or a spacer
DPI = 150  # about the native resolution of these bitmaps: no up- or downscaling
QUALITY = 85
INDEX = "index.jsonl"


def placements(page):
    """The figures on one page, as (bbox, width, height) in reading order.

    Keyed by rectangle rather than by image: the same drawing can be placed
    twice on a page, and each placement is its own figure. A rectangle the
    renderer cannot resolve is skipped rather than guessed at.
    """
    found = {}
    for image in page.get_images(full=True):
        width, height = image[2], image[3]
        if width * height < MIN_AREA:
            continue
        try:
            bbox = page.get_image_bbox(image)
        except (ValueError, RuntimeError):
            continue
        if bbox.is_empty or bbox.is_infinite or bbox.get_area() <= 0:
            continue
        found.setdefault(tuple(round(v, 2) for v in bbox), (bbox, width, height))
    return list(found.values())


def render(page, bbox, dpi, quality):
    """One figure, as the JPEG bytes a browser can show."""
    pixmap = page.get_pixmap(clip=bbox, dpi=dpi)
    return pixmap.tobytes("jpeg", jpg_quality=quality), pixmap.width, pixmap.height


def cut(pdf_path, out_root=FIGURE_DIR, dpi=DPI, quality=QUALITY):
    """Write one manual's figures and its index, and return the run's stats."""
    manual_id = Path(pdf_path).stem
    out_dir = Path(out_root) / manual_id
    records = []
    skipped = 0

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(pdf_path) as doc:
            for number, page in enumerate(doc, 1):
                for n, (bbox, width, height) in enumerate(placements(page), 1):
                    # One figure the decoder chokes on -- a JPEG whose height
                    # only arrives in a DNL marker, say -- must not cost the
                    # other three hundred in the manual.
                    try:
                        data, out_w, out_h = render(page, bbox, dpi, quality)
                    except Exception:
                        skipped += 1
                        continue
                    name = f"p{number:04d}_{n}.jpg"
                    (out_dir / name).write_bytes(data)
                    records.append(
                        {
                            "figure_id": f"{manual_id}:{number:04d}:{n}",
                            "manual_id": manual_id,
                            "page": number,
                            "n": n,
                            "file": name,
                            "bbox": [round(v, 2) for v in bbox],
                            "width": out_w,
                            "height": out_h,
                            "source_px": [width, height],
                        }
                    )
    except Exception as e:
        return {"name": manual_id, "error": str(e)}

    # The index goes last and in one move: its presence is what says this
    # manual is done, so it must never exist beside a half-written folder.
    tmp = out_dir / (INDEX + ".part")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    tmp.replace(out_dir / INDEX)

    return {
        "name": manual_id,
        "figures": len(records),
        "skipped": skipped,
        "pages": len({r["page"] for r in records}),
        "bytes": sum((out_dir / r["file"]).stat().st_size for r in records),
    }


def load(manual_id, out_root=FIGURE_DIR):
    """One manual's figures, or an error naming the command that makes them."""
    path = Path(out_root) / manual_id / INDEX
    if not path.is_file():
        raise LookupError(f"no figures for {manual_id} -- run crag-figures")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def on_pages(manual_id, pages, out_root=FIGURE_DIR):
    """The figures printed on the given pages, in page order.

    This is how a fragment finds its drawings: a chunk knows the pages it came
    from, and the figures beside that text are the ones on those pages.
    """
    wanted = set(pages)
    return [f for f in load(manual_id, out_root) if f["page"] in wanted]


def report(stats):
    """Print a validation summary and return True if anything looks wrong."""
    failed = [s for s in stats if "error" in s]
    ok = [s for s in stats if "error" not in s]
    figures = sum(s["figures"] for s in ok)
    size = sum(s["bytes"] for s in ok)
    skipped = sum(s["skipped"] for s in ok)

    print(
        f"\n{len(ok)} manuals, {figures:,} figures, {size / 1e9:.2f} GB "
        f"({size / max(figures, 1) / 1024:.0f} KB each), {skipped:,} undecodable"
    )

    # A manual with no figures at all, or with one per page, has not been read
    # the way the rest were: both are worth a look before trusting the run.
    empty = [s for s in ok if not s["figures"]]
    lossy = [s for s in ok if s["figures"] and s["skipped"] / (s["figures"] + s["skipped"]) > 0.1]
    crowded = [s for s in ok if s["pages"] and s["figures"] / s["pages"] > 4]

    for label, group in (
        ("failed to open", failed),
        ("no figures at all", empty),
        ("over 10% of figures undecodable", lossy),
        ("over 4 figures per page", crowded),
    ):
        print(f"  {label}: {len(group)}")
        for s in group[:10]:
            print(f"      {s['name']}{' - ' + s['error'] if 'error' in s else ''}")
        if len(group) > 10:
            print(f"      ... and {len(group) - 10} more")

    return bool(failed or empty or lossy)


def main():  # pragma: no cover - argparse and printing
    """Cut the figures out of every manual; 0 = nothing looked wrong."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", type=Path, default=PDF_DIR, help="directory of PDFs")
    p.add_argument("--dest", type=Path, default=FIGURE_DIR, help="output directory")
    p.add_argument("--dpi", type=int, default=DPI, help="render resolution")
    p.add_argument("--quality", type=int, default=QUALITY, help="JPEG quality")
    p.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    p.add_argument("--limit", type=int, help="only the first N manuals")
    p.add_argument("--force", action="store_true", help="redo manuals already done")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.source.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {args.source} -- run crag-download first")
        return 1

    pending = (
        pdfs if args.force else [f for f in pdfs if not (args.dest / f.stem / INDEX).is_file()]
    )
    skipped = len(pdfs) - len(pending)
    if args.limit:
        pending = pending[: args.limit]

    print(
        f"{len(pdfs)} PDFs in {args.source}, {skipped} already done, "
        f"{len(pending)} to process -> {args.dest}"
    )

    stats = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(cut, str(f), str(args.dest), args.dpi, args.quality) for f in pending
        ]
        for i, future in enumerate(futures, 1):
            s = future.result()
            stats.append(s)
            if "error" in s:
                print(f"[{i}/{len(pending)}] {s['name']}  FAILED: {s['error']}")
            else:
                print(
                    f"[{i}/{len(pending)}] {s['name']}  {s['figures']} figures, "
                    f"{s['bytes'] / 1e6:.1f} MB"
                )

    print(f"\nCut {len(stats)} manuals in {(time.time() - started) / 60:.1f} min")
    return 1 if report(stats) else 0


if __name__ == "__main__":
    sys.exit(main())
