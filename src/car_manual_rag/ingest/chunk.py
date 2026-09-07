"""Turn the extracted page text into retrieval-sized chunks, one JSONL per manual.

    data/interim/text/SEAT_Ibiza_11.25.jsonl
    -> data/interim/chunks/SEAT_Ibiza_11.25.jsonl
       {"chunk_id": "SEAT_Ibiza_11.25:00042", "section": "Climatizacion",
        "pages": [119, 120], "printed": ["117", "118"], "refs": [59],
        "figures": [81], "text": "..."}

The user picks brand, model, year and edition before asking, so a query never
crosses manuals: the manual is the unit and no chunk ever mixes two of them.

What this stage undoes, all of it visible in the extracted text:

  * words split across lines ("automo-\\nviles") and the hard line wraps of the
    two-column layout, which would otherwise be embedded as broken tokens;
  * the running head and foot repeated on every page ("Climatizacion", "119"),
    which is noise 92,000 times over -- but the head is kept as the section of
    the chunk, since it is the only structure the raw text layer preserves;
  * the table of contents and the alphabetical index, whose dotted leaders are
    pure navigation and would flood any search for a part name;
  * the private-use glyphs left behind by the icon fonts;
  * the manual's own page cross-references ("=> pagina 14"), which send the
    reader to a page. An answer that shows the content cannot send anybody
    anywhere, so the reference leaves the text and stays as data.

Cross-references to a figure are left in place. A figure is content, and the
number is the only thing tying a passage to the drawing printed beside it.

The head and foot are found by frequency, not by position: older manuals put
the page number in the footer and newer ones in the header, so any fixed rule
breaks on half the corpus.
"""

import argparse
import collections
import json
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from car_manual_rag.config import CHUNK_DIR, TEXT_DIR

TARGET = 1200  # chars per chunk (~300 tokens), a step of a procedure
OVERLAP = 150  # chars repeated from the previous chunk
EDGE_LINES = 2  # lines at each end of a page that may be head or foot
MIN_REPEATS = 5  # times a line must repeat there to count as chrome
NAV_RATIO = 0.3  # share of dotted-leader lines that marks a nav page

DOTTED = re.compile(r"\.\s*\.\s*\.\s*\.")  # table-of-contents leaders
# Icon-font glyphs and control characters: neither is readable text.
NON_TEXT = re.compile(r"[\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
TITLE = re.compile(r"\w")  # a section title has letters; a glyph does not
PAGE_NUMBER = re.compile(r"\d{1,4}")
HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")
BULLET = re.compile(r"^\s*[●▪•–-]\s*")  # noqa: RUF001 - the en dash is a real bullet here
FIGURE = re.compile(r"^Fig\.\s*\d")
# The manual pointing elsewhere. Three glyphs are in use across the corpus --
# the older manuals write '=>', the newer ones '>>>' and a few '>>' -- and what
# follows is a page, a figure, or prose ('capitulo "Apoyacabezas"').
REMISSION = r"[\u203a\u00bb\u21d2]+"
# A page reference comes in two shapes. The newer manuals hang it off a glyph
# ('>>> pag. 59'); the older ones write it into the sentence ('vease la pagina
# 158', '"Airbags", pag. 38'). Both are stripped, so the leading verb, article
# and comma are part of the match -- removing only the number would leave
# 'consulte la' behind. Every optional word is bounded, or the 'de' inside
# 'grande' would match and eat the word.
PAGE_REF = re.compile(
    r"(?i)[,;]?[^\S\n]*"
    rf"(?:{REMISSION}[^\S\n]*)?"
    r"(?:\b(?:v[ée]ase|ver|consulte|consultar)\b[^\S\n]*)?"
    r"(?:\b(?:de|en)\b[^\S\n]*)?"
    r"(?:\bla\b[^\S\n]*)?"
    r"\bp[áa]g(?:ina|s)?\b\.?\s*(\d{1,4})"
)
FIG_REF = re.compile(r"\bfig\.?\s*(\d{1,4})", re.I)
GLYPH = re.compile(rf"[^\S\n]*{REMISSION}[^\S\n]*")
SPACES = re.compile(r"[^\S\n]{2,}")


def lines_of(text):
    """The page's non-empty lines, stripped of anything unreadable."""
    stripped = (line.strip() for line in NON_TEXT.sub(" ", text).split("\n"))
    return [line for line in stripped if line]


def is_nav(lines):
    """True for a table of contents or alphabetical index page."""
    if not lines:
        return False
    dotted = sum(1 for line in lines if DOTTED.search(line))
    return dotted / len(lines) >= NAV_RATIO


def find_chrome(pages):
    """Lines that repeat at the top or bottom of many pages of this manual.

    Section titles run for a whole chapter, so they clear MIN_REPEATS while
    body text at a page edge does not.
    """
    counts = collections.Counter()
    for lines in pages:
        edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update(set(edge))
    return {line for line, n in counts.items() if n >= MIN_REPEATS}


def strip_chrome(lines, chrome):
    """Drop head and foot, returning (body, section title, printed number).

    Only the page edges are examined: the same words in the middle of a page
    are body text, not chrome.
    """
    section = printed = None
    keep = []
    for i, line in enumerate(lines):
        at_edge = i < EDGE_LINES or i >= len(lines) - EDGE_LINES
        if at_edge and PAGE_NUMBER.fullmatch(line):
            printed = printed or line
        elif at_edge and line in chrome:
            # Decorative glyphs ('>>', bullets) repeat at page edges too, so
            # they are stripped like any chrome -- but they are not a title.
            if section is None and TITLE.search(line):
                section = line
        else:
            keep.append(line)
    return keep, section, printed


def reflow(lines):
    """Join the layout's hard wraps back into paragraphs.

    A paragraph breaks at a bullet, at a figure caption, and after a line that
    ends a sentence. That over-splits a paragraph made of several sentences,
    which is harmless -- the chunker regroups them -- while the opposite,
    welding two topics together, is not.
    """

    # Rejoin a hyphenated word, but keep the hyphen of a compound like
    # 'Start-\nStop', where the second half is capitalised.
    def rejoin(match):
        first, second = match.group(1), match.group(2)
        return first + ("-" if second.isupper() else "") + second

    text = HYPHEN_WRAP.sub(rejoin, "\n".join(lines))

    paragraphs = []
    current = []
    previous = ""
    for line in text.split("\n"):
        starts_new = (
            BULLET.match(line) or FIGURE.match(line) or previous.endswith((".", ":", "!", "?"))
        )
        if current and starts_new:
            paragraphs.append(" ".join(current))
            current = []
        current.append(BULLET.sub("- ", line) if BULLET.match(line) else line)
        previous = line
    if current:
        paragraphs.append(" ".join(current))
    return [p.strip() for p in paragraphs if p.strip()]


def strip_remissions(text):
    """Drop the page cross-references and report what the paragraph pointed at.

    Returns the text without them, the pages it sent the reader to, and the
    figures it named. A page reference is removed outright: the reader of an
    answer has no manual open in front of them to turn to. A figure reference
    stays, because the drawing is content and this number is what will find it.

    Removing a reference can leave the sentence hanging ("averias del sistema
    en."), since the manual wrote the pointer as part of the grammar. That is
    accepted: a dangling preposition reads better than an instruction to turn
    to a page nobody can see.
    """
    pages = [int(n) for n in PAGE_REF.findall(text)]
    text = PAGE_REF.sub("", text)
    figures = [int(n) for n in FIG_REF.findall(text)]
    text = GLYPH.sub(" ", text).replace("\xa0", " ")
    lines = [SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line), sorted(set(pages)), sorted(set(figures))


def units_of(records):
    """Flatten a manual's pages into paragraphs tagged with page and section."""
    pages = [lines_of(r["text"]) for r in records]
    chrome = find_chrome(pages)

    units = []
    section = None
    dropped = 0
    for record, lines in zip(records, pages, strict=True):
        if is_nav(lines):
            dropped += 1
            continue
        body, head, printed = strip_chrome(lines, chrome)
        section = head or section  # carry the chapter across its pages
        for paragraph in reflow(body):
            for piece in split_long(paragraph, TARGET):
                units.append(
                    {"text": piece, "page": record["page"], "printed": printed, "section": section}
                )
    return units, dropped


def split_long(paragraph, target):
    """Cut a paragraph that is already longer than a chunk (tables, spec lists)."""
    if len(paragraph) <= target:
        return [paragraph]
    pieces = []
    while len(paragraph) > target:
        cut = paragraph.rfind(" ", 0, target)
        cut = cut if cut > target // 2 else target
        pieces.append(paragraph[:cut].strip())
        paragraph = paragraph[cut:].strip()
    if paragraph:
        pieces.append(paragraph)
    return pieces


def carry(units, overlap):
    """The tail of a chunk to repeat at the head of the next one."""
    tail = []
    size = 0
    for unit in reversed(units):
        if size + len(unit["text"]) > overlap:
            break
        tail.insert(0, unit)
        size += len(unit["text"])
    return tail


def group(units, target, overlap):
    """Gather paragraphs into chunks, never crossing a section boundary."""
    chunks = []
    buffer = []
    size = 0
    for unit in units:
        if buffer:
            new_section = unit["section"] != buffer[-1]["section"]
            if new_section or size + len(unit["text"]) > target:
                chunks.append(buffer)
                buffer = [] if new_section else carry(buffer, overlap)
                size = sum(len(u["text"]) for u in buffer)
        buffer.append(unit)
        size += len(unit["text"])
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk(path, out_dir, target, overlap):
    """Write one JSONL of chunks for a manual and return its stats."""
    manual_id = Path(path).stem
    out_path = Path(out_dir) / (manual_id + ".jsonl")
    tmp = out_path.with_suffix(".part")

    try:
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh]
        units, nav = units_of(records)

        sizes = []
        with tmp.open("w", encoding="utf-8") as fh:
            for i, group_units in enumerate(group(units, target, overlap)):
                # Cleaned here, on the assembled chunk, not paragraph by
                # paragraph: a table row splits 'pag.' from its number across
                # the join, and neither half is a reference on its own.
                text, refs, figures = strip_remissions("\n".join(u["text"] for u in group_units))
                pages = sorted({u["page"] for u in group_units})
                printed = [p for p in dict.fromkeys(u["printed"] for u in group_units) if p]
                fh.write(
                    json.dumps(
                        {
                            "chunk_id": f"{manual_id}:{i:05d}",
                            "manual_id": manual_id,
                            "section": group_units[0]["section"],
                            "pages": pages,
                            "printed": printed,
                            "refs": refs,
                            "figures": figures,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sizes.append(len(text))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return {"name": Path(path).name, "error": str(e)}

    tmp.rename(out_path)
    return {
        "name": out_path.name,
        "pages": len(records),
        "nav": nav,
        "chunks": len(sizes),
        "chars": sum(sizes),
        "median": statistics.median(sizes) if sizes else 0,
        "unsectioned": sum(1 for u in units if u["section"] is None),
    }


def report(stats):
    """Print a validation summary and return True if anything looks wrong."""
    failed = [s for s in stats if "error" in s]
    ok = [s for s in stats if "error" not in s]
    chunks = sum(s["chunks"] for s in ok)
    chars = sum(s["chars"] for s in ok)
    nav = sum(s["nav"] for s in ok)

    print(
        f"\n{len(ok)} manuals, {chunks:,} chunks, {chars / 1e6:.1f}M chars "
        f"({chars / max(chunks, 1):.0f} chars/chunk), {nav} nav pages dropped"
    )

    # A manual that chunked badly shows up as far too few chunks per page, or
    # as chunks with no section, meaning the running head was never found.
    empty = [s for s in ok if not s["chunks"]]
    sparse = [s for s in ok if s["pages"] and s["chunks"] / s["pages"] < 1]
    headless = [s for s in ok if s["chunks"] and s["unsectioned"] > 0.1 * s["chunks"]]

    for label, group_stats in (
        ("failed", failed),
        ("no chunks at all", empty),
        ("under 1 chunk per page", sparse),
        ("over 10% chunks with no section", headless),
    ):
        print(f"  {label}: {len(group_stats)}")
        for s in group_stats[:10]:
            print(f"      {s['name']}{' - ' + s['error'] if 'error' in s else ''}")
        if len(group_stats) > 10:
            print(f"      ... and {len(group_stats) - 10} more")

    return bool(failed or empty or sparse or headless)


def main():  # pragma: no cover - argparse and printing
    """Run the chunking; return a shell exit code (0 = nothing suspicious)."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", type=Path, default=TEXT_DIR, help="directory of page JSONL")
    p.add_argument("--dest", type=Path, default=CHUNK_DIR, help="output directory")
    p.add_argument("--target", type=int, default=TARGET, help="chars per chunk")
    p.add_argument("--overlap", type=int, default=OVERLAP, help="chars repeated between chunks")
    p.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    p.add_argument("--limit", type=int, help="chunk only the first N manuals")
    p.add_argument("--force", action="store_true", help="re-chunk manuals already done")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.source.glob("*.jsonl"))
    if not sources:
        print(f"No extracted text in {args.source} -- run crag-extract first")
        return 1

    pending = sources if args.force else [f for f in sources if not (args.dest / f.name).exists()]
    skipped = len(sources) - len(pending)
    if args.limit:
        pending = pending[: args.limit]

    print(
        f"{len(sources)} manuals in {args.source}, {skipped} already chunked, "
        f"{len(pending)} to process -> {args.dest}"
    )

    stats = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(chunk, str(f), str(args.dest), args.target, args.overlap) for f in pending
        ]
        for i, future in enumerate(futures, 1):
            s = future.result()
            stats.append(s)
            if "error" in s:
                print(f"[{i}/{len(pending)}] {s['name']}  FAILED: {s['error']}")
            else:
                print(
                    f"[{i}/{len(pending)}] {s['name']}  {s['chunks']} chunks, "
                    f"median {s['median']:.0f} chars"
                )

    elapsed = time.time() - started
    print(f"\nChunked {len(stats)} manuals in {elapsed / 60:.1f} min")
    return 1 if report(stats) else 0


if __name__ == "__main__":
    sys.exit(main())
