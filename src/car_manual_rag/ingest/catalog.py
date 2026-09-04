"""The catalogue read as the filter the user drives before asking a question.

The user narrows brand -> model -> year -> edition, and that lands on exactly
one manual; the question is then answered from that manual alone. So the
catalogue is not just the download list -- it is the index of the picker, and
the manual id it yields is the name every later stage uses for its files.

Two shapes of the data drive the code here:

  * model, year and edition are lists, and 32 entries cover several models at
    once (one manual serves the Leon and the Leon Sportstourer), so a manual
    appears under every combination it covers -- 274 manuals, 320 paths;
  * each (brand, model, year, edition) must land on one manual and no more.
    Nothing in manuals.json enforces that, so validate() checks it.
"""

import argparse
import itertools
import json
import re
import sys

from car_manual_rag.config import CATALOG, CHUNK_DIR

EDITION = re.compile(r"(\d{2})\.(\d{2})")  # 'MM.YY', the date SEAT prints


def slug(text):
    """Make a model name safe for a filename: 'Leon SC' -> 'Leon-SC'."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def manual_id(manual):
    """The manual's identity: brand, model(s) and edition(s).

    This is the name of its PDF, of its extracted text and of its chunks, so
    it lives here rather than in any one stage.
    """
    models = "+".join(slug(m) for m in manual["model"])
    editions = "+".join(manual["edition"])
    return f"{manual['brand']}_{models}_{editions}"


def load(path=CATALOG):
    """The catalogue, with each entry tagged with its manual id."""
    manuals = json.loads(path.read_text(encoding="utf-8"))
    for m in manuals:
        m["manual_id"] = manual_id(m)
    return manuals


def paths(manuals):
    """Every (brand, model, year, edition) -> manual id the picker can offer."""
    found = {}
    for m in manuals:
        for combo in itertools.product(m["model"], m["year"], m["edition"]):
            found.setdefault((m["brand"], *combo), []).append(m["manual_id"])
    return found


def newest_first(editions):
    """Sort 'MM.YY' editions by date, not as text: 06.22 is older than 11.22."""

    def when(edition):
        match = EDITION.fullmatch(edition)
        return (int(match.group(2)), int(match.group(1))) if match else (0, 0)

    return sorted(editions, key=when, reverse=True)


def tree(manuals):
    """The picker, nested brand -> model -> year -> edition -> manual id.

    Years and editions come out newest first, which is the order a driver
    looks for their own car in.
    """
    out = {}
    for (brand, model, year, edition), ids in paths(manuals).items():
        out.setdefault(brand, {}).setdefault(model, {}).setdefault(year, {})[edition] = ids[0]
    return {
        b: {
            m: {
                y: {e: years[y][e] for e in newest_first(years[y])}
                for y in sorted(years, reverse=True)
            }
            for m, years in sorted(models.items())
        }
        for b, models in sorted(out.items())
    }


def options(manuals, brand=None, model=None, year=None):
    """What the next dropdown should offer, given what is already chosen.

    Called with nothing it returns the brands, with a brand the models of that
    brand, and so on -- one function for the whole cascade.
    """
    levels = tree(manuals)
    for chosen in (brand, model, year):
        if chosen is None:
            break
        if chosen not in levels:
            raise LookupError(f"unknown choice {chosen!r}; expected one of {sorted(levels)}")
        levels = levels[chosen]
    return list(levels)


def resolve(manuals, brand, model, year, edition):
    """The manual id for a complete selection."""
    ids = paths(manuals).get((brand, model, year, edition))
    if not ids:
        raise LookupError(f"no manual for {brand} {model} {year} edition {edition}")
    return ids[0]


def validate(manuals, chunk_dir=CHUNK_DIR):
    """Return the catalogue's problems as a list of messages, empty if sound."""
    problems = []

    ids = [m["manual_id"] for m in manuals]
    for duplicate in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"duplicate manual id: {duplicate}")

    for combo, matches in paths(manuals).items():
        if len(matches) > 1:
            problems.append(
                f"{' '.join(combo)} matches {len(matches)} manuals: {', '.join(matches)}"
            )

    chunked = {p.stem for p in chunk_dir.glob("*.jsonl")} if chunk_dir.is_dir() else set()
    for missing in sorted(set(ids) - chunked):
        problems.append(f"no chunks for {missing} -- run crag-chunk")
    for orphan in sorted(chunked - set(ids)):
        problems.append(f"chunks for {orphan}, which is not in the catalogue")

    return problems


def main():  # pragma: no cover - argparse and printing
    """Inspect and validate the catalogue; 0 = the picker is sound."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tree", action="store_true", help="print the whole picker")
    p.add_argument(
        "--options",
        nargs="*",
        metavar="CHOICE",
        help="what to offer next, given 0-3 choices (brand, model, year)",
    )
    p.add_argument(
        "--resolve",
        nargs=4,
        metavar=("BRAND", "MODEL", "YEAR", "EDITION"),
        help="print the manual id for a complete selection",
    )
    args = p.parse_args()

    manuals = load()

    # A wrong choice is a user typo, not a crash: report it and exit 1.
    if args.resolve:
        try:
            print(resolve(manuals, *args.resolve))
        except LookupError as e:
            print(e)
            return 1
        return 0

    if args.options is not None:
        try:
            for option in options(manuals, *args.options):
                print(option)
        except LookupError as e:
            print(e)
            return 1
        return 0

    if args.tree:
        for brand, models in tree(manuals).items():
            print(brand)
            for model, years in models.items():
                print(f"  {model}")
                for year, editions in years.items():
                    print(f"    {year}: " + ", ".join(f"{e} -> {i}" for e, i in editions.items()))

    combos = paths(manuals)
    print(
        f"\n{len(manuals)} manuals, {len({m['brand'] for m in manuals})} brands, "
        f"{len({mo for m in manuals for mo in m['model']})} models, "
        f"{len(combos)} paths through the picker"
    )

    problems = validate(manuals)
    print(f"  problems: {len(problems)}")
    for problem in problems[:20]:
        print(f"      {problem}")
    if len(problems) > 20:
        print(f"      ... and {len(problems) - 20} more")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
