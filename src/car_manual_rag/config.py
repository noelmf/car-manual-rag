"""Project paths, resolved once so every stage agrees on where data lives."""
import os
from pathlib import Path

CATALOG_NAME = "manuals.json"


def find_root():
    """Locate the project root (the directory holding manuals.json).

    Honours CAR_MANUAL_RAG_ROOT, then the repo checkout this file sits in,
    then the current directory and its parents -- so the commands work both
    from a checkout and from an installed package.
    """
    env = os.environ.get("CAR_MANUAL_RAG_ROOT")
    if env:
        return Path(env).resolve()

    candidates = [Path(__file__).resolve().parents[2], Path.cwd(), *Path.cwd().parents]
    for path in candidates:
        if (path / CATALOG_NAME).is_file():
            return path
    raise FileNotFoundError(
        f"{CATALOG_NAME} not found. Run from the project directory or set "
        "CAR_MANUAL_RAG_ROOT."
    )


ROOT = find_root()
CATALOG = ROOT / CATALOG_NAME
DATA = ROOT / "data"
PDF_DIR = DATA / "raw" / "pdf"
TEXT_DIR = DATA / "interim" / "text"
CHUNK_DIR = DATA / "interim" / "chunks"
