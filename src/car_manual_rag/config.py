"""Project paths and the .env file, so every stage agrees on where things are."""
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
INDEX_DIR = DATA / "processed" / "index"
ENV_FILE = ROOT / ".env"

API_KEY = "GEMINI_API_KEY"
EMBEDDING = "GEMINI_EMBEDDING"
MODEL = "GEMINI_MODEL"

# Every setting, with where to get it. None has a default: an index is only
# meaningful next to the model that built it, so guessing is the one thing
# this must not do.
SETTINGS = {API_KEY: "get one at aistudio.google.com/apikey",
            EMBEDDING: "e.g. gemini-embedding-001",
            MODEL: "e.g. gemini-3.6-flash"}

_loaded = False


def load_env(path=ENV_FILE):
    """Read KEY=value lines from .env into the environment.

    A variable already set in the environment is never overwritten: an explicit
    export on the command line has to win over the file, or overriding a key
    for one run would be impossible. Reads the file once per process.

    Kept deliberately small -- 'export' prefixes and surrounding quotes are
    tolerated because they survive a copy-paste, and anything else is ignored
    rather than raising: a typo in .env should not stop the pipeline.
    """
    global _loaded
    if _loaded or not path.is_file():
        return
    _loaded = True

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def required(name):
    """A mandatory setting, from the environment or .env."""
    load_env()
    value = os.environ.get(name)
    if not value:
        raise LookupError(f"{name} is not set -- {SETTINGS.get(name, 'see .env.example')}")
    return value
