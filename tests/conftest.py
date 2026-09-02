"""Fixtures built from what the real manuals actually look like."""
import json

import pytest

# Two pages in the newer layout: the running head carries the section, the
# printed number sits next to it, and the text is wrapped mid-word.
PAGES = [
    {"page": 1, "text": "Climatizacion\n117\nCalefaccion y refrigeracion\n"
                        "El aire acondicionado enfria y deshumedece el aire del habita-\n"
                        "culo.\n●Gire el regulador para ajustar la temperatura.\n"
                        "●Pulse la tecla para conectar el sistema.\n"},
    {"page": 2, "text": "Climatizacion\n118\nMandos\n"
                        "Fig. 81  En la consola central: mandos del aire acondi-\ncionado.\n"
                        "El nivel 0 desconecta el ventilador. El nivel 4 es el maximo.\n"},
    {"page": 3, "text": "Climatizacion\n119\nAvisos\n"
                        "Nota\nNo bloquee las salidas de aire.\n"},
]

# A table of contents page: dotted leaders, no prose.
NAV_PAGE = {"page": 4, "text": "Indice\n120\n"
                               "Climatizacion . . . . . . . . . . . . . . 117\n"
                               "Mandos . . . . . . . . . . . . . . . . . . 118\n"
                               "Avisos . . . . . . . . . . . . . . . . . . 119\n"}


@pytest.fixture
def pages():
    return [dict(p) for p in PAGES]


@pytest.fixture
def chunk_file(tmp_path):
    """Write a chunks JSONL and return its directory and the records."""
    def write(records, manual_id="TEST_Manual_01.25"):
        path = tmp_path / f"{manual_id}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                        encoding="utf-8")
        return tmp_path, records
    return write


@pytest.fixture
def chunks():
    return [
        {"chunk_id": "TEST_Manual_01.25:00000", "manual_id": "TEST_Manual_01.25",
         "section": "Climatizacion", "pages": [1], "printed": ["117"], "text": "El aire acondicionado."},
        {"chunk_id": "TEST_Manual_01.25:00001", "manual_id": "TEST_Manual_01.25",
         "section": "Climatizacion", "pages": [1, 2], "printed": ["117", "118"], "text": "Mandos del sistema."},
    ]
