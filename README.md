# car-manual-rag

[![checks](https://github.com/noelmf/car-manual-rag/actions/workflows/checks.yml/badge.svg)](https://github.com/noelmf/car-manual-rag/actions/workflows/checks.yml)
[![codecov](https://codecov.io/gh/noelmf/car-manual-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/noelmf/car-manual-rag)

Sistema RAG (Retrieval Augmented Generation) sobre manuales de usuario de
coches. 

> [!NOTE]
> Incluye 274 manuales de la marca SEAT, cubriendo 19 modelos fabricados entre 2006 y 2026. Cada modelo y año puede tener varias versiones.


## Instalación

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
cp .env.example .env
```

Las tres variables del `.env` son obligatorias: `GEMINI_API_KEY`,
`GEMINI_EMBEDDING` y `GEMINI_MODEL`.


## Preparar los datos

Cada etapa es reanudable: al repetirla salta lo que ya está hecho.

```bash
crag-download    # descarga los PDFs del catálogo -> data/raw/pdf/
crag-extract     # extrae el texto, una línea por página -> data/interim/text/
crag-chunk       # trocea el texto en fragmentos -> data/interim/chunks/
crag-figures     # recorta las figuras de cada página -> data/interim/figures/
```


## Preguntar

```bash
# 1. Encontrar el nombre del manual. Cada modelo y año puede tener varias versiones
crag-catalog --options SEAT                     # modelos disponibles
crag-catalog --options SEAT Ibiza               # años de ese modelo
crag-catalog --options SEAT Ibiza 2026.         # versiones de ese año
crag-catalog --resolve SEAT Ibiza 2026 11.25    # nombre del manual
# -> SEAT_Ibiza_11.25

# 2. Indexar manual
crag-index SEAT_Ibiza_11.25

# 3. Preguntar sobre un manual
crag-ask SEAT_Ibiza_11.25 "¿cada cuánto se cambia el aceite?"
# -> la respuesta, y debajo los fragmentos del manual de los que sale
```


## La API

`crag-serve` levanta cuatro rutas:

| Ruta | Para qué |
|---|---|
| `GET /api/catalog` | El selector entero: marca → modelo → año → edición |
| `POST /api/ask` | `{manual, question}` → la respuesta completa |
| `POST /api/ask/stream` | Lo mismo, según se produce |
| `GET /api/figure/<manual>/<fichero>` | Una figura, en JPEG |

```bash
curl -N -X POST localhost:8000/api/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"manual":"SEAT_Ibiza_11.25","question":"¿cada cuánto se cambia el aceite?"}'
```

Responde JSON delimitado por saltos de línea, un evento por línea: `hits` una
vez, `token` muchas, y después `done` **o** `error`. Cada fragmento de `hits`
lleva sus figuras como URL.

Un stream que acaba sin `done` ni `error` no terminó, y quien lo pinte tiene que
decirlo: medio procedimiento se lee igual que uno entero.


## Comandos

| Comando | Para qué |
|---|---|
| `crag-download` | Descarga los PDFs del catálogo |
| `crag-extract` | Extrae el texto de cada página |
| `crag-chunk` | Trocea el texto en fragmentos con su sección, página y remisiones |
| `crag-figures` | Recorta en JPEG todas las figuras de los manuales |
| `crag-catalog` | Explora el catálogo y valida que todo cuadra |
| `crag-index` | Crea el índice de un manual |
| `crag-ask` | Pregunta y responde mostrando los fragmentos |

Todos aceptan `--help`.


## Tests

```bash
./.venv/bin/pytest --cov
```

Mide cobertura de rama y falla por debajo del 96%. El umbral y el alcance
viven en `pyproject.toml`, así que la orden es la misma aquí que en CI.

Los mismos `ruff` y `pytest` corren en cada push. Para que `ruff` rechace
también los commits locales, una vez por checkout:

```bash
git config core.hooksPath hooks
```


