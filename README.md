# car-manual-rag

[![checks](https://github.com/noelmf/car-manual-rag/actions/workflows/checks.yml/badge.svg)](https://github.com/noelmf/car-manual-rag/actions/workflows/checks.yml)
[![codecov](https://codecov.io/gh/noelmf/car-manual-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/noelmf/car-manual-rag)
![status](https://img.shields.io/badge/status-work%20in%20progress-yellow)

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


## Interfaz web

Esqueleto de React con Vite, en `web/`. De momento solo pinta el nombre del
proyecto.

```bash
pnpm --dir web install
pnpm --dir web run build      # -> web/dist, que crag-serve sirve
./.venv/bin/crag-serve        # http://127.0.0.1:8000
```

Para trabajar en el front, Vite lo sirve con recarga en caliente y manda `/api`
al servidor de Python, así que corren los dos a la vez:

```bash
./.venv/bin/crag-serve        # una terminal
pnpm --dir web run dev        # otra: http://localhost:5173
```

En desarrollo se abre el 5173, no el 8000.


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

El front se prueba aparte, con Vitest y Testing Library:

```bash
pnpm --dir web run test         # una pasada, con cobertura
pnpm --dir web run test:watch   # en watch, sin cobertura
pnpm --dir web run typecheck    # solo los tipos
```

Vitest lee `web/vite.config.ts`, así que los tests resuelven los imports igual
que la aplicación y no hay una segunda cadena de build que mantener al día. Los
tests pulsan las teclas que pulsa una persona en vez de asomarse al estado del
componente.

Codecov recibe **dos informes con flag propio**, `python` y `web`, no uno
mezclado: un front de dos ficheros desaparecería dentro de un backend de
ochocientas sentencias y una caída ahí no se vería nunca.

En cada push corren las dos mitades: `ruff` y `pytest` por un lado, y
`typecheck`, `test` y `build` por el otro. Para que `ruff` rechace también los
commits locales, una vez por checkout:

```bash
git config core.hooksPath hooks
```

Ese hook solo mira Python, así que CI es el único sitio donde se comprueban los
tipos y los tests del front.


