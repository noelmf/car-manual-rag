# car-manual-rag

Preguntas en lenguaje natural sobre manuales de instrucciones de coches, con la
respuesta citando la página del manual donde está.

El usuario filtra primero **marca, modelo, año y edición**, así que cada pregunta
se responde contra un único manual. Hoy el catálogo son 274 manuales de SEAT.

## Instalación

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
cp .env.example .env      # y pon tu clave de aistudio.google.com/apikey
```

Las tres variables del `.env` son obligatorias: `GEMINI_API_KEY`,
`GEMINI_EMBEDDING` y `GEMINI_MODEL`.

## Preparar los datos

Cada etapa es reanudable: al repetirla salta lo que ya está hecho.

```bash
crag-download    # descarga los PDFs del catálogo -> data/raw/pdf/
crag-extract     # extrae el texto, una línea por página -> data/interim/text/
crag-chunk       # trocea el texto en fragmentos -> data/interim/chunks/
```

## Preguntar

```bash
# 1. Encontrar el manual
crag-catalog --options SEAT              # modelos disponibles
crag-catalog --options SEAT Ibiza        # años de ese modelo
crag-catalog --resolve SEAT Ibiza 2026 11.25
# -> SEAT_Ibiza_11.25

# 2. Indexarlo (una vez por manual, ~20 segundos)
crag-index SEAT_Ibiza_11.25

# 3. Preguntar
crag-ask SEAT_Ibiza_11.25 "¿cada cuánto se cambia el aceite?"
```

```
Servicio flexible: solo hay que cambiar el aceite cuando el vehículo lo
requiera, con un máximo de 2 años (pag. 25, pag. 324).
```

`crag-ask` nunca indexa: si falta el índice, avisa y dice qué comando lo crea.
Para ver qué fragmentos se recuperan sin gastar en generar la respuesta:

```bash
crag-index SEAT_Ibiza_11.25 --search "presión de los neumáticos"
```

## Comandos

| Comando | Para qué |
|---|---|
| `crag-download` | Descarga los PDFs del catálogo |
| `crag-extract` | Extrae el texto de cada página |
| `crag-chunk` | Trocea el texto en fragmentos con su sección y página |
| `crag-catalog` | Explora el catálogo y valida que todo cuadra |
| `crag-index` | Crea el índice de un manual, o busca en él con `--search` |
| `crag-ask` | Pregunta y responde citando la página |

Todos aceptan `--help`.
