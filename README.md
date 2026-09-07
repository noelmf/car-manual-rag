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


## Decisiones técnicas

**El usuario filtra antes de preguntar.** Es la decisión de la que dependen las
demás. Marca, modelo, año y edición aterrizan en un único manual, así que una
pregunta se busca contra 575 fragmentos de mediana, no contra los 185.000 del
corpus. Y como muchos manuales son ediciones casi idénticas del mismo coche,
sin ese filtro la respuesta citaría la página de la edición equivocada.

**Sin base de datos vectorial.** FAISS o pgvector resuelven el problema de
buscar entre millones de vectores; con 1.107 como máximo, ese problema no
existe. El índice de un manual es una matriz de 2,5 MB y buscar en ella es un
producto escalar de 72 microsegundos. `numpy` no es una base de datos, pero es
lo que ocupa ese lugar aquí: guarda la matriz y la multiplica.

**Un índice por manual, bajo demanda.** Indexar el corpus entero cuesta unos 7
dólares, así que se indexa el primer manual que alguien abre y se reutiliza. Y
`crag-ask` nunca indexa por su cuenta: preguntar no debe gastar dinero como
efecto secundario.

**Dos dependencias y nada más.** `pymupdf` para extraer el texto de los PDFs y
`numpy` para la búsqueda; el resto es biblioteca estándar. LangChain se descartó
porque sus dos abstracciones fuertes son el troceador —peor que uno que conozca
el layout de estos manuales— y el almacén vectorial, que aquí no se usa.

**El texto se guarda en crudo y se limpia al trocear.** Unir palabras partidas o
quitar cabeceras repetidas ocurre en `crag-chunk`, no al extraer. Así se puede
rehacer la limpieza, que es lo que más se toca al afinar, sin releer 1,9 GB de
PDFs: reprocesar el corpus entero tarda 7 segundos.

**Gemini para las dos mitades.** Su modelo de embeddings encabeza la tabla MTEB
Multilingual y este corpus es español técnico, donde un modelo pensado para
inglés rinde peor. Usar el mismo proveedor para generar significa una clave, un
transporte y una factura. Se piden 768 dimensiones en vez de las 3.072 nativas:
el índice ocupa cuatro veces menos a cambio de algo de precisión.

**La respuesta trae el contenido, no la página.** Una cita solo sirve si el
lector tiene el manual delante; quien pregunta aquí no lo tiene, así que se le
enseña el fragmento entero debajo de la respuesta y el modelo tiene prohibido
remitir a ninguna parte. Eso obligó a limpiar el corpus, no solo el prompt: los
manuales se remiten a sí mismos constantemente y el **36% de los fragmentos**
llevaba una referencia a página dentro de su propio texto —`››› pág. 59` en los
manuales nuevos, `véase la página 158` en los viejos—. Ahora sale del texto y se
guarda como dato en `refs`, que es lo que permitirá convertirla en un enlace.
Las remisiones a figuras se quedan: la figura es contenido, y ese número es lo
único que ata un párrafo al dibujo impreso a su lado.

**Las figuras se recortan renderizando la página, no extrayendo el bitmap.**
Parece más directo sacar la imagen que el PDF ya lleva dentro, y es un error
que solo se ve mirando el 8% de casos raros: los números de llamada —el 1, el 2
y el 3 que atan el dibujo al texto— están dibujados **encima** como vectores, y
el bitmap guarda la orientación que tuviera el maquetador, así que un esquema de
asientos extraído en crudo sale tumbado y sin un solo número. Además el 98% son
JPEG en CMYK, que un navegador no pinta. Renderizar la región resuelve las tres
cosas de una vez y da RGB gratis. Se descartan las imágenes de menos de 10.000
píxeles cuadrados: en el corpus, cortar entre 5.000 y 20.000 descarta siempre el
mismo 8%, porque casi nada mide eso —son filetes y separadores, y las figuras de
verdad empiezan en 190×190—.

**El sistema se niega antes que adivinar.** Cada índice guarda el modelo que lo
generó y un hash del texto del que salió; si alguno no cuadra, se rechaza en vez
de usarse. Las variables de entorno tampoco tienen valor por defecto. Unos
vectores construidos sobre otro texto citan páginas equivocadas sin dar ninguna
señal, y una cita falsa es peor que ninguna respuesta porque el lector va a
actuar sobre ella.

Por lo mismo, una respuesta que se corta se rechaza en vez de imprimirse. El
modelo razona antes de escribir y ese razonamiento sale del mismo techo de
tokens que la respuesta: con el techo en 1024, «¿cómo se cambia una rueda?»
gastaba 979 tokens pensando y dejaba 41 para responder, así que el
procedimiento se cortaba tras «aparque el vehículo en un lugar seguro». Medio
procedimiento se lee igual que uno entero.
