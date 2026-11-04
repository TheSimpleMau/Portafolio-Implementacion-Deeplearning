# Predicción de jugadas de ajedrez con una CNN

**Autor:** Mauricio Olguín Sánchez
**Matrícula:** A01711522

**Reporte:** [reporte/reporte.pdf](reporte/reporte.pdf)

Proyecto del módulo de *Deep Learning* (Inteligencia artificial avanzada para la ciencia de datos, Tecnológico de Monterrey).

## Objetivo

Entrenar con TensorFlow/Keras una red neuronal convolucional que, dada una posición del tablero, prediga la jugada que se jugó. Es un problema de clasificación multiclase: la entrada es el tablero (8×8 con canales por tipo de pieza) y la salida es una distribución de probabilidad sobre 4,096 jugadas posibles (casilla de origen × casilla de destino).

## Dataset

Partidas de la [Lichess Open Database](https://database.lichess.org/) (licencia CC0), archivo `lichess_db_standard_rated_2026-09.pgn.zst` (partidas que no son Bullet entre jugadores humanos con Elo de 2,200 o más). El conjunto se divide por partida, no por posición, para evitar fuga de datos entre entrenamiento, validación y prueba. Quedan 18,022,575 posiciones (14.4 M de entrenamiento, 1.8 M de validación y 1.8 M de prueba).

## Resultados

El modelo final es una red de dos capas convolucionales (64 y 128 filtros de 3×3), una capa densa de 2,048 unidades con *Dropout* 0.5 y una salida *softmax* de 4,096 clases (25.3 M de parámetros). Se entrenó con entrenamiento y validación juntos durante 6 épocas y se evaluó una sola vez en prueba, eligiendo la jugada real entre las jugadas legales:

| Corte | Posiciones | Top-1 | Top-3 | Top-5 | Primera opción ilegal |
|---|---|---|---|---|---|
| Todas | 1,802,151 | 43.0 % | 70.3 % | 81.1 % | 1.4 % |
| Apertura | 553,696 | 53.9 % | 82.7 % | 91.9 % | 0.1 % |
| Medio juego | 1,083,263 | 37.9 % | 64.2 % | 75.6 % | 1.7 % |
| Final | 165,192 | 40.1 % | 68.5 % | 80.4 % | 3.8 % |

La última columna es el porcentaje de posiciones donde la jugada con mayor probabilidad del modelo es ilegal; el resto de las columnas se calcula después de descartar las ilegales. El análisis completo está en el [reporte](reporte/reporte.pdf).

## Entorno

El entorno vive dentro del proyecto, en `.venv/` (ignorado por git), y se crea una sola vez:

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Después basta activarlo en cada terminal nueva, desde la carpeta del proyecto:

```bash
source .venv/bin/activate
```

Los datos se descargan manualmente a `data/raw/` (la carpeta está ignorada por git).

### Entrenar con la GPU (opcional, Mac con Apple silicon)

`tensorflow-metal` todavía no funciona con TensorFlow 2.21, así que la GPU usa un segundo entorno, `.venv-metal/` (también ignorado), con TensorFlow 2.19.1:

```bash
uv venv --python 3.12 .venv-metal
```

```bash
uv pip install --python .venv-metal/bin/python -r requirements-metal.txt
```

Con ese entorno, `python main.py` entrena unas 2.4 veces más rápido y da el mismo modelo que en la CPU. Lo demás (ETL, EDA, evaluación y juego) se corre con `.venv`.

## Estructura

```
model/   # todo lo del modelo
  preprocessing.py   # tablero y jugadas a tensores (4,096 clases)
  etlProcess.py      # PGN.zst de Lichess a Parquet filtrado y a posiciones; división por partida
  eda.py, plotUtil.py
  evaluation.py      # precisión top-k entre jugadas legales y líneas base
  chessModels.py     # arquitecturas
  experiment.py      # entrenamiento por lotes y evaluación en validación
  experimentStorage.py  # resultados en resultados/
main.py  # entrena y evalúa el modelo
game/    # jugar contra el modelo
  players.py         # jugadores: modelo, Stockfish, Maia, captura y azar
  match.py           # partidas CPU vs CPU sin ventana y PGN
  gui.py             # ventana con pygame-ce
  predict.py         # jugadas más probables del modelo para una posición, por consola
  elo.py             # Elo del modelo con una escalera de rivales
tests/   # pytest: las jugadas del juego respetan las reglas
```

## Uso

Con el entorno activado:

```bash
python main.py                # con FINAL = True, la corrida final (entrenamiento + validación, prueba una vez); con FINAL = False, las corridas de EXPERIMENTS en validación. La configuración está al inicio del archivo
python -m model.etlProcess filtrar   # crudo de data/raw/ a data/processed/games_<mes>.parquet (una vez)
python -m model.etlProcess    # ETL: posiciones desde el Parquet (o desde el crudo si no existe)
python -m model.eda           # gráficas del análisis exploratorio en graficas/
python -m model.eda aperturas # aperturas, ritmos, terminación y duración de las partidas usadas
python -m model.evaluation    # líneas base en validación
python -m model.evaluation aperturas models/final_dropout_05_dense_2048.keras   # acierto del modelo por apertura
python -m game.elo models/final_dropout_05_dense_2048.keras   # Elo del modelo (ver abajo)
pytest -q
```

Cada corrida guarda su mejor modelo (menor pérdida en validación) en `models/<nombre>.keras`; la corrida final guarda `models/final_dropout_05_dense_2048.keras`. Las métricas van en `resultados/` y las curvas en `graficas/`.

## Obtener un modelo

La carpeta `models/` no está en el repositorio, así que tras clonarlo no hay ningún `.keras` y ni `game.predict` ni la ventana del juego funcionan hasta crear uno. Con el entorno activado (para entrenar en GPU, `main.py` se corre con `.venv-metal`):

1. Descargar `lichess_db_standard_rated_2026-09.pgn.zst` (unos 29 GB) de la [Lichess Open Database](https://database.lichess.org/) a `data/raw/`.
2. `python -m model.etlProcess filtrar` deja las partidas filtradas en `data/processed/games_2026-09.parquet` (una vez).
3. `python -m model.etlProcess` genera las posiciones (unos 40 minutos; necesita varios GB de memoria).
4. `python main.py` con `FINAL = True` entrena y guarda `models/final_dropout_05_dense_2048.keras` (unas 3.5 horas en la GPU de un Mac con Apple silicon).

## Jugar contra el modelo

Las reglas, la validación de jugadas y la comunicación con el motor las da [`python-chess`](https://python-chess.readthedocs.io/). El rival de comparación es [Stockfish](https://stockfishchess.org/) (GPL-3.0), que se instala aparte y no forma parte del repositorio:

```bash
brew install stockfish
```

```bash
python -m game.gui
```

- **Menú:** una tarjeta por color (blancas y negras) con Humano, Modelo, Stockfish o Azar. Solo se muestra lo que aplica: con Stockfish, su fuerza (Elo 1320 a 2800, o completo); con Modelo, el archivo `.keras` de `models/`. Por defecto juega un humano con blancas contra el modelo.
- **Jugar:** clic en la pieza y clic en el destino (las jugadas legales se marcan con un punto o un anillo); la promoción es siempre a dama, como en el modelo. Si el humano lleva negras, el tablero se muestra desde su lado.
- **Barra de ventaja** a la izquierda del tablero: la altura de la parte blanca es la probabilidad de ganar de las blancas según Stockfish a fuerza completa (fórmula de Lichess) y el número del extremo es la evaluación en peones (o `#n` si hay mate). Se anima al cambiar y no aparece si Stockfish no está instalado.
- **Flechas del modelo:** tras cada jugada suya, tres flechas muestran sus jugadas más probables (renormalizadas sobre las legales), cada una con su color y su número (1, 2, 3). La tarjeta «Modelo» del panel repite esos colores con la probabilidad de cada jugada y la que el modelo puso en jugadas ilegales.
- **Velocidad:** cuando juega alguna CPU, `+` y `-` (o ↑ y ↓, o los botones del panel) cambian el tiempo entre sus jugadas, de 2 s a la velocidad máxima; la pausa se activa con espacio o con su botón.
- **Panel:** nombres de los jugadores, lista de jugadas en notación algebraica, tarjeta del modelo y controles (Nueva, Deshacer, Menú, Pausa, Flechas y velocidad), todos clicables.
- **Teclas:** `n` nueva partida con los mismos jugadores, `u` deshacer, `a` mostrar u ocultar las flechas, `+`/`-` velocidad, espacio pausa y `Esc` vuelve al menú.
- Al terminar, la partida se guarda en `resultados/partidas/` como PGN, con el top-3 del modelo como comentario en cada jugada suya.

## Predicciones por consola

`game.predict` recibe una posición en notación FEN e imprime el tablero recibido, las cinco jugadas más probables del modelo final (en notación algebraica, con su porcentaje), la probabilidad que el modelo puso en jugadas ilegales y el tablero tras la jugada predicha. La FEN va entre comillas porque lleva espacios. El modelo (`MODEL_PATH`) y cuántas jugadas mostrar (`TOP_K`) son constantes al inicio de `game/predict.py`.

Sin argumento usa la posición inicial:

```bash
python -m game.predict
```

Respuesta de las negras a 1.e4:

```bash
python -m game.predict "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
```

```
Posición recibida:
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R

Mueven negras. Jugadas más probables:
  c5        43.0 %
  e5        19.1 %
  c6        11.0 %
  e6        10.0 %
  d5         4.7 %
Probabilidad que el modelo puso en jugadas ilegales: 0.0 %

Posición tras la jugada predicha, c5:
r n b q k b n r
p p . p p p p p
. . . . . . . .
. . p . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R
```

Una dama negra sin defensa, que el caballo de f3 puede capturar:

```bash
python -m game.predict "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
```

## Elo del modelo

`python -m game.elo models/<modelo>.keras` juega el modelo contra una escalera de rivales (azar, captura de mayor valor, Stockfish a profundidad 1, Maia 1100/1500/1900 y Stockfish con `UCI_Elo` 1320), desde aperturas variadas y con los dos colores, y estima el Elo de todos por máxima verosimilitud con Stockfish 1320 como ancla. Las aperturas salen de `python -m model.eda aperturas`. Maia ([CSSLab/maia-chess](https://github.com/CSSLab/maia-chess), GPL-3.0) necesita lc0 y sus pesos, que no forman parte del repositorio:

```bash
brew install lc0
```

```bash
mkdir -p models/maia && for e in 1100 1500 1900; do curl -L -o models/maia/maia-$e.pb.gz https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-$e.pb.gz; done
```
