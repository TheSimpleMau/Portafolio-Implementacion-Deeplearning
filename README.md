# Predicción de jugadas de ajedrez con una CNN

**Autor:** Mauricio Olguín Sánchez
**Matrícula:** A01711522

Proyecto del módulo de *Deep Learning* (Inteligencia artificial avanzada para la ciencia de datos, Tecnológico de Monterrey).

## Objetivo

Entrenar con TensorFlow/Keras una red neuronal convolucional que, dada una posición del tablero, prediga la jugada que se jugó. Es un problema de clasificación multiclase: la entrada es el tablero (8×8 con canales por tipo de pieza) y la salida es una distribución de probabilidad sobre 4,096 jugadas posibles (casilla de origen × casilla de destino).

## Dataset

Partidas de la [Lichess Open Database](https://database.lichess.org/) (licencia CC0), archivo `lichess_db_standard_rated_2026-09.pgn.zst` (partidas que no son Bullet entre jugadores humanos con Elo de 2,200 o más). El conjunto se divide por partida, no por posición, para evitar fuga de datos entre entrenamiento, validación y prueba.

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
  elo.py             # Elo del modelo con una escalera de rivales
tests/   # pytest
```

## Uso

Con el entorno activado:

```bash
python main.py                # entrena y evalúa las corridas de EXPERIMENTS; la configuración está al inicio del archivo
python -m model.etlProcess filtrar   # crudo de data/raw/ a data/processed/games_<mes>.parquet (una vez)
python -m model.etlProcess    # ETL: posiciones desde el Parquet (o desde el crudo si no existe)
python -m model.eda           # gráficas del análisis exploratorio en graficas/
python -m model.eda aperturas # aperturas, ritmos, terminación y duración de las partidas usadas
python -m model.evaluation    # líneas base en validación
python -m model.evaluation aperturas models/m1.keras   # acierto del modelo por apertura
python -m game.elo models/m1.keras   # Elo del modelo (ver abajo)
pytest -q
```

El mejor modelo (menor pérdida en validación) se guarda en `models/m1.keras`; las métricas, en `resultados/`, y las curvas, en `graficas/`.

## Jugar contra el modelo

Las reglas, la validación de jugadas y la comunicación con el motor las da [`python-chess`](https://python-chess.readthedocs.io/). El rival de comparación es [Stockfish](https://stockfishchess.org/) (GPL-3.0), que se instala aparte y no forma parte del repositorio:

```bash
brew install stockfish
```

```bash
python -m game.gui
```

- En el menú se eligen las blancas y las negras (Humano, Modelo, Stockfish o Azar), la fuerza de cada Stockfish por separado (Elo 1320 a 2800, o completo) y el modelo entre los archivos `.keras` de `models/`.
- Clic en la pieza y clic en el destino; la promoción es siempre a dama, como en el modelo. Si el humano lleva negras, el tablero se muestra desde su lado.
- La barra junto al tablero y el panel muestran la ventaja según Stockfish a fuerza completa: evaluación en peones (o `#n` si hay mate) y probabilidad de ganar de las blancas con la fórmula de Lichess.
- Cuando juega el modelo, las flechas y el panel muestran sus tres jugadas más probables (renormalizadas sobre las legales) y cuánta probabilidad puso en jugadas ilegales.
- Cuando juegan dos CPU se elige el modo: **Automático** (juegan solas, con espacio para pausar) o **Manual** (cada jugada espera la flecha derecha o Enter).
- Teclas: `n` nueva partida con los mismos jugadores, `u` deshacer, `Esc` vuelve al menú.
- Al terminar, la partida se guarda en `resultados/partidas/` como PGN, con el top-3 del modelo como comentario en cada jugada suya.

## Elo del modelo

`python -m game.elo models/<modelo>.keras` juega el modelo contra una escalera de rivales (azar, captura de mayor valor, Stockfish a profundidad 1, Maia 1100/1500/1900 y Stockfish con `UCI_Elo` 1320), desde aperturas variadas y con los dos colores, y estima el Elo de todos por máxima verosimilitud con Stockfish 1320 como ancla. Las aperturas salen de `python -m model.eda aperturas`. Maia ([CSSLab/maia-chess](https://github.com/CSSLab/maia-chess), GPL-3.0) necesita lc0 y sus pesos, que no forman parte del repositorio:

```bash
brew install lc0
```

```bash
mkdir -p models/maia && for e in 1100 1500 1900; do curl -L -o models/maia/maia-$e.pb.gz https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-$e.pb.gz; done
```
