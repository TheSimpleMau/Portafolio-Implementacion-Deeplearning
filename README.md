# Predicción de jugadas de ajedrez con una CNN

**Autor:** Mauricio Olguín Sánchez
**Matrícula:** A01711522

Proyecto del módulo de *Deep Learning* (Inteligencia artificial avanzada para la ciencia de datos, Tecnológico de Monterrey).

## Objetivo

Entrenar con TensorFlow/Keras una red neuronal convolucional que, dada una posición del tablero, prediga la jugada que se jugó. Es un problema de clasificación multiclase: la entrada es el tablero (8×8 con canales por tipo de pieza) y la salida es una distribución de probabilidad sobre 4,096 jugadas posibles (casilla de origen × casilla de destino).

## Dataset

Partidas de la [Lichess Open Database](https://database.lichess.org/) (licencia CC0), archivo `lichess_db_standard_rated_2013-03.pgn.zst`. El conjunto se divide por partida, no por posición, para evitar fuga de datos entre entrenamiento, validación y prueba.

## Entorno

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Los datos se descargan manualmente a `data/raw/` (la carpeta está ignorada por git).

## Estructura

```
model/   # todo lo del modelo
  preprocessing.py   # tablero y jugadas a tensores (4,096 clases)
  etlProcess.py      # PGN.zst de Lichess a posiciones, filtros y división por partida
  eda.py, plotUtil.py
game/    # jugar contra el modelo
  players.py         # jugadores: modelo, Stockfish y azar
  match.py           # partidas CPU vs CPU sin ventana y PGN
  gui.py             # ventana con pygame-ce
tests/   # pytest
```

## Uso

```bash
.venv/bin/python -m model.etlProcess   # ETL (los datos crudos van en data/raw/)
.venv/bin/python -m model.eda          # gráficas del análisis exploratorio en graficas/
.venv/bin/python -m pytest -q
```

## Jugar contra el modelo

Las reglas, la validación de jugadas y la comunicación con el motor las da [`python-chess`](https://python-chess.readthedocs.io/). El rival de comparación es [Stockfish](https://stockfishchess.org/) (GPL-3.0), que se instala aparte y no forma parte del repositorio:

```bash
brew install stockfish
```

```bash
.venv/bin/python -m game.gui
```

- En el menú se eligen las blancas y las negras (Humano, Modelo, Stockfish o Azar), la fuerza de cada Stockfish por separado (Elo 1320 a 2800, o completo) y el modelo entre los archivos `.keras` de `models/`.
- Clic en la pieza y clic en el destino; la promoción es siempre a dama, como en el modelo. Si el humano lleva negras, el tablero se muestra desde su lado.
- La barra junto al tablero y el panel muestran la ventaja según Stockfish a fuerza completa: evaluación en peones (o `#n` si hay mate) y probabilidad de ganar de las blancas con la fórmula de Lichess.
- Cuando juega el modelo, las flechas y el panel muestran sus tres jugadas más probables (renormalizadas sobre las legales) y cuánta probabilidad puso en jugadas ilegales.
- Cuando juegan dos CPU se elige el modo: **Automático** (juegan solas, con espacio para pausar) o **Manual** (cada jugada espera la flecha derecha o Enter).
- Teclas: `n` nueva partida con los mismos jugadores, `u` deshacer, `Esc` vuelve al menú.
- Al terminar, la partida se guarda en `resultados/partidas/` como PGN, con el top-3 del modelo como comentario en cada jugada suya.
