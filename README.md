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
tests/   # pytest
```

## Uso

```bash
.venv/bin/python -m model.etlProcess   # ETL (los datos crudos van en data/raw/)
.venv/bin/python -m model.eda          # gráficas del análisis exploratorio en graficas/
.venv/bin/python -m pytest -q
```
