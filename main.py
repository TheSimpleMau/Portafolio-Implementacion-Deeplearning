"""Entrena y evalúa en validación cada corrida de EXPERIMENTS, una tras otra.

Uso, desde la raíz del proyecto con el entorno activado (source .venv/bin/activate):
    python main.py

Las constantes de abajo son la configuración base; cada corrida de EXPERIMENTS cambia una sola cosa
(su "model" va a get_model; las demás claves reemplazan las de la base). Para correr menos, comentar renglones.
TRAIN_LIMIT / VAL_LIMIT = un número usa esa muestra de train / val; None usa el conjunto completo.
PATIENCE = None quita EarlyStopping. M1 completo: TRAIN_LIMIT = VAL_LIMIT = PATIENCE = None, EPOCHS = 10
y EXPERIMENTS = [{"name": "m1"}].
Cada mejor modelo (menor val_loss) queda en models/<name>.keras, donde lo encuentra el menú del juego.
"""
from model.experiment import run_experiment

BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 8
PATIENCE = 2
SEED = 42
TRAIN_LIMIT = 250_000
VAL_LIMIT = 100_000
EVALUATE_TEST = False  # test se usa una sola vez, al final del proyecto

# Fase 1: un solo cambio frente a M1 en cada corrida.
PHASE_1 = [
    {"name": "b0_base"},
    {"name": "d1_dropout_dense_03", "model": {"dropout": 0.3}},
    {"name": "d2_dropout_dense_05", "model": {"dropout": 0.5}},
    {"name": "d3_dropout_flatten_03", "model": {"dropout": 0.3, "dropout_after": "flatten"}},
    {"name": "h1_lr_3e-4", "learning_rate": 0.0003},
    {"name": "h3_lote_256", "batch_size": 256},
    {"name": "h4_lote_1024", "batch_size": 1024},
    {"name": "h5_dense_512", "model": {"dense_units": 512}},
    {"name": "h6_filtros_32_64", "model": {"filters": (32, 64)}},
]

# Fase 2: el Dropout en dense_1 (lo único que ayudó en la fase 1) con más tasa o más capacidad,
# y dos corridas con otra semilla para medir el ruido.
PHASE_2 = [
    {"name": "d4_dropout_dense_06", "model": {"dropout": 0.6}},
    {"name": "d5_dropout_dense_07", "model": {"dropout": 0.7}},
    {"name": "c1_dropout_05_dense_2048", "model": {"dropout": 0.5, "dense_units": 2048}},
    {"name": "c2_dropout_05_filtros_128_256", "model": {"dropout": 0.5, "filters": (128, 256)}},
    {"name": "s1_base_semilla_43", "seed": 43},
    {"name": "s2_dropout_dense_05_semilla_43", "model": {"dropout": 0.5}, "seed": 43},
]

# Fase 3 (pospuesta): las tres candidatas con todo train y val completo, hasta 10 épocas (~7-8 h en total).
# Para lanzarla: EXPERIMENTS = PHASE_3.
FULL_DATA = {"train_limit": None, "val_limit": None, "epochs": 10}
PHASE_3 = [
    {"name": "f3_c2_dropout_05_filtros_128_256", "model": {"dropout": 0.5, "filters": (128, 256)}, **FULL_DATA},
    {"name": "f3_d4_dropout_dense_06", "model": {"dropout": 0.6}, **FULL_DATA},
    {"name": "f3_c1_dropout_05_dense_2048", "model": {"dropout": 0.5, "dense_units": 2048}, **FULL_DATA},
]

EXPERIMENTS = PHASE_2

if __name__ == "__main__":
    base = {
        "model": {},
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "train_limit": TRAIN_LIMIT,
        "val_limit": VAL_LIMIT,
        "evaluate_test": EVALUATE_TEST,
    }
    for number, experiment in enumerate(EXPERIMENTS, start=1):
        config = {**base, **experiment}
        config["model_path"] = f"models/{config['name']}.keras"
        print(f"\nCorrida {number}/{len(EXPERIMENTS)}: {config['name']}", flush=True)
        run_experiment(config)
