"""Entrena y evalúa en validación cada corrida de EXPERIMENTS, una tras otra.

Uso, desde la raíz del proyecto con el entorno activado (source .venv/bin/activate):
    python main.py

Las constantes de abajo son la configuración base; cada corrida de EXPERIMENTS puede cambiarla
(su "model" va a get_model; las demás claves reemplazan las de la base). Para correr menos, comentar renglones.
TRAIN_LIMIT / VAL_LIMIT = un número usa esa muestra de train / val; None usa el conjunto completo.
PATIENCE = None quita EarlyStopping.
Cada mejor modelo (menor val_loss) queda en models/<name>.keras, donde lo encuentra el menú del juego.
FINAL = True entrena en cambio FINAL_EXPERIMENT con train+val durante sus épocas fijas y lo evalúa en test (una sola vez).
"""
from model.experiment import run_experiment, run_final

BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 10
PATIENCE = 2
SEED = 42
TRAIN_LIMIT = None
VAL_LIMIT = None
EVALUATE_TEST = False  # test se usa una sola vez, al final del proyecto
FINAL = True

# Las tres configuraciones que mejor resultaron en las pruebas con una muestra de los datos.
EXPERIMENTS = [
    # Con todos los datos en GPU, cada época tarda ~37 min (filtros 128/256) y ~31 min (Dense 2048): así cada corrida dura ≤ 4 h.
    {"name": "dropout_05_filtros_128_256", "model": {"dropout": 0.5, "filters": (128, 256)}, "epochs": 6},
    # {"name": "dropout_06", "model": {"dropout": 0.6}},
    {"name": "dropout_05_dense_2048", "model": {"dropout": 0.5, "dense_units": 2048}, "epochs": 7},
]

# La mejor en validación (top-1 43.1 %), con las épocas de su mejor val_loss.
FINAL_EXPERIMENT = {"name": "final_dropout_05_dense_2048", "model": {"dropout": 0.5, "dense_units": 2048}, "epochs": 6}

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
    if FINAL:
        config = {**base, **FINAL_EXPERIMENT}
        config["model_path"] = f"models/{config['name']}.keras"
        print(f"\nCorrida final: {config['name']}", flush=True)
        run_final(config)
    else:
        for number, experiment in enumerate(EXPERIMENTS, start=1):
            config = {**base, **experiment}
            config["model_path"] = f"models/{config['name']}.keras"
            print(f"\nCorrida {number}/{len(EXPERIMENTS)}: {config['name']}", flush=True)
            run_experiment(config)
