"""Entrena y evalúa el modelo en validación.

Uso, desde la raíz del proyecto con el entorno activado (source .venv/bin/activate):
    python main.py

La configuración se cambia en las constantes de abajo. TRAIN_LIMIT = un número (p. ej. 400_000)
entrena con esa muestra de train para probar el flujo rápido; None usa todo train.
El mejor modelo (menor val_loss) queda en MODEL_PATH, donde lo encuentra el menú del juego.
"""
from model.experiment import run_experiment

NAME = "m1"
MODEL_PATH = "models/m1.keras"
BATCH_SIZE = 512
EPOCHS = 10
SEED = 42
TRAIN_LIMIT = None
EVALUATE_TEST = False  # test se usa una sola vez, al final del proyecto

if __name__ == "__main__":
    run_experiment({
        "name": NAME,
        "model_path": MODEL_PATH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "train_limit": TRAIN_LIMIT,
        "evaluate_test": EVALUATE_TEST,
    })
