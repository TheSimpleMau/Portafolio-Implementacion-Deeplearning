import math
import time
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import evaluation, experimentStorage, plotUtil
from model.chessModels import compile_model, get_model
from model.etlProcess import RANDOM_STATE, load_data, split_data
from model.preprocessing import expand_planes

PROGRESS_EVERY = 0.05  # fracción de la época entre líneas de avance


def batch_generator(data, idx, batch_size, shuffle=False, seed=None, epochs=None):
    """Lotes (tablero one-hot en float32, jugada) sin expandir todo el conjunto en memoria.

    Con epochs, imprime el avance dentro de cada época (lo lee el mod «entrenamiento»).
    """
    rng = np.random.default_rng(seed)
    n_steps = steps(idx, batch_size)
    every = max(1, round(n_steps * PROGRESS_EVERY))
    epoch, step1_end, seconds_per_step, epoch_seconds, val_seconds = 0, None, None, None, None
    while True:
        epoch += 1
        order = rng.permutation(idx) if shuffle else idx
        for step, start in enumerate(range(0, len(order), batch_size), start=1):
            batch = order[start:start + batch_size]
            x = expand_planes(data["squares"][batch], data["castling"][batch], data["ep"][batch])
            given = time.time()
            yield x.astype(np.float32), data["move"][batch].astype(np.int32)
            if not epochs:
                continue
            now = time.time()
            if step == 1:
                # Keras pide el primer lote de una época antes de validar la anterior: esa pausa es la validación.
                if step1_end is not None:
                    epoch_seconds, val_seconds = now - step1_end, now - given - seconds_per_step
                step1_end = now
            elif step % every == 0 or step == n_steps:
                seconds_per_step = (now - step1_end) / (step - 1)
                print(progress_line(epoch, epochs, step, n_steps, seconds_per_step,
                                    epoch_seconds, val_seconds), flush=True)


def format_duration(seconds):
    minutes = round(seconds / 60)
    if seconds < 60:
        return f"{round(seconds)} s"
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60} min"


def progress_line(epoch, epochs, step, n_steps, seconds_per_step, epoch_seconds=None, val_seconds=None):
    """Avance y tiempo restante; en la primera época todavía no se sabe cuánto tarda la validación."""
    epoch_left = (n_steps - step) * seconds_per_step + (val_seconds or 0)
    full_epoch = epoch_seconds or n_steps * seconds_per_step + (val_seconds or 0)
    total_left = epoch_left + (epochs - epoch) * full_epoch
    finish = datetime.now() + timedelta(seconds=total_left)
    return (f"Progreso: época {epoch}/{epochs} · paso {step}/{n_steps} · {seconds_per_step * 1000:.0f} ms/paso"
            f" · época: faltan {format_duration(epoch_left)} · total: faltan {format_duration(total_left)}"
            f" · termina {finish:%H:%M}")


def steps(idx, batch_size):
    return math.ceil(len(idx) / batch_size)


def train_model(model, data, train_idx, val_idx, config):
    checkpoint = ModelCheckpoint(filepath=config["model_path"], monitor="val_loss",
                                 save_best_only=True, verbose=1)
    callbacks = [checkpoint]
    if config["patience"]:
        # El checkpoint ya guarda la mejor época: EarlyStopping solo corta la corrida cuando val_loss deja de mejorar.
        callbacks.append(EarlyStopping(monitor="val_loss", patience=config["patience"], verbose=1))
    history = model.fit(batch_generator(data, train_idx, config["batch_size"], shuffle=True, seed=config["seed"],
                                        epochs=config["epochs"]),
                        steps_per_epoch=steps(train_idx, config["batch_size"]),
                        validation_data=batch_generator(data, val_idx, config["batch_size"]),
                        validation_steps=steps(val_idx, config["batch_size"]),
                        epochs=config["epochs"],
                        shuffle=False,  # el generador ya revuelve
                        callbacks=callbacks,
                        verbose=2)
    return history


def training_summary(history):
    """Mejor época (la que guarda el checkpoint), perplejidad y brecha entre train y val."""
    best = int(np.argmin(history["val_loss"]))
    return {
        "mejor_epoca": best + 1,
        "val_loss": history["val_loss"][best],
        "perplejidad": math.exp(history["val_loss"][best]),
        "brecha_loss": history["val_loss"][best] - history["loss"][best],
        "brecha_accuracy": history["accuracy"][best] - history["val_accuracy"][best],
    }


def print_metrics(metrics):
    print(f"\nMejor época: {metrics['mejor_epoca']}  val_loss {metrics['val_loss']:.3f}  "
          f"perplejidad {metrics['perplejidad']:.1f}  brecha de accuracy {metrics['brecha_accuracy'] * 100:.1f} pts")
    for cut, values in metrics["por_corte"].items():
        row = "  ".join(f"{name} {value * 100:5.1f} %" for name, value in values.items() if name != "posiciones")
        print(f"  {cut:<28} {row}")


def run_experiment(config):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    data = load_data()
    train, val, test = split_data(data["game_id"])
    if config["train_limit"]:
        rng = np.random.default_rng(config["seed"])
        train = np.sort(rng.choice(train, config["train_limit"], replace=False))
    if config["val_limit"]:
        # Semilla fija: todas las corridas se comparan con la misma muestra de validación.
        rng = np.random.default_rng(RANDOM_STATE)
        val = np.sort(rng.choice(val, config["val_limit"], replace=False))

    model = get_model(**config["model"])
    compile_model(model, config["learning_rate"])
    model.summary()

    start = time.time()
    history = train_model(model, data, train, val, config).history
    minutes = (time.time() - start) / 60

    model = tf.keras.models.load_model(config["model_path"])
    metrics = {"posiciones_train": len(train), "posiciones_val": len(val), "minutos": minutes, **training_summary(history),
               **evaluation.evaluate_model(model, data, val)}
    if config["evaluate_test"]:
        metrics["test"] = evaluation.evaluate_model(model, data, test)

    plotUtil.training_history(history, config["name"])
    experimentStorage.save_run(config, history, metrics)
    print_metrics(metrics)
    return metrics
