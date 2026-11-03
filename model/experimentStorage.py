import csv
import json
import os
from datetime import datetime

RESULTS_DIRECTORY = "resultados"
REGISTRY_NAME = "registro_ejecuciones.csv"


def save_run(config, history, metrics, directory=RESULTS_DIRECTORY):
    """Guarda la configuración, las métricas y el historial de una corrida.

    Solo las corridas con validación van al registro, para que no mezcle cifras de test con las de val.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(directory, f"{config['name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "metricas.json"), "w", encoding="utf-8") as file:
        json.dump({"timestamp": timestamp, "config": config, **metrics}, file, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "historial.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    if "por_corte" in metrics:
        add_to_registry(directory, timestamp, config, history, metrics)

    print(f"\nCorrida guardada en: {run_dir}")
    return run_dir


def add_to_registry(directory, timestamp, config, history, metrics):
    all_positions = metrics["por_corte"]["todas"]
    row = {
        "timestamp": timestamp,
        "modelo": config["name"],
        "posiciones_train": metrics["posiciones_train"],
        "epocas": len(history["loss"]),
        "mejor_epoca": metrics["mejor_epoca"],
        "val_loss": metrics["val_loss"],
        "perplejidad": metrics["perplejidad"],
        "top1": all_positions["top1"],
        "top3": all_positions["top3"],
        "top5": all_positions["top5"],
        "top1_ilegal": all_positions["top1_ilegal"],
        "minutos": metrics["minutos"],
    }
    registry = os.path.join(directory, REGISTRY_NAME)
    is_new = not os.path.exists(registry)
    with open(registry, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
