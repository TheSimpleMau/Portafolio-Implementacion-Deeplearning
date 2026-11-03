import json
import os
import sys

import numpy as np

from model import eda, plotUtil
from model.etlProcess import PROCESSED_PATH, RANDOM_STATE, load_data, split_data
from model.preprocessing import N_CLASSES, expand_planes

RESULTS_PATH = "resultados/lineas_base.json"
KS = (1, 3, 5)
SPLIT_NAMES = ("train", "val")
OPENINGS_RESULTS_PATH = "resultados/acierto_por_apertura.json"
TOP_OPENING_FAMILIES = 10


def subset_legal(legal_offsets, legal_moves, legal_capture, idx):
    """Jugadas legales (formato CSR) solo de las posiciones idx, en ese orden."""
    counts = np.diff(legal_offsets)[idx]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    gather = np.repeat(legal_offsets[idx] - offsets[:-1], counts) + np.arange(offsets[-1])
    return offsets, legal_moves[gather], legal_capture[gather]


def played_rank(legal_scores, legal_offsets, legal_moves, y):
    """Cuántas jugadas legales tienen más puntaje que la jugada real (0 = acierto top-1)."""
    owner = np.repeat(np.arange(len(y)), np.diff(legal_offsets))
    is_played = legal_moves == y[owner]
    played_score = np.full(len(y), -np.inf)
    played_score[owner[is_played]] = legal_scores[is_played]
    return np.bincount(owner, weights=legal_scores > played_score[owner], minlength=len(y))


def topk_accuracy(ranks, ks=KS):
    return {f"top{k}": float((ranks < k).mean()) for k in ks}


def random_accuracy(legal_offsets, ks=KS):
    """Acierto esperado al elegir una jugada legal al azar: promedio de min(k, n) / n."""
    n = np.diff(legal_offsets)
    return {f"top{k}": float((np.minimum(k, n) / n).mean()) for k in ks}


def tie_break(scores, rng):
    """Desempata al azar puntajes enteros sin cambiar su orden."""
    return scores + rng.random(len(scores)) * 0.5


def frequency_scores(train_move, legal_moves):
    return np.bincount(train_move, minlength=N_CLASSES)[legal_moves].astype(np.float64)


def capture_scores(legal_capture):
    return legal_capture.astype(np.float64)


def book_moves(train_keys, train_move, n_keys):
    """Jugada modal en train de cada llave de posición; -1 si la posición no aparece en train."""
    modes = np.full(n_keys, -1, dtype=np.int32)
    train_modes = eda.mode_moves(train_keys, train_move)
    modes[:len(train_modes)] = train_modes
    return modes


def book_scores(keys, modes, legal_offsets, legal_moves, fallback):
    """La jugada del «libro» va primero; las demás conservan el orden de fallback."""
    owner = np.repeat(np.arange(len(keys)), np.diff(legal_offsets))
    in_book = legal_moves == modes[keys][owner]
    return fallback + in_book * (fallback.max() + 1)


def phase_masks(ply, squares):
    phase = eda.game_phase(ply, eda.piece_count(squares))
    masks = {"todas": np.ones(len(ply), dtype=bool)}
    masks.update({name: phase == code for code, name in enumerate(eda.PHASE_NAMES)})
    return masks


def model_ranks(model, data, idx, batch_size=4096):
    """Rango de la jugada real entre las legales y si el top-1 sin máscara es ilegal.

    Predice por lotes y de cada uno guarda solo las columnas legales: la matriz completa
    de validación (450 mil × 4,096) no cabe en memoria.
    """
    ranks, illegal = [], []
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        x = expand_planes(data["squares"][batch], data["castling"][batch], data["ep"][batch])
        probs = model.predict(x.astype(np.float32), verbose=0)
        offsets, legal_moves, _ = subset_legal(data["legal_offsets"], data["legal_moves"],
                                               data["legal_capture"], batch)
        owner = np.repeat(np.arange(len(batch)), np.diff(offsets))
        ranks.append(played_rank(probs[owner, legal_moves], offsets, legal_moves, data["move"][batch]))
        top1_is_legal = legal_moves == probs.argmax(axis=1)[owner]
        illegal.append(np.bincount(owner, weights=top1_is_legal, minlength=len(batch)) == 0)
    return np.concatenate(ranks), np.concatenate(illegal)


def evaluate_model(model, data, idx, batch_size=4096):
    """Top-k del modelo entre las legales, por corte, y % de top-1 ilegal sin máscara."""
    ranks, illegal = model_ranks(model, data, idx, batch_size)
    results = {"por_corte": {}}
    for cut, mask in phase_masks(data["ply"][idx], data["squares"][idx]).items():
        results["por_corte"][cut] = {"posiciones": int(mask.sum()), **topk_accuracy(ranks[mask]),
                                     "top1_ilegal": float(illegal[mask].mean())}
    return results


def baseline_ranks(data, train, val, seed=RANDOM_STATE):
    """Rangos de las líneas base en val y si cada posición de val ya aparece en train."""
    offsets, legal_moves, legal_capture = subset_legal(
        data["legal_offsets"], data["legal_moves"], data["legal_capture"], val)
    y = data["move"][val]
    rng = np.random.default_rng(seed)

    both = np.concatenate([train, val])
    keys = eda.position_keys(data["squares"][both], data["castling"][both], data["ep"][both])
    modes = book_moves(keys[:len(train)], data["move"][train], keys.max() + 1)
    val_keys = keys[len(train):]
    in_train = modes[val_keys] >= 0

    frequency = tie_break(frequency_scores(data["move"][train], legal_moves), rng)
    ranks = {
        "más frecuente": played_rank(frequency, offsets, legal_moves, y),
        "captura de mayor valor": played_rank(tie_break(capture_scores(legal_capture), rng),
                                              offsets, legal_moves, y),
        "libro": played_rank(book_scores(val_keys, modes, offsets, legal_moves, frequency),
                             offsets, legal_moves, y),
    }
    return ranks, in_train, offsets


def run_baselines(path=PROCESSED_PATH, results_path=RESULTS_PATH, seed=RANDOM_STATE):
    """Líneas base en validación (test no se toca) por fase, y comparación entre train y val."""
    data = load_data(path)
    train, val, _ = split_data(data["game_id"])

    phase = eda.game_phase(data["ply"], eda.piece_count(data["squares"]))
    plotUtil.split_comparison(data["elo"], data["ply"], phase, (train, val), SPLIT_NAMES, eda.PHASE_NAMES)

    ranks, in_train, offsets = baseline_ranks(data, train, val, seed)
    results = {"posiciones_val": len(val), "por_corte": {}}
    for cut, mask in phase_masks(data["ply"][val], data["squares"][val]).items():
        cut_offsets = np.concatenate([[0], np.cumsum(np.diff(offsets)[mask])])
        results["por_corte"][cut] = {
            "posiciones": int(mask.sum()),
            "val_en_train": float(in_train[mask].mean()),
            "azar": random_accuracy(cut_offsets),
            **{name: topk_accuracy(rank[mask]) for name, rank in ranks.items()},
        }

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    return results


def print_table(results):
    for cut, values in results["por_corte"].items():
        print(f"\n{cut} ({values['posiciones']:,} posiciones; {values['val_en_train'] * 100:.1f} % ya en train)")
        for name, accuracy in values.items():
            if isinstance(accuracy, dict):
                row = "  ".join(f"{k} {v * 100:5.1f} %" for k, v in accuracy.items())
                print(f"  {name:<24} {row}")


def topk_by_group(ranks, groups, names, ks=KS):
    """Top-k de las posiciones de cada grupo (groups = código de grupo por posición)."""
    return {name: {"posiciones": int((groups == code).sum()), **topk_accuracy(ranks[groups == code], ks)}
            for code, name in enumerate(names) if (groups == code).any()}


def run_by_opening(model_path, path=PROCESSED_PATH, openings_path=eda.GAME_OPENINGS_PATH,
                   results_path=OPENINGS_RESULTS_PATH, top_families=TOP_OPENING_FAMILIES):
    """Top-k del modelo y del «libro» en val por grupo ECO y por las familias más jugadas."""
    import keras  # TensorFlow tarda en cargar: solo se importa aquí

    data = load_data(path)
    train, val, _ = split_data(data["game_id"])
    with np.load(openings_path) as file:
        eco, family, family_names = file["eco"], file["family"], file["family_names"]
    val_games = data["game_id"][val]

    ranks = {"modelo": model_ranks(keras.models.load_model(model_path), data, val)[0],
             "libro": baseline_ranks(data, train, val)[0]["libro"]}
    group_names = eda.ECO_GROUPS + ("?",)
    group_code = {group: code for code, group in enumerate(group_names)}
    eco_group = np.array([group_code.get(code[:1], group_code["?"]) for code in eco])[val_games]
    # Las familias vienen ordenadas de la más a la menos jugada; el resto queda en «otras».
    val_family = np.minimum(family[val_games], top_families)
    family_labels = tuple(family_names[:top_families]) + ("Otras",)

    results = {"modelo": model_path, "posiciones_val": len(val)}
    for cut, groups, names in (("grupo_eco", eco_group, group_names), ("familia", val_family, family_labels)):
        results[cut] = {name: topk_by_group(rank, groups, names) for name, rank in ranks.items()}
    plotUtil.accuracy_by_opening(results)

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    if sys.argv[1:2] == ["aperturas"]:
        print(json.dumps(run_by_opening(sys.argv[2]), indent=2, ensure_ascii=False))
        sys.exit()
    print_table(run_baselines())
