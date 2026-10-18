import json
import os

import numpy as np

import plotUtil
from etlProcess import PROCESSED_PATH, load_data, split_data
from preprocessing import N_CLASSES

SUMMARY_PATH = "graficas/eda_resumen.json"

# Umbrales provisionales de fase; se ajustan después de ver las gráficas.
OPENING_PLY = 20
ENDGAME_PIECES = 12
PHASE_NAMES = ("apertura", "medio juego", "final")
SPLIT_NAMES = ("train", "val", "test")

MAX_PLY_PLOT = 40
MIN_POSITIONS_PER_PLY = 100


def moved_piece(squares, move):
    return squares[np.arange(len(move)), move // 64]


def piece_count(squares):
    return (squares != 0).sum(axis=1)


def game_phase(ply, n_pieces, opening_ply=OPENING_PLY, endgame_pieces=ENDGAME_PIECES):
    phase = np.ones(len(ply), dtype=np.int8)
    phase[ply < opening_ply] = 0
    phase[n_pieces <= endgame_pieces] = 2
    return phase


def legal_counts(legal_offsets):
    return np.diff(legal_offsets)


def played_capture_value(move, legal_offsets, legal_moves, legal_capture):
    counts = legal_counts(legal_offsets)
    owner = np.repeat(np.arange(len(move)), counts)
    is_played = legal_moves == move[owner]
    values = np.zeros(len(move), dtype=np.int8)
    values[owner[is_played]] = legal_capture[is_played]
    return values


def max_capture_value(legal_offsets, legal_capture):
    return np.maximum.reduceat(legal_capture, legal_offsets[:-1])


def position_keys(squares, castling, ep):
    rows = np.ascontiguousarray(
        np.column_stack([squares, castling, ep]).astype(np.int8)
    )
    as_bytes = rows.view(np.dtype((np.void, rows.shape[1])))[:, 0]
    _, keys = np.unique(as_bytes, return_inverse=True)
    return keys


def book_moves(keys, move):
    """Jugada más frecuente de cada posición (por llave); -1 si la llave no aparece."""
    pairs = keys.astype(np.int64) * N_CLASSES + move
    unique_pairs, counts = np.unique(pairs, return_counts=True)
    pair_keys = unique_pairs // N_CLASSES
    order = np.lexsort((-counts, pair_keys))
    first = order[np.r_[True, np.diff(pair_keys[order]) != 0]]

    book = np.full(keys.max() + 1, -1, dtype=np.int32)
    book[pair_keys[first]] = unique_pairs[first] % N_CLASSES
    return book


def repeated_positions(keys, move, train_idx, eval_idx):
    """Qué posiciones de evaluación ya estaban en train y cuánto acierta el «libro»."""
    book = book_moves(keys[train_idx], move[train_idx])
    book = np.pad(book, (0, keys.max() + 1 - len(book)), constant_values=-1)
    predicted = book[keys[eval_idx]]
    seen = predicted >= 0
    hit = seen & (predicted == move[eval_idx])
    return seen, hit


def accuracy_ceiling(keys, move):
    """Por posición repetida: True si su jugada es la modal de esa posición (techo de top-1)."""
    key_counts = np.bincount(keys)
    repeated = key_counts[keys] > 1
    is_mode = book_moves(keys, move)[keys] == move
    return repeated, is_mode


def _rate_by_ply(ply, values, mask=None, max_ply=MAX_PLY_PLOT, min_positions=MIN_POSITIONS_PER_PLY):
    mask = np.ones(len(ply), bool) if mask is None else mask
    rates = np.full(max_ply + 1, np.nan)
    for p in range(max_ply + 1):
        at_ply = (ply == p) & mask
        if at_ply.sum() >= min_positions:
            rates[p] = values[at_ply].mean()
    return rates


def initial_graphs(data, summary_path=SUMMARY_PATH):
    plotUtil.ensure_graphics_directory()
    squares, move, ply, elo = data["squares"], data["move"], data["ply"], data["elo"]
    splits = split_data(data["game_id"])
    train_idx, val_idx, _ = splits
    split_of = np.empty(len(move), dtype=np.int8)
    for code, idx in enumerate(splits):
        split_of[idx] = code

    n_legal = legal_counts(data["legal_offsets"])
    piece = moved_piece(squares, move)
    n_pieces = piece_count(squares)
    phase = game_phase(ply, n_pieces)
    captured = played_capture_value(move, data["legal_offsets"], data["legal_moves"], data["legal_capture"])
    best_capture = max_capture_value(data["legal_offsets"], data["legal_capture"])
    keys = position_keys(squares, data["castling"], data["ep"])
    seen, book_hit = repeated_positions(keys, move, train_idx, val_idx)
    repeated, is_mode = accuracy_ceiling(keys, move)

    plotUtil.elo_hist(elo)
    plotUtil.ply_hist(ply)
    plotUtil.legal_moves_hist(n_legal)
    plotUtil.class_frequency(np.bincount(move, minlength=N_CLASSES))
    plotUtil.moved_piece_bar(piece)
    plotUtil.capture_bar(captured)
    plotUtil.phase_bar(phase, PHASE_NAMES)
    plotUtil.split_comparison(elo, ply, phase, split_of, SPLIT_NAMES, PHASE_NAMES)
    plotUtil.repeated_positions_by_ply(
        _rate_by_ply(ply[val_idx], seen),
        _rate_by_ply(ply[val_idx], book_hit),
        _rate_by_ply(ply, is_mode, repeated),
    )

    with_capture = best_capture > 0
    summary = {
        "posiciones": int(len(move)),
        "legales_media": round(float(n_legal.mean()), 2),
        "legales_min_max": [int(n_legal.min()), int(n_legal.max())],
        "azar_top1": round(float((1 / n_legal).mean()), 4),
        "ply_mayor_a_10": round(float((ply > 10).mean()), 4),
        "elo_p5_mediana_p95": [int(v) for v in np.percentile(elo, [5, 50, 95])],
        "elo_min_max": [int(elo.min()), int(elo.max())],
        "clases_que_aparecen": int((np.bincount(move, minlength=N_CLASSES) > 0).sum()),
        "pieza_movida": {name: round(float((piece == code).mean()), 4) for code, name in enumerate(
            ("peón", "caballo", "alfil", "torre", "dama", "rey"), start=1)},
        "jugada_es_captura": round(float((captured > 0).mean()), 4),
        "hay_captura_disponible": round(float(with_capture.mean()), 4),
        "elige_captura_de_mayor_valor_si_hay": round(float((captured[with_capture] == best_capture[with_capture]).mean()), 4),
        "fases": {name: round(float((phase == code).mean()), 4) for code, name in enumerate(PHASE_NAMES)},
        "umbrales_fase": {"apertura_ply_menor_a": OPENING_PLY, "final_piezas_hasta": ENDGAME_PIECES},
        "val_vista_en_train": round(float(seen.mean()), 4),
        "libro_top1_val": round(float(book_hit.mean()), 4),
        "libro_top1_val_si_vista": round(float(book_hit[seen].mean()), 4),
        "posiciones_repetidas": round(float(repeated.mean()), 4),
        "techo_top1_en_repetidas": round(float(is_mode[repeated].mean()), 4),
        "por_split": {
            name: {
                "posiciones": int(len(idx)),
                "elo_media": round(float(elo[idx].mean()), 1),
                "ply_media": round(float(ply[idx].mean()), 2),
                "fases": [round(float((phase[idx] == c).mean()), 4) for c in range(3)],
            }
            for name, idx in zip(SPLIT_NAMES, splits)
        },
    }
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    summary = initial_graphs(load_data(PROCESSED_PATH))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
