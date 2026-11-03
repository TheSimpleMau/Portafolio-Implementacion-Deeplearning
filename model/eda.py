import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from model import plotUtil
from model.etlProcess import (
    FILTERED_PATH, GAME_SAMPLE_RATE, MIN_ELO, MONTH, POSITIONS_PER_GAME, PROCESSED_PATH, RANDOM_STATE,
    _header_discard_reason, load_data, read_filtered_games,
)
from model.preprocessing import N_CLASSES

SUMMARY_PATH = "graficas/eda_resumen.json"
OPENINGS_SUMMARY_PATH = "graficas/aperturas_resumen.json"
# Aperturas iniciales para medir el Elo (las usa game/elo.py).
ELO_OPENINGS_PATH = "graficas/aperturas_elo.json"
# Apertura de cada partida usada, por game_id (para cruzarla con las posiciones).
GAME_OPENINGS_PATH = f"data/processed/aperturas_{MONTH}.npz"

# Tamaño de la tabla de aperturas de Lichess (lichess-org/chess-openings).
LICHESS_ECO_CODES = 500
LICHESS_OPENING_NAMES = 3177
LICHESS_FAMILIES = 150
ECO_GROUPS = ("A", "B", "C", "D", "E")
SPEEDS = (("Blitz", "Blitz"), ("Rapid", "Rápidas"), ("Classical", "Clásicas y correspondencia"),
          ("Correspondence", "Clásicas y correspondencia"))
TOP_FAMILIES = 20
ELO_OPENING_PLIES = 8

COMMENT = re.compile(r"\{[^}]*\}")
NON_MOVE = re.compile(r"^(\d+\.+|1-0|0-1|1/2-1/2|\*|\$\d+)$")

# Umbrales de fase decididos con el EDA.
OPENING_PLY = 20
ENDGAME_PIECES = 12
PHASE_NAMES = ("apertura", "medio juego", "final")

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


def mode_moves(keys, move):
    """Jugada más frecuente de cada posición (por llave); -1 si la llave no aparece."""
    pairs = keys.astype(np.int64) * N_CLASSES + move
    unique_pairs, counts = np.unique(pairs, return_counts=True)
    pair_keys = unique_pairs // N_CLASSES
    order = np.lexsort((-counts, pair_keys))
    first = order[np.r_[True, np.diff(pair_keys[order]) != 0]]

    modes = np.full(keys.max() + 1, -1, dtype=np.int32)
    modes[pair_keys[first]] = unique_pairs[first] % N_CLASSES
    return modes


def repeated_across_games(keys, game_id):
    """True si la posición aparece en al menos dos partidas distintas."""
    n_games = int(game_id.max()) + 1
    key_game = np.unique(keys.astype(np.int64) * n_games + game_id)
    games_per_key = np.bincount(key_game // n_games, minlength=keys.max() + 1)
    return games_per_key[keys] > 1


def accuracy_ceiling(keys, move):
    """True si la jugada es la modal de su posición (techo de top-1 al ver solo el tablero)."""
    return mode_moves(keys, move)[keys] == move


def _share(values):
    return round(float(values.mean()), 4) if len(values) else None


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

    n_legal = legal_counts(data["legal_offsets"])
    piece = moved_piece(squares, move)
    n_pieces = piece_count(squares)
    phase = game_phase(ply, n_pieces)
    captured = played_capture_value(move, data["legal_offsets"], data["legal_moves"], data["legal_capture"])
    best_capture = max_capture_value(data["legal_offsets"], data["legal_capture"])
    keys = position_keys(squares, data["castling"], data["ep"])
    repeated = repeated_across_games(keys, data["game_id"])
    is_mode = accuracy_ceiling(keys, move)

    plotUtil.elo_hist(elo)
    plotUtil.ply_hist(ply)
    plotUtil.legal_moves_hist(n_legal)
    plotUtil.class_frequency(np.bincount(move, minlength=N_CLASSES))
    plotUtil.moved_piece_bar(piece)
    plotUtil.capture_bar(captured)
    plotUtil.phase_bar(phase, PHASE_NAMES)
    plotUtil.repeated_positions_by_ply(
        _rate_by_ply(ply, repeated),
        _rate_by_ply(ply, is_mode, repeated),
    )

    with_capture = best_capture > 0
    opening = phase == 0
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
        "repetidas_entre_partidas": _share(repeated),
        "repetidas_en_apertura": _share(repeated[opening]),
        "repetidas_fuera_de_apertura": _share(repeated[~opening]),
        "techo_top1_en_repetidas": _share(is_mode[repeated]),
        "techo_top1_posicion_inicial": _share(is_mode[ply == 0]),
    }
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary


def game_moves(movetext):
    """Jugadas en SAN del texto de la partida, sin comentarios, números de jugada ni signos ?!."""
    tokens = COMMENT.sub(" ", movetext).split()
    return [token.rstrip("?!") for token in tokens if not NON_MOVE.match(token)]


def used_games(games, positions_per_game=POSITIONS_PER_GAME, min_elo=MIN_ELO,
               game_sample_rate=GAME_SAMPLE_RATE, random_state=RANDOM_STATE):
    """Repite el sorteo de transform_data sin analizar las jugadas: (game_id, cabeceras, jugadas, plies)."""
    rng = np.random.default_rng(random_state)
    game_id = 0
    for headers, pgn in games:
        if _header_discard_reason(headers, min_elo):
            continue
        if game_sample_rate < 1 and rng.random() >= game_sample_rate:
            continue
        moves = game_moves(pgn.partition("\n\n")[2])
        if not moves:
            continue
        plies = rng.choice(len(moves), size=min(positions_per_game, len(moves)), replace=False)
        yield game_id, headers, moves, plies
        game_id += 1


def opening_family(name):
    return name.split(":")[0]


def game_speed(event):
    return next((label for key, label in SPEEDS if key in event), "Otro")


def _coverage(counts, shares=(0.5, 0.8, 0.95)):
    """Cuántas categorías (de la más a la menos común) cubren cada fracción de las partidas."""
    cumulative = np.cumsum(np.sort(np.asarray(counts))[::-1]) / np.sum(counts)
    return {f"{share:.0%}": int(np.searchsorted(cumulative, share) + 1) for share in shares}


def _shares(series):
    return {str(key): round(float(value), 4) for key, value in series.value_counts(normalize=True).items()}


def dataset_graphs(games, processed_path=PROCESSED_PATH, summary_path=OPENINGS_SUMMARY_PATH,
                   elo_openings_path=ELO_OPENINGS_PATH, game_openings_path=GAME_OPENINGS_PATH):
    """Aperturas, primera jugada, ritmos, terminación y duración de las partidas que usa el ETL."""
    with np.load(processed_path) as file:
        game_id, ply = file["game_id"], file["ply"]
    starts = np.searchsorted(game_id, np.arange(game_id.max() + 2))

    rows, mismatched, missing_plies = [], 0, 0
    for gid, headers, moves, plies in used_games(games):
        stored = set(ply[starts[gid]:starts[gid + 1]].tolist())
        if not stored <= set(plies.tolist()):
            mismatched += 1
        missing_plies += len(plies) - len(stored)
        opening = headers.get("Opening", "?")
        rows.append({
            "eco": headers.get("ECO", "?"),
            "family": opening_family(opening),
            "name": opening,
            "speed": game_speed(headers.get("Event", "")),
            "termination": headers.get("Termination", "?"),
            "plies": len(moves),
            "first": moves[0],
            "reply": " ".join(moves[:2]),
            "line": " ".join(moves[:ELO_OPENING_PLIES]) if len(moves) >= ELO_OPENING_PLIES else None,
        })
    if mismatched or len(rows) != game_id.max() + 1:
        raise RuntimeError(f"El sorteo no coincide con {processed_path}: {mismatched} partidas distintas, "
                           f"{len(rows):,} partidas contra {game_id.max() + 1:,}.")
    games_df = pd.DataFrame(rows)
    games_df["group"] = games_df["eco"].str[0].where(games_df["eco"].str[0].isin(ECO_GROUPS), "?")

    families = games_df["family"].value_counts()
    top = families.index[:TOP_FAMILIES]
    lines = games_df[games_df["family"].isin(top)].dropna(subset=["line"])
    elo_openings = [
        {"familia": family, "jugadas": lines.loc[lines["family"] == family, "line"].mode()[0].split(),
         "partidas": int(families[family])}
        for family in top
    ]
    by_speed = pd.crosstab(games_df["speed"], games_df["group"], normalize="index")
    termination = pd.crosstab(games_df["speed"], games_df["termination"], normalize="index")
    first_moves = games_df["first"].value_counts(normalize=True)
    replies = {first: games_df.loc[games_df["first"] == first, "reply"].str.split().str[1]
               .value_counts(normalize=True) for first in ("e4", "d4")}

    plotUtil.eco_group_bar(games_df["group"].value_counts(normalize=True).reindex(ECO_GROUPS, fill_value=0))
    plotUtil.opening_families_bar(families[:TOP_FAMILIES] / len(games_df))
    plotUtil.opening_cumulative(families.to_numpy())
    plotUtil.first_moves_bar(first_moves, replies)
    plotUtil.openings_by_speed(by_speed.reindex(columns=list(ECO_GROUPS), fill_value=0),
                               games_df["speed"].value_counts())
    plotUtil.termination_bar(termination, games_df["speed"].value_counts())
    plotUtil.game_length_hist(games_df["plies"].to_numpy())

    plies_per_game = games_df["plies"]
    summary = {
        "partidas": len(games_df),
        "partidas_con_un_ply_menos_en_el_npz": int(missing_plies),
        "eco_distintos": int(games_df["eco"].nunique()),
        "eco_en_la_tabla_de_lichess": LICHESS_ECO_CODES,
        "nombres_distintos": int(games_df["name"].nunique()),
        "nombres_en_la_tabla_de_lichess": LICHESS_OPENING_NAMES,
        "familias_distintas": int(len(families)),
        "familias_en_la_tabla_de_lichess": LICHESS_FAMILIES,
        "grupos_eco": _shares(games_df["group"]),
        "familias_que_cubren": _coverage(families.to_numpy()),
        "nombres_que_cubren": _coverage(games_df["name"].value_counts().to_numpy()),
        "familias_principales": {name: round(float(n / len(games_df)), 4) for name, n in families[:TOP_FAMILIES].items()},
        "primera_jugada": {name: round(float(v), 4) for name, v in first_moves[:8].items()},
        "respuesta_a_e4": {name: round(float(v), 4) for name, v in replies["e4"][:6].items()},
        "respuesta_a_d4": {name: round(float(v), 4) for name, v in replies["d4"][:6].items()},
        "ritmos": _shares(games_df["speed"]),
        "grupos_eco_por_ritmo": by_speed.round(4).to_dict(orient="index"),
        "terminacion": _shares(games_df["termination"]),
        "terminacion_por_ritmo": termination.round(4).to_dict(orient="index"),
        "plies_por_partida_media": round(float(plies_per_game.mean()), 1),
        "plies_por_partida_p5_mediana_p95": [int(v) for v in np.percentile(plies_per_game, [5, 50, 95])],
    }
    for path, content in ((summary_path, summary), (elo_openings_path, elo_openings)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(content, file, indent=2, ensure_ascii=False)

    family_names = families.index.to_numpy(dtype=str)
    np.savez(
        game_openings_path,
        eco=games_df["eco"].to_numpy(dtype="U3"),
        family=pd.Categorical(games_df["family"], categories=family_names).codes.astype(np.int16),
        family_names=family_names,
    )
    return summary


if __name__ == "__main__":
    if sys.argv[1:] == ["aperturas"]:
        summary = dataset_graphs(read_filtered_games(FILTERED_PATH))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        sys.exit()
    summary = initial_graphs(load_data(PROCESSED_PATH))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
