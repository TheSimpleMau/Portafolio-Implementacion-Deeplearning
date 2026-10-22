import io
import json
import os
import time

import chess
import chess.pgn
import numpy as np
import zstandard

from model.preprocessing import encode_move, encode_position, legal_moves_info

# 2013-03 solo sirve para depurar; el conjunto principal es el mes más reciente.
MONTH = "2026-09"
RAW_DATA_PATH = f"data/raw/lichess_db_standard_rated_{MONTH}.pgn.zst"
PROCESSED_PATH = f"data/processed/positions_{MONTH}.npz"
SUMMARY_PATH = f"data/processed/positions_{MONTH}_resumen.json"

POSITIONS_PER_GAME = 12
# También excluye UltraBullet.
EXCLUDED_EVENTS = ("Bullet",)
# Cuentas de motores: sus jugadas no son humanas.
EXCLUDED_TITLES = ("BOT",)
# Elo mínimo de ambos jugadores.
MIN_ELO = 2200
# Fracción de las partidas que pasan los filtros que se analizan.
GAME_SAMPLE_RATE = 0.25
# Pico en 1500: probablemente jugadores nuevos con la calificación inicial.
EXCLUDED_MOVER_ELOS = (1500,)

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
RANDOM_STATE = 42

PROGRESS_EVERY = 10_000


def read_games(path):
    """Separa el PGN en partidas sin analizar las jugadas: (cabeceras, texto)."""
    with open(path, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        headers, lines = {}, []
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.startswith("[Event ") and lines:
                yield headers, "".join(lines)
                headers, lines = {}, []
            if line.startswith("["):
                key, _, value = line.strip()[1:-1].partition(" ")
                headers[key] = value.strip('"')
            lines.append(line)
        if lines:
            yield headers, "".join(lines)


def _parse_elo(value):
    return int(value) if value and value.isdigit() else None


def _header_discard_reason(headers, min_elo):
    if any(event in headers.get("Event", "") for event in EXCLUDED_EVENTS):
        return "bullet"
    if headers.get("WhiteTitle") in EXCLUDED_TITLES or headers.get("BlackTitle") in EXCLUDED_TITLES:
        return "bot"
    elos = (_parse_elo(headers.get("WhiteElo")), _parse_elo(headers.get("BlackElo")))
    if None in elos:
        return "sin_elo"
    if min(elos) < min_elo:
        return "elo_bajo"
    return None


def _game_discard_reason(game):
    if game.errors:
        return "error_pgn"
    if game.next() is None:
        return "sin_jugadas"
    return None


def transform_data(games, positions_per_game=POSITIONS_PER_GAME, min_elo=MIN_ELO,
                   game_sample_rate=GAME_SAMPLE_RATE, random_state=RANDOM_STATE,
                   max_games=None):
    rng = np.random.default_rng(random_state)
    counters = {
        "partidas_leidas": 0,
        "partidas_usadas": 0,
        "bullet": 0,
        "bot": 0,
        "sin_elo": 0,
        "elo_bajo": 0,
        "fuera_de_muestra": 0,
        "error_pgn": 0,
        "sin_jugadas": 0,
        "subpromociones": 0,
        "posiciones_elo_excluido": 0,
    }
    columns = {name: [] for name in ("squares", "castling", "ep", "move", "game_id", "ply", "elo")}
    legal_moves, legal_capture, legal_counts = [], [], []
    started = time.perf_counter()

    for headers, pgn in games:
        if max_games is not None and counters["partidas_leidas"] >= max_games:
            break
        counters["partidas_leidas"] += 1
        if counters["partidas_leidas"] % PROGRESS_EVERY == 0:
            print(
                f"{counters['partidas_leidas']:,} partidas leídas, "
                f"{len(columns['move']):,} posiciones "
                f"({time.perf_counter() - started:.0f} s)"
            )

        reason = _header_discard_reason(headers, min_elo)
        if reason:
            counters[reason] += 1
            continue
        # Solo se sortea con muestreo para que con 1.0 los resultados no cambien.
        if game_sample_rate < 1 and rng.random() >= game_sample_rate:
            counters["fuera_de_muestra"] += 1
            continue
        game = chess.pgn.read_game(io.StringIO(pgn))
        reason = _game_discard_reason(game)
        if reason:
            counters[reason] += 1
            continue

        game_id = counters["partidas_usadas"]
        counters["partidas_usadas"] += 1
        elos = {
            chess.WHITE: _parse_elo(game.headers["WhiteElo"]),
            chess.BLACK: _parse_elo(game.headers["BlackElo"]),
        }
        moves = list(game.mainline_moves())
        n_samples = min(positions_per_game, len(moves))
        sampled = set(rng.choice(len(moves), size=n_samples, replace=False).tolist())
        last_ply = max(sampled)

        board = game.board()
        for ply, move in enumerate(moves):
            if ply in sampled:
                index = encode_move(move, board.turn == chess.BLACK)
                if elos[board.turn] in EXCLUDED_MOVER_ELOS:
                    counters["posiciones_elo_excluido"] += 1
                elif index is None:
                    counters["subpromociones"] += 1
                else:
                    squares, castling, ep = encode_position(board)
                    indices, captures = legal_moves_info(board)
                    columns["squares"].append(squares)
                    columns["castling"].append(castling)
                    columns["ep"].append(ep)
                    columns["move"].append(index)
                    columns["game_id"].append(game_id)
                    columns["ply"].append(ply)
                    columns["elo"].append(elos[board.turn])
                    legal_moves.append(indices)
                    legal_capture.append(captures)
                    legal_counts.append(len(indices))
            if ply == last_ply:
                break
            board.push(move)

    data = {
        "squares": np.stack(columns["squares"]).astype(np.int8),
        "castling": np.array(columns["castling"], dtype=np.uint8),
        "ep": np.array(columns["ep"], dtype=np.int8),
        "move": np.array(columns["move"], dtype=np.int16),
        "game_id": np.array(columns["game_id"], dtype=np.int32),
        "ply": np.array(columns["ply"], dtype=np.int16),
        "elo": np.array(columns["elo"], dtype=np.int16),
        "legal_offsets": np.concatenate([[0], np.cumsum(legal_counts)]).astype(np.int64),
        "legal_moves": np.concatenate(legal_moves).astype(np.int16),
        "legal_capture": np.concatenate(legal_capture).astype(np.int8),
    }
    counters["posiciones"] = len(data["move"])
    counters["duracion_segundos"] = round(time.perf_counter() - started, 1)
    return data, counters


def save_data(data, counters, path=PROCESSED_PATH, summary_path=SUMMARY_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **data)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(counters, file, indent=2, ensure_ascii=False)


def load_data(path=PROCESSED_PATH):
    with np.load(path) as file:
        return {key: file[key] for key in file.files}


def split_data(game_ids, train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION,
               test_fraction=TEST_FRACTION, random_state=RANDOM_STATE):
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Las proporciones de split deben sumar 1.")

    games = np.unique(game_ids)
    n = len(games)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)

    split_of_game = np.empty(n, dtype=np.int8)
    order = np.random.default_rng(random_state).permutation(n)
    split_of_game[order[:train_end]] = 0
    split_of_game[order[train_end:val_end]] = 1
    split_of_game[order[val_end:]] = 2

    position_split = split_of_game[np.searchsorted(games, game_ids)]
    return tuple(np.flatnonzero(position_split == split) for split in range(3))


def run_etl(raw_path=RAW_DATA_PATH, processed_path=PROCESSED_PATH):
    if os.path.exists(processed_path):
        print("Datos ya transformados. Saltando transformación.")
        return load_data(processed_path)

    print(f"Leyendo {raw_path}...")
    data, counters = transform_data(read_games(raw_path))
    save_data(data, counters, processed_path)
    print("\nResumen del ETL:")
    for name, value in counters.items():
        print(f"  {name}: {value:,}")
    print(f"Datos guardados en {processed_path}")
    return data


if __name__ == "__main__":
    data = run_etl()
    train, val, test = split_data(data["game_id"])
    print("\nPosiciones por conjunto (split por partida):")
    print(f"Train:      {len(train):,}")
    print(f"Validation: {len(val):,}")
    print(f"Test:       {len(test):,}")
