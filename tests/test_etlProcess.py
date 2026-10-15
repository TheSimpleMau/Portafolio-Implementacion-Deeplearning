import chess
import numpy as np
import pytest
import zstandard

from etlProcess import load_data, read_games, save_data, split_data, transform_data
from preprocessing import decode_move

PGN = """[Event "Rated Blitz game"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

[Event "Rated Bullet game"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1. d4 d5 1-0

[Event "Rated Blitz game"]
[WhiteElo "?"]
[BlackElo "1600"]
[Result "0-1"]

1. c4 e5 0-1

[Event "Rated Classical game"]
[WhiteElo "1800"]
[BlackElo "1750"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 0-1

[Event "Rated Blitz game"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1-0

"""

USED_GAMES = [
    (1500, 1600, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]),
    (1800, 1750, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1", "g8f6"]),
]


@pytest.fixture
def pgn_path(tmp_path):
    path = tmp_path / "partidas.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(PGN.encode("utf-8")))
    return path


def test_contadores_de_filtros(pgn_path):
    _, counters = transform_data(read_games(pgn_path), positions_per_game=100)
    assert counters["partidas_leidas"] == 5
    assert counters["partidas_usadas"] == 2
    assert counters["bullet"] == 1
    assert counters["sin_elo"] == 1
    assert counters["sin_jugadas"] == 1
    assert counters["posiciones"] == 6 + 8


def test_posiciones_y_etiquetas_coinciden_con_la_partida(pgn_path):
    data, _ = transform_data(read_games(pgn_path), positions_per_game=100)

    i = 0
    for game_id, (white_elo, black_elo, moves) in enumerate(USED_GAMES):
        board = chess.Board()
        for ply, uci in enumerate(moves):
            assert data["game_id"][i] == game_id
            assert data["ply"][i] == ply
            assert data["elo"][i] == (white_elo if board.turn == chess.WHITE else black_elo)
            assert decode_move(data["move"][i], board).uci() == uci

            start, end = data["legal_offsets"][i], data["legal_offsets"][i + 1]
            legal = data["legal_moves"][start:end]
            assert data["move"][i] in legal
            assert len(legal) == board.legal_moves.count()
            assert len(data["legal_capture"][start:end]) == len(legal)

            board.push_uci(uci)
            i += 1
    assert i == len(data["move"])


def test_muestreo_por_partida_es_reproducible(pgn_path):
    data, counters = transform_data(read_games(pgn_path), positions_per_game=3)
    again, _ = transform_data(read_games(pgn_path), positions_per_game=3)

    assert counters["posiciones"] == 6
    for game_id in (0, 1):
        plies = data["ply"][data["game_id"] == game_id]
        assert len(set(plies.tolist())) == 3
    for key in data:
        assert np.array_equal(data[key], again[key])


def test_limite_de_partidas(pgn_path):
    _, counters = transform_data(read_games(pgn_path), max_games=1)
    assert counters["partidas_leidas"] == 1


def test_guardar_y_cargar(pgn_path, tmp_path):
    data, counters = transform_data(read_games(pgn_path))
    path = tmp_path / "procesado.npz"
    save_data(data, counters, path, tmp_path / "resumen.json")
    loaded = load_data(path)
    assert loaded.keys() == data.keys()
    for key in data:
        assert np.array_equal(loaded[key], data[key])
        assert loaded[key].dtype == data[key].dtype


def test_split_por_partida():
    game_ids = np.repeat(np.arange(100), 12)
    train, val, test = split_data(game_ids)

    sets = [set(game_ids[idx].tolist()) for idx in (train, val, test)]
    assert [len(s) for s in sets] == [80, 10, 10]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(train) + len(val) + len(test) == len(game_ids)
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(len(game_ids)))

    again = split_data(game_ids)
    for a, b in zip((train, val, test), again):
        assert np.array_equal(a, b)


def test_split_rechaza_proporciones_invalidas():
    with pytest.raises(ValueError):
        split_data(np.arange(10), 0.8, 0.1, 0.2)
