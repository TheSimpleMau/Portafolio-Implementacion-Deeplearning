import functools
import io

import chess
import chess.pgn
import numpy as np
import pytest
import zstandard

from model.etlProcess import filter_raw, read_filtered_games, read_games, split_data
from model.etlProcess import transform_data as etl_transform_data
from model.preprocessing import decode_move

# Sin filtro de Elo ni muestreo, para no depender de la configuración del conjunto vigente.
transform_data = functools.partial(etl_transform_data, min_elo=0, game_sample_rate=1.0)

PGN = """[Event "Rated Blitz game"]
[WhiteElo "1510"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

[Event "Rated Bullet game"]
[WhiteElo "1510"]
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
[WhiteElo "1510"]
[BlackElo "1600"]
[Result "1-0"]

1-0

"""

USED_GAMES = [
    (1510, 1600, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]),
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


def test_muestreo_de_posiciones_por_partida_reproducible(pgn_path):
    data, counters = transform_data(read_games(pgn_path), positions_per_game=3)
    again, _ = transform_data(read_games(pgn_path), positions_per_game=3)

    assert counters["posiciones"] == 6
    for game_id in (0, 1):
        plies = data["ply"][data["game_id"] == game_id]
        assert len(set(plies.tolist())) == 3
    for key in data:
        assert np.array_equal(data[key], again[key])


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


def test_excluye_posiciones_de_jugadores_con_elo_1500(tmp_path):
    pgn = (
        '[Event "Rated Blitz game"]\n[WhiteElo "1500"]\n[BlackElo "1600"]\n[Result "1-0"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 1-0\n\n"
    )
    path = tmp_path / "elo1500.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(pgn.encode("utf-8")))
    data, counters = transform_data(read_games(path), positions_per_game=100)

    assert counters["posiciones_elo_excluido"] == 2
    assert counters["posiciones"] == 2
    assert data["ply"].tolist() == [1, 3]
    assert set(data["elo"].tolist()) == {1600}


def test_read_games_separa_cabeceras_sin_analizar_jugadas(pgn_path):
    games = list(read_games(pgn_path))
    assert len(games) == 5
    headers, pgn = games[3]
    assert headers == {
        "Event": "Rated Classical game",
        "WhiteElo": "1800",
        "BlackElo": "1750",
        "Result": "0-1",
    }
    moves = [move.uci() for move in chess.pgn.read_game(io.StringIO(pgn)).mainline_moves()]
    assert moves == USED_GAMES[1][2]


def test_filtro_de_elo_minimo_de_ambos_jugadores(pgn_path):
    data, counters = transform_data(read_games(pgn_path), positions_per_game=100, min_elo=1700)
    assert counters["partidas_usadas"] == 1
    assert counters["elo_bajo"] == 2
    assert counters["bullet"] == 1
    assert counters["sin_elo"] == 1
    assert counters["posiciones"] == 8
    assert set(data["elo"].tolist()) == {1800, 1750}


LICHESS_GAME = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcdefgh"]
[WhiteElo "2300"]
[BlackElo "2250"]
[Result "1-0"]
[TimeControl "180+0"]

1. e4 { [%eval 0.2] [%clk 0:03:00] } 1... c5 { [%eval 0.3] [%clk 0:03:00] } 2. Nf3 { [%clk 0:02:59] } 1-0

"""


def test_muestreo_de_partidas_reproducible(tmp_path):
    path = tmp_path / "recientes.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress((LICHESS_GAME * 40).encode("utf-8")))

    data, counters = transform_data(read_games(path), game_sample_rate=0.5)
    again, _ = transform_data(read_games(path), game_sample_rate=0.5)

    assert 0 < counters["partidas_usadas"] < 40
    assert counters["partidas_usadas"] + counters["fuera_de_muestra"] == 40
    assert counters["posiciones"] == 3 * counters["partidas_usadas"]
    for key in data:
        assert np.array_equal(data[key], again[key])


def test_por_bloques_igual_que_de_una_vez(tmp_path):
    path = tmp_path / "bloques.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress((PGN + LICHESS_GAME * 3).encode("utf-8")))

    expected, _ = transform_data(read_games(path), positions_per_game=100, block_games=1_000)
    data, _ = transform_data(read_games(path), positions_per_game=100, block_games=2)
    assert data.keys() == expected.keys()
    for key in expected:
        assert np.array_equal(data[key], expected[key])


def test_excluye_partidas_de_cuentas_bot(tmp_path):
    bot_game = LICHESS_GAME.replace('[WhiteElo "2300"]', '[WhiteTitle "BOT"]\n[WhiteElo "2300"]')
    path = tmp_path / "bots.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress((bot_game + LICHESS_GAME).encode("utf-8")))
    _, counters = transform_data(read_games(path))
    assert counters["bot"] == 1
    assert counters["partidas_usadas"] == 1


def test_etl_desde_parquet_igual_que_desde_el_crudo(tmp_path):
    path = tmp_path / "mezcla.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress((PGN + LICHESS_GAME * 3).encode("utf-8")))
    parquet = tmp_path / "partidas.parquet"
    counters = filter_raw(path, parquet, tmp_path / "filtros.json", min_elo=0)

    expected, _ = transform_data(read_games(path), positions_per_game=2)
    data, _ = transform_data(read_filtered_games(parquet), positions_per_game=2)
    assert counters["partidas_guardadas"] == 6
    assert counters["bullet"] == 1 and counters["sin_elo"] == 1
    for key in expected:
        assert np.array_equal(data[key], expected[key])
