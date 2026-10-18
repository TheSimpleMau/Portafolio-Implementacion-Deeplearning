import functools
import io
import json

import chess
import chess.pgn
import numpy as np

import eda
import plotUtil
from etlProcess import transform_data as etl_transform_data
from preprocessing import encode_move, encode_position, legal_moves_info

# Sin filtro de Elo ni muestreo, para no depender de la configuración del conjunto vigente.
transform_data = functools.partial(etl_transform_data, min_elo=0, game_sample_rate=1.0)

GAME = """[Event "Rated Blitz game"]
[WhiteElo "1510"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. O-O f6 1-0

"""


def _position(fen, uci):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    squares, castling, ep = encode_position(board)
    indices, captures = legal_moves_info(board)
    return squares, castling, ep, encode_move(move, board.turn == chess.BLACK), indices, captures


def _stack(positions):
    squares, castling, ep, move, indices, captures = zip(*positions)
    offsets = np.concatenate([[0], np.cumsum([len(i) for i in indices])])
    return (np.stack(squares), np.array(castling), np.array(ep), np.array(move),
            offsets, np.concatenate(indices), np.concatenate(captures))


def test_pieza_movida_y_conteo_de_piezas():
    squares, _, _, move, *_ = _stack([
        _position(chess.STARTING_FEN, "g1f3"),
        _position(chess.STARTING_FEN, "e2e4"),
    ])
    assert eda.moved_piece(squares, move).tolist() == [chess.KNIGHT, chess.PAWN]
    assert eda.piece_count(squares).tolist() == [32, 32]


def test_pieza_movida_con_negras_usa_el_tablero_orientado():
    board = chess.Board()
    board.push_uci("e2e4")
    squares, _, _, move, *_ = _stack([_position(board.fen(), "g8f6")])
    assert eda.moved_piece(squares, move).tolist() == [chess.KNIGHT]


def test_fases_por_ply_y_por_piezas():
    ply = np.array([0, 30, 30, 5])
    n_pieces = np.array([32, 32, 10, 10])
    assert eda.game_phase(ply, n_pieces).tolist() == [0, 1, 2, 2]


def test_valor_de_la_captura_jugada():
    fen = "4k3/8/8/3q4/8/8/3Q4/4K3 w - - 0 1"
    _, _, _, move, offsets, legal, capture = _stack([
        _position(fen, "d2d5"),
        _position(fen, "d2d3"),
        _position(chess.STARTING_FEN, "e2e4"),
    ])
    assert eda.played_capture_value(move, offsets, legal, capture).tolist() == [9, 0, 0]
    assert eda.max_capture_value(offsets, capture).tolist() == [9, 9, 0]


def test_llaves_de_posicion():
    board = chess.Board()
    no_castling = chess.Board(chess.STARTING_FEN.replace("KQkq", "-"))
    squares, castling, ep, *_ = _stack([
        _position(board.fen(), "e2e4"),
        _position(board.fen(), "d2d4"),
        _position(no_castling.fen(), "e2e4"),
    ])
    keys = eda.position_keys(squares, castling, ep)
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


def test_jugada_modal_y_techo_de_acierto():
    keys = np.array([0, 0, 0, 1])
    move = np.array([5, 5, 7, 9])
    assert eda.mode_moves(keys, move).tolist() == [5, 9]
    assert eda.accuracy_ceiling(keys, move).tolist() == [True, True, False, True]


def test_repetidas_solo_cuentan_entre_partidas_distintas():
    keys = np.array([0, 0, 1, 1, 2])
    game_id = np.array([0, 1, 2, 2, 3])  # la llave 1 se repite dentro de la misma partida
    assert eda.repeated_across_games(keys, game_id).tolist() == [True, True, False, False, False]


def test_initial_graphs_de_punta_a_punta(tmp_path, monkeypatch):
    headers = dict(chess.pgn.read_game(io.StringIO(GAME)).headers)
    data, _ = transform_data(iter([(headers, GAME)] * 30), positions_per_game=100)
    monkeypatch.setattr(plotUtil, "GRAPHICS_DIR", str(tmp_path))

    summary = eda.initial_graphs(data, tmp_path / "resumen.json")

    assert summary["posiciones"] == 30 * 10
    assert summary["clases_que_aparecen"] == 9  # 1.e4 y 1...e5 son la misma clase al orientar el tablero
    assert summary["repetidas_entre_partidas"] == 1.0
    assert summary["techo_top1_en_repetidas"] == 1.0
    assert summary["jugada_es_captura"] == 0.2  # Bxc6 y dxc6
    assert len(list(tmp_path.glob("*.png"))) == 8
    assert json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8")) == summary
