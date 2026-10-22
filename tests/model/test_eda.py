import chess
import numpy as np

from model import eda
from model.preprocessing import encode_move, encode_position, legal_moves_info


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


def test_fases_por_ply_y_por_piezas():
    ply = np.array([0, 30, 30, 5])
    n_pieces = np.array([32, 32, 10, 10])
    assert eda.game_phase(ply, n_pieces).tolist() == [0, 1, 2, 2]


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
