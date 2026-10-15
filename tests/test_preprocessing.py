import random

import chess
import numpy as np
import pytest

from preprocessing import (
    N_CLASSES,
    decode_move,
    encode_move,
    encode_position,
    expand_ordinal,
    expand_planes,
    legal_moves_info,
)

INITIAL_FEN = chess.STARTING_FEN
CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
CASTLING_BLACK_FEN = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
EN_PASSANT_FEN = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
PROMOTION_FEN = "8/P6k/8/8/8/8/8/K7 w - - 0 1"
CAPTURE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"
BLACK_TO_MOVE_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def random_positions(n_plies=80, seed=0):
    rng = random.Random(seed)
    board = chess.Board()
    positions = []
    for _ in range(n_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        positions.append(board.copy())
        board.push(rng.choice(moves))
    return positions


def all_test_boards():
    fens = [INITIAL_FEN, CASTLING_FEN, CASTLING_BLACK_FEN, EN_PASSANT_FEN,
            PROMOTION_FEN, CAPTURE_FEN, BLACK_TO_MOVE_FEN]
    return [chess.Board(fen) for fen in fens] + random_positions()


def test_posicion_inicial_tiene_20_jugadas_legales():
    indices, _ = legal_moves_info(chess.Board())
    assert len(indices) == 20
    assert N_CLASSES == 4096


def test_codificacion_posicion_inicial():
    squares, castling, ep = encode_position(chess.Board())
    tablero = squares.reshape(8, 8)
    assert tablero[0].tolist() == [4, 2, 3, 5, 6, 3, 2, 4]
    assert tablero[1].tolist() == [1] * 8
    assert tablero[6].tolist() == [7] * 8
    assert tablero[7].tolist() == [10, 8, 9, 11, 12, 9, 8, 10]
    assert (tablero[2:6] == 0).all()
    assert castling == 0b1111
    assert ep == -1


def test_negras_se_ven_desde_quien_mueve():
    squares, _, _ = encode_position(chess.Board(BLACK_TO_MOVE_FEN))
    tablero = squares.reshape(8, 8)
    assert tablero[0].tolist() == [4, 2, 3, 5, 6, 3, 2, 4]
    assert tablero[1].tolist() == [1] * 8
    assert tablero[7].tolist() == [10, 8, 9, 11, 12, 9, 8, 10]
    assert tablero[6].tolist() == [7, 7, 7, 7, 0, 7, 7, 7]
    assert tablero[4, 4] == 7


def test_negras_y_blancas_espejadas_tienen_la_misma_codificacion():
    for board in all_test_boards():
        espejo = board.mirror()
        for a, b in zip(encode_position(board), encode_position(espejo)):
            assert np.array_equal(a, b)


def test_jugada_simetrica_para_blancas_y_negras():
    blancas = chess.Board()
    negras = chess.Board(BLACK_TO_MOVE_FEN)
    assert encode_move(chess.Move.from_uci("e2e4"), flipped=False) == 12 * 64 + 28
    assert encode_move(chess.Move.from_uci("e7e5"), flipped=True) == 12 * 64 + 28
    assert decode_move(12 * 64 + 28, blancas) == chess.Move.from_uci("e2e4")
    assert decode_move(12 * 64 + 28, negras) == chess.Move.from_uci("e7e5")


@pytest.mark.parametrize("board", all_test_boards(), ids=lambda b: b.fen())
def test_ida_y_vuelta_de_todas_las_jugadas_legales(board):
    flipped = board.turn == chess.BLACK
    indices, _ = legal_moves_info(board)
    assert indices.tolist() == sorted(set(indices.tolist()))
    for move in board.legal_moves:
        index = encode_move(move, flipped)
        if index is None:
            assert move.promotion != chess.QUEEN
            continue
        assert index in indices
        decoded = decode_move(index, board)
        assert decoded in board.legal_moves
        assert (decoded.from_square, decoded.to_square) == (move.from_square, move.to_square)
    for index in indices:
        assert decode_move(int(index), board) in board.legal_moves


def test_enroque_y_captura_al_paso():
    board = chess.Board(CASTLING_FEN)
    assert encode_move(chess.Move.from_uci("e1g1"), flipped=False) == 4 * 64 + 6
    indices, _ = legal_moves_info(board)
    assert 4 * 64 + 6 in indices
    assert 4 * 64 + 2 in indices

    board = chess.Board(EN_PASSANT_FEN)
    _, castling, ep = encode_position(board)
    assert ep == 5
    assert castling == 0b1111
    assert decode_move(encode_move(chess.Move.from_uci("e5f6"), False), board).uci() == "e5f6"


def test_promocion_solo_a_dama():
    board = chess.Board(PROMOTION_FEN)
    indices, _ = legal_moves_info(board)
    a7a8 = 48 * 64 + 56
    assert indices.tolist().count(a7a8) == 1
    assert encode_move(chess.Move.from_uci("a7a8q"), False) == a7a8
    assert encode_move(chess.Move.from_uci("a7a8n"), False) is None
    assert encode_move(chess.Move.from_uci("a7a8r"), False) is None
    assert decode_move(a7a8, board).uci() == "a7a8q"


def test_valor_de_captura_por_jugada_legal():
    board = chess.Board(CAPTURE_FEN)
    indices, capturas = legal_moves_info(board)
    por_indice = dict(zip(indices.tolist(), capturas.tolist()))
    assert por_indice[encode_move(chess.Move.from_uci("d4e5"), False)] == 1
    assert por_indice[encode_move(chess.Move.from_uci("d4d5"), False)] == 0
    assert sum(1 for v in capturas if v > 0) == 1

    indices, capturas = legal_moves_info(chess.Board(EN_PASSANT_FEN))
    por_indice = dict(zip(indices.tolist(), capturas.tolist()))
    assert por_indice[encode_move(chess.Move.from_uci("e5f6"), False)] == 1


def test_planos_one_hot():
    boards = all_test_boards()
    codificadas = [encode_position(b) for b in boards]
    squares = np.stack([c[0] for c in codificadas])
    castling = np.array([c[1] for c in codificadas])
    ep = np.array([c[2] for c in codificadas])
    planos = expand_planes(squares, castling, ep)

    assert planos.shape == (len(boards), 8, 8, 17)
    assert planos.dtype == np.uint8
    piezas = planos[..., :12]
    assert piezas.sum(axis=-1).max() == 1
    for i, board in enumerate(boards):
        assert piezas[i].sum() == len(board.piece_map())

    inicial = planos[0]
    assert (inicial[..., 12:16] == 1).all()
    assert inicial[..., 16].sum() == 0
    assert inicial[0, 0, 3] == 1 and inicial[0, 0, :].sum() == 1 + 4

    en_passant = planos[3]
    assert en_passant[..., 16].sum() == 1
    assert en_passant[5, 5, 16] == 1


def test_codificacion_ordinal():
    squares, _, _ = encode_position(chess.Board())
    ordinal = expand_ordinal(squares[None])
    assert ordinal.shape == (1, 8, 8, 1)
    assert ordinal.dtype == np.int8
    assert ordinal[0, 0, :, 0].tolist() == [4, 2, 3, 5, 6, 3, 2, 4]
    assert ordinal[0, 1, :, 0].tolist() == [1] * 8
    assert ordinal[0, 6, :, 0].tolist() == [-1] * 8
    assert ordinal[0, 7, :, 0].tolist() == [-4, -2, -3, -5, -6, -3, -2, -4]
    assert (ordinal[0, 2:6] == 0).all()
