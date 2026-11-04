import random

import chess

from model.preprocessing import decode_move, encode_move, legal_moves_info

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


def test_jugada_simetrica_para_blancas_y_negras():
    blancas = chess.Board()
    negras = chess.Board(BLACK_TO_MOVE_FEN)
    assert encode_move(chess.Move.from_uci("e2e4"), flipped=False) == 12 * 64 + 28
    assert encode_move(chess.Move.from_uci("e7e5"), flipped=True) == 12 * 64 + 28
    assert decode_move(12 * 64 + 28, blancas) == chess.Move.from_uci("e2e4")
    assert decode_move(12 * 64 + 28, negras) == chess.Move.from_uci("e7e5")


def test_ida_y_vuelta_de_todas_las_jugadas_legales():
    for board in all_test_boards():
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
