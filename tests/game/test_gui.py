import chess
import pytest

from game.gui import human_move

FENS = [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # enroque
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # captura al paso
    "8/P6k/8/8/8/8/8/K7 w - - 0 1",  # promoción de blancas
    "k7/8/8/8/8/8/p7/7K b - - 0 1",  # promoción de negras
    "4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1",  # caballo clavado
    "4k3/8/8/8/7b/8/8/4K2R w K - 0 1",  # en jaque: no puede enrocar
]


@pytest.mark.parametrize("fen", FENS)
def test_clic_del_humano_solo_da_jugadas_legales(fen):
    board = chess.Board(fen)
    legal = {(move.from_square, move.to_square): move for move in board.legal_moves
             if move.promotion in (None, chess.QUEEN)}
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            move = human_move(board, from_square, to_square)
            if (from_square, to_square) in legal:
                assert move == legal[from_square, to_square]
            else:  # python-chess acepta el rey sobre su propia torre como enroque
                assert move is None or board.is_castling(move)
