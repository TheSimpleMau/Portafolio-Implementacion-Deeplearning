import chess
import numpy as np

N_CLASSES = 64 * 64
N_PIECE_PLANES = 12
N_PLANES = 17

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def orient(board: chess.Board):
    if board.turn == chess.BLACK:
        return board.mirror(), True
    return board, False


def encode_position(board: chess.Board):
    board, _ = orient(board)

    squares = np.zeros(64, dtype=np.int8)
    for square, piece in board.piece_map().items():
        squares[square] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)

    castling = (
        int(board.has_kingside_castling_rights(chess.WHITE))
        | int(board.has_queenside_castling_rights(chess.WHITE)) << 1
        | int(board.has_kingside_castling_rights(chess.BLACK)) << 2
        | int(board.has_queenside_castling_rights(chess.BLACK)) << 3
    )
    ep = chess.square_file(board.ep_square) if board.has_legal_en_passant() else -1
    return squares, castling, ep


def encode_move(move: chess.Move, flipped: bool):
    if move.promotion not in (None, chess.QUEEN):
        return None
    from_square, to_square = move.from_square, move.to_square
    if flipped:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)
    return from_square * 64 + to_square


def decode_move(index: int, board: chess.Board):
    from_square, to_square = divmod(int(index), 64)
    if board.turn == chess.BLACK:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)

    promotion = None
    if (
        board.piece_type_at(from_square) == chess.PAWN
        and chess.square_rank(to_square) in (0, 7)
    ):
        promotion = chess.QUEEN
    return chess.Move(from_square, to_square, promotion)


def _capture_value(board: chess.Board, move: chess.Move):
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    captured = board.piece_type_at(move.to_square)
    return PIECE_VALUES[captured] if captured else 0


def legal_moves_info(board: chess.Board):
    flipped = board.turn == chess.BLACK
    captures = {}
    for move in board.legal_moves:
        index = encode_move(move, flipped)
        if index is None:
            continue
        captures[index] = _capture_value(board, move)

    indices = np.array(sorted(captures), dtype=np.int16)
    values = np.array([captures[i] for i in indices.tolist()], dtype=np.int8)
    return indices, values


def expand_planes(squares: np.ndarray, castling: np.ndarray, ep: np.ndarray):
    n = len(squares)
    board = squares.reshape(n, 8, 8)
    planes = np.zeros((n, 8, 8, N_PLANES), dtype=np.uint8)

    for code in range(1, N_PIECE_PLANES + 1):
        planes[..., code - 1] = board == code

    castling = np.asarray(castling)
    for bit in range(4):
        planes[..., N_PIECE_PLANES + bit] = ((castling >> bit) & 1)[:, None, None]

    ep = np.asarray(ep)
    has_ep = np.flatnonzero(ep >= 0)
    planes[has_ep, 5, ep[has_ep], N_PLANES - 1] = 1
    return planes


def expand_ordinal(squares: np.ndarray):
    n = len(squares)
    codes = squares.astype(np.int8)
    ordinal = np.where(codes > 6, -(codes - 6), codes).astype(np.int8)
    return ordinal.reshape(n, 8, 8, 1)
