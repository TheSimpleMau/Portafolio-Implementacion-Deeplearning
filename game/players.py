import os
import random
import shutil
import subprocess

import chess
import chess.engine
import numpy as np

from model.preprocessing import decode_move, encode_position, expand_planes, legal_moves_info

PLAYER_KINDS = ("humano", "modelo", "stockfish", "azar")
STOCKFISH_TIME = 0.1
MAIA_DIR = "models/maia"


class RandomPlayer:
    name = "Azar"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_move(self, board):
        return self.rng.choice(list(board.legal_moves))


class CapturePlayer:
    """Captura la pieza de mayor valor; si no hay captura, juega al azar (la línea base como rival)."""

    name = "Captura"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_move(self, board):
        indices, captures = legal_moves_info(board)
        best = np.flatnonzero(captures == captures.max())
        return decode_move(indices[self.rng.choice(best.tolist())], board)


def open_stockfish():
    path = shutil.which("stockfish")
    if path is None:
        raise RuntimeError("No se encontró stockfish en el PATH (brew install stockfish).")
    return chess.engine.SimpleEngine.popen_uci(path)


class StockfishPlayer:
    def __init__(self, elo=None, time_limit=STOCKFISH_TIME, depth=None):
        self.engine = open_stockfish()
        if elo is not None:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self.limit = chess.engine.Limit(depth=depth) if depth else chess.engine.Limit(time=time_limit)
        if depth:
            self.name = f"Stockfish (profundidad {depth})"
        else:
            self.name = f"Stockfish (Elo {elo})" if elo is not None else "Stockfish"

    def choose_move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


class MaiaPlayer:
    """Maia (McIlroy-Young et al., 2020) con lc0 y sin búsqueda: la jugada más probable de su red."""

    def __init__(self, level):
        path = shutil.which("lc0")
        if path is None:
            raise RuntimeError("No se encontró lc0 en el PATH (brew install lc0).")
        # Con el backend metal, lc0 0.32 responde «a1a1» a go nodes 1; en CPU (blas) funciona.
        self.engine = chess.engine.SimpleEngine.popen_uci(
            [path, f"--weights={MAIA_DIR}/maia-{level}.pb.gz", "--backend=blas"], stderr=subprocess.DEVNULL)
        self.limit = chess.engine.Limit(nodes=1)
        self.name = f"Maia {level}"

    def choose_move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


class ModelPlayer:
    def __init__(self, model_path):
        import keras  # TensorFlow tarda en cargar: solo se importa si juega el modelo

        self.model = keras.models.load_model(model_path)
        self.name = f"Modelo ({os.path.basename(model_path)})"

    def move_probabilities(self, board):
        """Jugadas legales ordenadas con su probabilidad renormalizada y la masa puesta en ilegales."""
        squares, castling, ep = encode_position(board)
        planes = expand_planes(squares[None], np.array([castling]), np.array([ep]))
        probs = self.model(planes.astype(np.float32), training=False).numpy()[0]

        indices, _ = legal_moves_info(board)
        legal = probs[indices]
        illegal_mass = 1.0 - float(legal.sum())
        legal = legal / legal.sum()
        order = np.argsort(legal)[::-1]
        moves = [(decode_move(indices[i], board), float(legal[i])) for i in order]
        return moves, illegal_mass

    def choose_move(self, board):
        moves, _ = self.move_probabilities(board)
        return moves[0][0]


def make_player(kind, model_path=None, elo=None, time_limit=STOCKFISH_TIME):
    """None representa al humano: sus jugadas llegan desde la ventana."""
    if kind == "humano":
        return None
    if kind == "modelo":
        return ModelPlayer(model_path)
    if kind == "stockfish":
        return StockfishPlayer(elo, time_limit)
    if kind == "azar":
        return RandomPlayer()
    raise ValueError(f"Jugador desconocido: {kind}")
