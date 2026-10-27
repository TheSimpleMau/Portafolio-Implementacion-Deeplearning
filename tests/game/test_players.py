import chess
import keras
import numpy as np
import pytest

from game.players import ModelPlayer
from model.preprocessing import N_CLASSES, N_PLANES

E2E4 = 12 * 64 + 28
E2E5 = 12 * 64 + 36  # ilegal desde la posición inicial
BLACK_TO_MOVE_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def model_with_output(tmp_path, logits):
    """Modelo cuya salida no depende del tablero: softmax(logits)."""
    inputs = keras.Input((8, 8, N_PLANES))
    hidden = keras.layers.Dense(1)(keras.layers.Flatten()(inputs))
    outputs = keras.layers.Dense(N_CLASSES, activation="softmax", kernel_initializer="zeros",
                                 bias_initializer=keras.initializers.Constant(logits))(hidden)
    path = tmp_path / "modelo.keras"
    keras.Model(inputs, outputs).save(path)
    return ModelPlayer(str(path))


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, BLACK_TO_MOVE_FEN])
def test_modelo_solo_juega_legales_y_sus_probabilidades_suman_uno(tmp_path, fen):
    player = model_with_output(tmp_path, np.random.default_rng(0).normal(size=N_CLASSES))
    board = chess.Board(fen)
    moves, illegal_mass = player.move_probabilities(board)

    assert {move for move, _ in moves} == set(board.legal_moves)
    assert sum(prob for _, prob in moves) == pytest.approx(1.0)
    assert [prob for _, prob in moves] == sorted((prob for _, prob in moves), reverse=True)
    assert 0 < illegal_mass < 1
    assert player.choose_move(board) == moves[0][0]


def test_si_lo_mas_probable_es_ilegal_elige_la_mejor_legal(tmp_path):
    logits = np.zeros(N_CLASSES)
    logits[E2E5] = 20
    logits[E2E4] = 10
    player = model_with_output(tmp_path, logits)

    _, illegal_mass = player.move_probabilities(chess.Board())
    assert illegal_mass > 0.99
    assert player.choose_move(chess.Board()) == chess.Move.from_uci("e2e4")
    assert player.choose_move(chess.Board(BLACK_TO_MOVE_FEN)) == chess.Move.from_uci("e7e5")
