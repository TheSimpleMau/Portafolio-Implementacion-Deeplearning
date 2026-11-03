import numpy as np
import pytest

from model import evaluation

# Tres posiciones con 3, 2 y 4 jugadas legales (índices de clase) y su valor de captura.
OFFSETS = np.array([0, 3, 5, 9])
LEGAL = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
CAPTURE = np.array([0, 1, 9, 0, 0, 3, 0, 0, 1], dtype=np.int8)


def test_subconjunto_de_jugadas_legales():
    offsets, legal, capture = evaluation.subset_legal(OFFSETS, LEGAL, CAPTURE, np.array([2, 0]))
    assert offsets.tolist() == [0, 4, 7]
    assert legal.tolist() == [60, 70, 80, 90, 10, 20, 30]
    assert capture.tolist() == [3, 0, 0, 1, 0, 1, 9]


def test_topk_con_la_jugada_real_en_segundo_lugar():
    scores = np.array([0.2, 0.5, 0.3, 0.9, 0.1, 0.1, 0.2, 0.3, 0.4])
    y = np.array([30, 40, 60])  # 2.º lugar, 1.er lugar y último lugar
    ranks = evaluation.played_rank(scores, OFFSETS, LEGAL, y)
    assert ranks.tolist() == [1, 0, 3]
    assert evaluation.topk_accuracy(ranks, ks=(1, 3)) == pytest.approx({"top1": 1 / 3, "top3": 2 / 3})


def test_azar_entre_legales():
    accuracy = evaluation.random_accuracy(np.array([0, 2, 6]), ks=(1, 3))
    assert accuracy == pytest.approx({"top1": (1 / 2 + 1 / 4) / 2, "top3": (1 + 3 / 4) / 2})


def test_captura_de_mayor_valor_va_primero():
    scores = evaluation.tie_break(evaluation.capture_scores(CAPTURE), np.random.default_rng(0))
    ranks = evaluation.played_rank(scores, OFFSETS, LEGAL, np.array([30, 40, 60]))
    assert ranks[0] == 0  # dama (9) antes que peón (1)
    assert ranks[2] == 0
    assert ranks[1] in (0, 1)  # empate sin capturas: se decide al azar


def test_libro_usa_la_jugada_modal_de_train_y_si_no_la_mas_frecuente():
    train_keys = np.array([0, 0, 0])
    train_move = np.array([20, 20, 30])
    modes = evaluation.book_moves(train_keys, train_move, n_keys=2)
    assert modes.tolist() == [20, -1]

    offsets = np.array([0, 3, 6])
    legal = np.array([10, 20, 30, 10, 20, 30])
    fallback = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])  # la más frecuente es 30
    scores = evaluation.book_scores(np.array([0, 1]), modes, offsets, legal, fallback)
    ranks = evaluation.played_rank(scores, offsets, legal, np.array([20, 30]))
    assert ranks.tolist() == [0, 0]  # posición en train: libro (20); posición nueva: la más frecuente (30)


class FixedModel:
    """Modelo falso que devuelve siempre las mismas probabilidades."""

    def __init__(self, probs):
        self.probs = probs

    def predict(self, x, verbose=0):
        return self.probs[:len(x)]


def test_evaluar_modelo_entre_legales_y_top1_ilegal():
    data = {
        "squares": np.zeros((3, 64), dtype=np.int8),
        "castling": np.zeros(3, dtype=np.uint8),
        "ep": np.full(3, -1, dtype=np.int8),
        "ply": np.array([0, 5, 30]),
        "move": np.array([30, 40, 60]),
        "legal_offsets": OFFSETS,
        "legal_moves": LEGAL,
        "legal_capture": CAPTURE,
    }
    probs = np.tile(np.arange(4096) * 1e-7, (3, 1))  # fondo sin empates
    probs[0, [10, 30]] = [0.6, 0.3]       # la real en 2.º lugar
    probs[1, [40, 50]] = [0.2, 0.1]       # acierto
    probs[1, 999] = 0.7                   # pero la más probable es ilegal
    probs[2, [60, 70]] = [0.5, 0.1]       # acierto

    results = evaluation.evaluate_model(FixedModel(probs), data, np.arange(3))["por_corte"]["todas"]
    assert results["top1"] == pytest.approx(2 / 3)
    assert results["top3"] == pytest.approx(1.0)
    assert results["top1_ilegal"] == pytest.approx(1 / 3)


def test_topk_por_grupo():
    ranks = np.array([0, 2, 0, 5, 1, 0])
    groups = np.array([0, 0, 1, 1, 1, 0])
    result = evaluation.topk_by_group(ranks, groups, ("A", "B", "C"))
    assert set(result) == {"A", "B"}
    assert result["A"] == {"posiciones": 3, "top1": 2 / 3, "top3": 1.0, "top5": 1.0}
    assert result["B"] == {"posiciones": 3, "top1": 1 / 3, "top3": 2 / 3, "top5": 2 / 3}
