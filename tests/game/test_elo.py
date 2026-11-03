import math

from game.elo import fit_elo


def test_elo_por_maxima_verosimilitud_en_cadena():
    # B le hace 76 % al ancla A (con los dos colores) y C le hace 50 % a B: los dos quedan ~200 arriba.
    games = [("B", "A", 1.0)] * 38 + [("A", "B", 0.0)] * 38 + [("B", "A", 0.0)] * 12 + [("A", "B", 1.0)] * 12
    games += [("C", "B", 0.5)] * 20 + [("B", "C", 1.0)] * 5 + [("C", "B", 1.0)] * 5
    fitted = fit_elo(games, anchor="A", anchor_elo=1500, prior_draws=0)

    expected = 1500 + 400 * math.log10(76 / 24)
    assert fitted["A"] == (1500.0, 0.0)
    assert abs(fitted["B"][0] - expected) < 0.01
    assert abs(fitted["C"][0] - expected) < 0.01
    assert fitted["C"][1] > fitted["B"][1] > 0  # C está más lejos del ancla: más incertidumbre
