"""Estimación del Elo de un modelo con una escalera de rivales anclada en Stockfish con UCI_Elo 1320.

Uso, desde la raíz del proyecto con el entorno activado:
    python -m game.elo models/m1.keras

Primero calibra a los rivales entre sí (cada uno contra los dos siguientes de RIVALS) y sondea al
modelo con SURVEY_OPENINGS aperturas contra cada rival; después completa las partidas contra los
rivales a los que el modelo les hace entre SCORE_RANGE. Todas las partidas empiezan desde las
aperturas de graficas/aperturas_elo.json (python -m model.eda aperturas), una vez con cada color.
"""
import json
import os
import sys
import time

import numpy as np

from game.match import play_game
from game.players import CapturePlayer, MaiaPlayer, ModelPlayer, RandomPlayer, StockfishPlayer
from model.eda import ELO_OPENINGS_PATH

ANCHOR = "Stockfish 1320"
ANCHOR_ELO = 1320
# De más débil a más fuerte (orden esperado; solo decide contra quién se calibra cada uno).
RIVALS = {
    "Azar": lambda: RandomPlayer(seed=42),
    "Captura": lambda: CapturePlayer(seed=42),
    "Stockfish profundidad 1": lambda: StockfishPlayer(depth=1),
    "Maia 1100": lambda: MaiaPlayer(1100),
    "Maia 1500": lambda: MaiaPlayer(1500),
    "Stockfish 1320": lambda: StockfishPlayer(elo=1320),
    "Maia 1900": lambda: MaiaPlayer(1900),
}
NEIGHBORS = 2
SURVEY_OPENINGS = 5
SCORE_RANGE = (0.1, 0.9)
# Una partida empatada imaginaria por pareja: evita Elo infinito si alguien gana todas.
PRIOR_DRAWS = 1
SCORES = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "*": 0.5}  # «*»: se llegó al límite de plies
ELO_SCALE = 400
GAMES_DIR = "resultados/partidas_elo"


def expected_score(elo, rival_elo):
    return 1 / (1 + 10 ** ((rival_elo - elo) / ELO_SCALE))


def run_match(a, b, openings, games_dir=GAMES_DIR):
    """Cada apertura dos veces, una con cada color; guarda los PGN y devuelve (blancas, negras, puntos)."""
    os.makedirs(games_dir, exist_ok=True)
    results = []
    with open(os.path.join(games_dir, f"{a.name} vs {b.name}.pgn"), "a", encoding="utf-8") as file:
        for opening in openings:
            for white, black in ((a, b), (b, a)):
                game = play_game(white, black, opening=opening["jugadas"])
                game.headers["Opening"] = opening["familia"]
                print(game, file=file, end="\n\n")
                results.append((white.name, black.name, SCORES[game.headers["Result"]]))
    return results


def fit_elo(games, anchor=ANCHOR, anchor_elo=ANCHOR_ELO, prior_draws=PRIOR_DRAWS, iterations=100):
    """Elo de cada jugador por máxima verosimilitud (modelo logístico, tablas = ½) con el ancla fija.

    Devuelve {jugador: (elo, error estándar)}; el error sale de la información observada.
    """
    pairs = {tuple(sorted((white, black))) for white, black, _ in games}
    games = list(games) + [(a, b, 0.5) for a, b in pairs for _ in range(prior_draws)]
    players = sorted({name for white, black, _ in games for name in (white, black)} - {anchor})
    index = {name: i for i, name in enumerate(players)}
    white = np.array([index.get(w, -1) for w, _, _ in games])
    black = np.array([index.get(b, -1) for _, b, _ in games])
    score = np.array([s for _, _, s in games])
    k = np.log(10) / ELO_SCALE

    elo = np.full(len(players), float(anchor_elo))
    for _ in range(iterations):
        full = np.append(elo, anchor_elo)  # el índice -1 es el ancla
        p = expected_score(full[white], full[black])
        gradient, information = np.zeros(len(players)), np.zeros((len(players), len(players)))
        weight = k ** 2 * p * (1 - p)
        for sign, side in ((1, white), (-1, black)):
            mask = side >= 0
            np.add.at(gradient, side[mask], sign * k * (score - p)[mask])
            for other_sign, other in ((1, white), (-1, black)):
                both = mask & (other >= 0)
                np.add.at(information, (side[both], other[both]), sign * other_sign * weight[both])
        step = np.linalg.solve(information, gradient)
        elo += step
        if np.abs(step).max() < 1e-6:
            break
    errors = np.sqrt(np.diag(np.linalg.inv(information)))
    fitted = {name: (float(elo[i]), float(errors[i])) for name, i in index.items()}
    fitted[anchor] = (float(anchor_elo), 0.0)
    return fitted


def match_scores(games):
    """Puntos del primero de cada pareja (en orden alfabético) y número de partidas."""
    table = {}
    for white, black, score in games:
        pair = tuple(sorted((white, black)))
        points = score if white == pair[0] else 1 - score
        total, n = table.get(pair, (0.0, 0))
        table[pair] = (total + points, n + 1)
    return table


def score_against(name, rival, games):
    """Fracción de puntos de name contra rival y número de partidas entre los dos."""
    played = [(w, s) for w, b, s in games if {w, b} == {name, rival}]
    return sum(s if w == name else 1 - s for w, s in played) / len(played), len(played)


def close(player):
    getattr(player, "close", lambda: None)()


def run_ladder(model_path, openings_path=ELO_OPENINGS_PATH, games_dir=GAMES_DIR):
    with open(openings_path, encoding="utf-8") as file:
        openings = json.load(file)
    model_name = os.path.splitext(os.path.basename(model_path))[0].upper()
    results_path = f"resultados/elo_{model_name.lower()}.json"
    names = list(RIVALS)
    games, started = [], time.time()

    def play(a, b, selected):
        games.extend(run_match(a, b, selected, games_dir))
        share, n = score_against(a.name, b.name, games)
        print(f"{a.name} vs {b.name}: {n} partidas, {a.name} hace {share * 100:.0f} % "
              f"({(time.time() - started) / 60:.0f} min)", flush=True)
        return share

    model = ModelPlayer(model_path)
    model.name = model_name
    for i, name in enumerate(names):
        rival = RIVALS[name]()
        rival.name = name
        for other_name in names[i + 1:i + 1 + NEIGHBORS]:
            other = RIVALS[other_name]()
            other.name = other_name
            play(rival, other, openings)
            close(other)
        if SCORE_RANGE[0] <= play(model, rival, openings[:SURVEY_OPENINGS]) <= SCORE_RANGE[1]:
            play(model, rival, openings[SURVEY_OPENINGS:])
        close(rival)

    fitted = fit_elo(games)
    table = match_scores(games)
    results = {
        "ancla": {ANCHOR: ANCHOR_ELO},
        "partidas": len(games),
        "elo": {name: {"elo": round(e), "error": round(err), "ic95": [round(e - 1.96 * err), round(e + 1.96 * err)]}
                for name, (e, err) in sorted(fitted.items(), key=lambda item: item[1][0])},
        "parejas": [{"jugadores": list(pair), "puntos_del_primero": points, "partidas": n,
                     "esperado_del_primero": round(n * expected_score(fitted[pair[0]][0], fitted[pair[1]][0]), 1)}
                    for pair, (points, n) in sorted(table.items())],
        "minutos": round((time.time() - started) / 60, 1),
    }
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    for name, values in results["elo"].items():
        print(f"{name:<26} {values['elo']:>6}  IC 95 % {values['ic95']}")
    return results


if __name__ == "__main__":
    run_ladder(sys.argv[1])
