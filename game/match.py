import datetime
import os

import chess
import chess.pgn

from game.players import ModelPlayer

GAMES_DIR = "resultados/partidas"
MAX_PLIES = 400
TOP_K = 3


def next_move(player, board):
    """Jugada del jugador y, si es el modelo, su análisis (jugadas con probabilidad, masa ilegal)."""
    if isinstance(player, ModelPlayer):
        moves, illegal_mass = player.move_probabilities(board)
        return moves[0][0], (moves, illegal_mass)
    return player.choose_move(board), None


def analysis_comment(board, analysis, k=TOP_K):
    moves, illegal_mass = analysis
    top = ", ".join(f"{board.san(move)} {prob * 100:.0f} %" for move, prob in moves[:k])
    return f"modelo: {top}; ilegales {illegal_mass * 100:.0f} %"


def build_game(moves, comments, white_name, black_name, result):
    game = chess.pgn.Game()
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    node = game
    for move, comment in zip(moves, comments):
        node = node.add_variation(move, comment=comment)
    return game


def save_game(game, games_dir=GAMES_DIR):
    os.makedirs(games_dir, exist_ok=True)
    path = os.path.join(games_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".pgn")
    with open(path, "w", encoding="utf-8") as file:
        print(game, file=file)
    return path


def play_game(white, black, max_plies=MAX_PLIES):
    """Partida CPU vs CPU sin ventana; devuelve el PGN con el top-3 del modelo como comentario."""
    board = chess.Board()
    comments = []
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        player = white if board.turn == chess.WHITE else black
        move, analysis = next_move(player, board)
        comments.append(analysis_comment(board, analysis) if analysis else "")
        board.push(move)
    return build_game(board.move_stack, comments, white.name, black.name, board.result(claim_draw=True))
