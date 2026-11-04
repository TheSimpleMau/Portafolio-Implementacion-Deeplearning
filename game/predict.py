import sys

import chess

from game.players import ModelPlayer

FEN_FILE = "interesantes.fen"  # se usa solo si no se pasa una FEN por consola
MODEL_PATH = "models/final_dropout_05_dense_2048.keras"
TOP_K = 5


def read_fens(path):
    """Lista de (título, FEN). Una posición por línea; las líneas con # son el título de la siguiente."""
    cases, title = [], ""
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                title = line.lstrip("# ")
            else:
                cases.append((title, line))
                title = ""
    return cases


def predict(player, title, fen):
    try:
        board = chess.Board(fen)
    except ValueError as error:
        print(f"\nFEN inválido ({error}): {fen}")
        return
    print(f"\n{'=' * 60}\n{title or fen}\n{'=' * 60}")
    print("Posición recibida:")
    print(board, end="\n\n")
    if board.is_game_over():
        print("La partida ya terminó en esta posición.")
        return
    moves, illegal_mass = player.move_probabilities(board)
    print(f"Mueven {'blancas' if board.turn == chess.WHITE else 'negras'}. Jugadas más probables:")
    for move, prob in moves[:TOP_K]:
        print(f"  {board.san(move):<8} {prob * 100:5.1f} %")
    print(f"Probabilidad que el modelo puso en jugadas ilegales: {illegal_mass * 100:.1f} %")
    best = moves[0][0]
    san = board.san(best)
    board.push(best)
    result = " (jaque mate)" if board.is_checkmate() else " (ahogado)" if board.is_stalemate() else ""
    print(f"\nPosición tras la jugada predicha, {san}{result}:")
    print(board)


def main():
    player = ModelPlayer(MODEL_PATH)
    cases = [("", " ".join(sys.argv[1:]))] if len(sys.argv) > 1 else read_fens(FEN_FILE)
    for title, fen in cases:
        predict(player, title, fen)


if __name__ == "__main__":
    main()
