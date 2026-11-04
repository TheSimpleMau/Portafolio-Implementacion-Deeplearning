import sys

import chess

from game.players import ModelPlayer

MODEL_PATH = "models/final_dropout_05_dense_2048.keras"
TOP_K = 5


def main():
    fen = " ".join(sys.argv[1:]) or chess.STARTING_FEN
    try:
        board = chess.Board(fen)
    except ValueError as error:
        sys.exit(f"FEN inválido ({error}): {fen}")
    print("Posición recibida:")
    print(board, end="\n\n")
    if board.is_game_over():
        print("La partida ya terminó en esta posición.")
        return
    moves, illegal_mass = ModelPlayer(MODEL_PATH).move_probabilities(board)
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


if __name__ == "__main__":
    main()
