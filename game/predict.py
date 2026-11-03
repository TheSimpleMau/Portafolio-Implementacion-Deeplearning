import chess

from game.players import ModelPlayer

FEN = chess.STARTING_FEN
MODEL_PATH = "models/final_dropout_05_dense_2048.keras"
TOP_K = 5


def main():
    board = chess.Board(FEN)
    moves, illegal_mass = ModelPlayer(MODEL_PATH).move_probabilities(board)
    print(board, end="\n\n")
    print(f"Mueven {'blancas' if board.turn == chess.WHITE else 'negras'}. Jugadas más probables:")
    for move, prob in moves[:TOP_K]:
        print(f"  {board.san(move):<8} {prob * 100:5.1f} %")
    print(f"Probabilidad que el modelo puso en jugadas ilegales: {illegal_mass * 100:.1f} %")


if __name__ == "__main__":
    main()
