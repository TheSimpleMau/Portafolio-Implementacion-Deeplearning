import glob
import io
import os
import shutil
import time

import chess
import chess.engine
import chess.svg

os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "1")  # suaviza el escalado en pantallas sin Retina
import pygame
from pygame._sdl2.video import Renderer, Texture

from game.match import analysis_comment, build_game, next_move, save_game
from game.players import PLAYER_KINDS, make_player, open_stockfish

# Se dibuja en un lienzo al doble de la ventana: en pantallas Retina cada píxel del lienzo es uno real.
UI_SCALE = 2
SQUARE = 160
BOARD_SIZE = 8 * SQUARE
BAR_WIDTH = 40
PANEL_X = BOARD_SIZE + BAR_WIDTH
PANEL_WIDTH = 760
CANVAS_SIZE = (PANEL_X + PANEL_WIDTH, BOARD_SIZE)
FPS = 30

MODELS_DIR = "models"
ELO_OPTIONS = (1320, 1600, 2000, 2400, 2800, None)  # None = Stockfish sin límite
EVAL_TIME = 0.1

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
LAST_MOVE = (205, 210, 106, 140)
SELECTED = (20, 85, 30, 110)
CHECK = (220, 40, 40, 150)
ARROW = (21, 101, 192)
PANEL_BG = (38, 36, 33)
TEXT = (230, 230, 230)
MUTED = (150, 150, 150)
BUTTON = (64, 61, 57)
BUTTON_ON = (98, 153, 36)
BAR_WHITE = (240, 240, 240)
BAR_BLACK = (60, 60, 60)

TERMINATIONS = {
    chess.Termination.CHECKMATE: "jaque mate",
    chess.Termination.STALEMATE: "ahogado",
    chess.Termination.INSUFFICIENT_MATERIAL: "material insuficiente",
    chess.Termination.SEVENTYFIVE_MOVES: "regla de 75 jugadas",
    chess.Termination.FIVEFOLD_REPETITION: "repetición",
    chess.Termination.FIFTY_MOVES: "regla de 50 jugadas",
    chess.Termination.THREEFOLD_REPETITION: "repetición",
}


def square_origin(square, flipped):
    """Esquina superior izquierda de la casilla, en píxeles del lienzo."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    if flipped:
        return (7 - file) * SQUARE, rank * SQUARE
    return file * SQUARE, (7 - rank) * SQUARE


def square_at(pos, flipped):
    """Casilla bajo el píxel del lienzo, o None si cae fuera del tablero."""
    x, y = pos
    if x >= BOARD_SIZE or y >= BOARD_SIZE:
        return None
    file, row = x // SQUARE, y // SQUARE
    if flipped:
        return chess.square(7 - file, row)
    return chess.square(file, 7 - row)


def square_center(square, flipped):
    x, y = square_origin(square, flipped)
    return pygame.Vector2(x + SQUARE / 2, y + SQUARE / 2)


def load_pieces():
    pieces = {}
    for symbol in "PNBRQKpnbrqk":
        svg = chess.svg.piece(chess.Piece.from_symbol(symbol)).encode("utf-8")
        pieces[symbol] = pygame.image.load_sized_svg(io.BytesIO(svg), (SQUARE, SQUARE))
    return pieces


def human_move(board, from_square, to_square):
    """Jugada legal entre dos casillas (promoción solo a dama, igual que el modelo) o None."""
    move = chess.Move(from_square, to_square)
    if board.piece_type_at(from_square) == chess.PAWN and chess.square_rank(to_square) in (0, 7):
        move.promotion = chess.QUEEN
    return move if move in board.legal_moves else None


def evaluate(engine, board):
    """Ventaja según Stockfish: (texto, probabilidad de ganar de las blancas con la fórmula de Lichess)."""
    outcome = board.outcome(claim_draw=True)
    if outcome:
        return f"Fin: {outcome.result()}", {True: 1.0, False: 0.0, None: 0.5}[outcome.winner]
    score = engine.analyse(board, chess.engine.Limit(time=EVAL_TIME))["score"].white()
    if score.is_mate():
        side = "blancas" if score.mate() > 0 else "negras"
        text = f"#{abs(score.mate())} ({side})"
    else:
        pawns = score.score() / 100
        side = "blancas" if pawns >= 0.05 else "negras" if pawns <= -0.05 else "igualada"
        text = f"{pawns:+.1f} ({side})"
    return text, score.wdl(model="lichess").expectation()


def draw_arrow(overlay, start, end, prob):
    direction = (end - start).normalize()
    tip = end - direction * SQUARE * 0.15
    head = SQUARE * 0.3
    width = max(8, round(32 * prob ** 0.5))
    color = (*ARROW, round(90 + 140 * prob))
    pygame.draw.line(overlay, color, start, tip - direction * head * 0.8, width)
    normal = pygame.Vector2(-direction.y, direction.x)
    pygame.draw.polygon(overlay, color, [tip, tip - direction * head + normal * head * 0.5,
                                         tip - direction * head - normal * head * 0.5])


class Fonts:
    def __init__(self):
        self.title = pygame.font.SysFont("helvetica", 64)
        self.normal = pygame.font.SysFont("helvetica", 40)
        self.small = pygame.font.SysFont("helvetica", 32)


class Menu:
    """Pantalla inicial: quién juega con cada color, fuerza de Stockfish y archivo del modelo."""

    def __init__(self, canvas, fonts):
        self.canvas = canvas
        self.fonts = fonts
        self.models = sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))
        self.has_stockfish = shutil.which("stockfish") is not None
        self.kinds = {chess.WHITE: "humano", chess.BLACK: "stockfish" if self.has_stockfish else "azar"}
        self.elos = {chess.WHITE: ELO_OPTIONS[0], chess.BLACK: ELO_OPTIONS[0]}
        self.model_index = 0
        self.buttons = []

    def available(self, kind):
        return {"modelo": bool(self.models), "stockfish": self.has_stockfish}.get(kind, True)

    def config(self):
        model = self.models[self.model_index] if self.models else None
        return self.kinds, self.elos, model

    def button(self, rect, label, selected=False, enabled=True, action=None):
        rect = pygame.Rect(rect)
        pygame.draw.rect(self.canvas, BUTTON_ON if selected else BUTTON, rect, border_radius=12)
        text = self.fonts.normal.render(label, True, TEXT if enabled else MUTED)
        self.canvas.blit(text, text.get_rect(center=rect.center))
        if enabled and action:
            self.buttons.append((rect, action))

    def draw(self):
        self.buttons = []
        self.canvas.fill(PANEL_BG)
        width = CANVAS_SIZE[0]
        title = self.fonts.title.render("Ajedrez contra el modelo", True, TEXT)
        self.canvas.blit(title, title.get_rect(center=(width // 2, 120)))

        def label(text, y):
            self.canvas.blit(self.fonts.normal.render(text, True, TEXT), (160, y))

        def choose(color, kind):
            self.kinds[color] = kind

        for y, color, name in ((260, chess.WHITE, "Blancas"), (400, chess.BLACK, "Negras")):
            label(name, y + 20)
            for i, kind in enumerate(PLAYER_KINDS):
                self.button((560 + i * 330, y, 300, 90), kind.capitalize(), self.kinds[color] == kind,
                            self.available(kind), lambda c=color, k=kind: choose(c, k))

        def choose_elo(color, elo):
            self.elos[color] = elo

        # La fuerza de cada color solo se elige si ese color lo juega Stockfish.
        for y, color, name in ((560, chess.WHITE, "Stockfish blancas"), (680, chess.BLACK, "Stockfish negras")):
            label(name, y + 20)
            enabled = self.kinds[color] == "stockfish"
            for i, elo in enumerate(ELO_OPTIONS):
                self.button((560 + i * 220, y, 200, 90), str(elo) if elo else "Completo",
                            enabled and self.elos[color] == elo, enabled,
                            lambda c=color, e=elo: choose_elo(c, e))

        label("Modelo", 840)
        if self.models:
            name = os.path.basename(self.models[self.model_index])
            step = lambda d: setattr(self, "model_index", (self.model_index + d) % len(self.models))
            self.button((560, 820, 90, 90), "<", action=lambda: step(-1))
            self.button((670, 820, 700, 90), name)
            self.button((1390, 820, 90, 90), ">", action=lambda: step(1))
        else:
            text = f"No hay archivos .keras en {MODELS_DIR}/"
            self.canvas.blit(self.fonts.normal.render(text, True, MUTED), (560, 840))
        if not self.has_stockfish:
            self.canvas.blit(self.fonts.small.render("Stockfish no está instalado (brew install stockfish).",
                                                     True, MUTED), (560, 940))

        self.play_rect = pygame.Rect(width // 2 - 250, 1040, 500, 120)
        self.button(self.play_rect, "Jugar", selected=True, action=lambda: None)

    def on_click(self, pos):
        """Aplica el botón pulsado; devuelve True si fue «Jugar»."""
        for rect, action in self.buttons:
            if rect.collidepoint(pos):
                action()
                return rect == self.play_rect
        return False


class ChessGame:
    def __init__(self, canvas, fonts, pieces, players, evaluator, delay=0.5):
        self.canvas = canvas
        self.fonts = fonts
        self.pieces = pieces
        self.players = players
        self.evaluator = evaluator
        self.delay = delay
        self.flipped = players[chess.WHITE] is not None and players[chess.BLACK] is None
        self.new_game()

    def name(self, color):
        player = self.players[color]
        return "Humano" if player is None else player.name

    def new_game(self):
        self.board = chess.Board()
        self.comments = []
        self.analysis = None  # (ply, color, [(san, prob, move)], ilegales) de la última jugada del modelo
        self.selected = None
        self.paused = False
        self.saved_path = None
        self.last_cpu_move = time.monotonic()
        self.refresh_evaluation()

    def refresh_evaluation(self):
        self.evaluation = evaluate(self.evaluator, self.board) if self.evaluator else None

    def push(self, move, analysis=None):
        if analysis:
            moves, illegal_mass = analysis
            top = [(self.board.san(m), prob, m) for m, prob in moves[:3]]
            self.analysis = (self.board.ply(), self.board.turn, top, illegal_mass)
        self.comments.append(analysis_comment(self.board, analysis) if analysis else "")
        self.board.push(move)
        self.selected = None
        self.refresh_evaluation()
        if self.board.is_game_over(claim_draw=True):
            game = build_game(self.board.move_stack, self.comments, self.name(chess.WHITE),
                              self.name(chess.BLACK), self.board.result(claim_draw=True))
            self.saved_path = save_game(game)

    def undo(self):
        """Deshace una jugada; contra la CPU, también la suya para volver al turno del humano."""
        for _ in range(2):
            if not self.board.move_stack:
                break
            self.board.pop()
            self.comments.pop()
            if self.players[self.board.turn] is None or self.players[not self.board.turn] is not None:
                break
        self.analysis = None
        self.selected = None
        self.saved_path = None
        self.refresh_evaluation()

    def on_click(self, pos):
        if self.players[self.board.turn] is not None or self.board.is_game_over(claim_draw=True):
            return
        square = square_at(pos, self.flipped)
        if square is None:
            return
        if self.selected is not None:
            move = human_move(self.board, self.selected, square)
            if move:
                self.push(move)
                return
        piece = self.board.piece_at(square)
        self.selected = square if piece and piece.color == self.board.turn else None

    def on_key(self, key):
        if key == pygame.K_n:
            self.new_game()
        elif key == pygame.K_u:
            self.undo()
        elif key == pygame.K_SPACE:
            self.paused = not self.paused

    def cpu_turn(self):
        player = self.players[self.board.turn]
        if (player is None or self.paused or self.board.is_game_over(claim_draw=True)
                or time.monotonic() - self.last_cpu_move < self.delay):
            return
        move, analysis = next_move(player, self.board)
        self.push(move, analysis)
        self.last_cpu_move = time.monotonic()

    def close(self):
        for player in self.players.values():
            if hasattr(player, "close"):
                player.close()

    def fill_square(self, square, color):
        overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
        overlay.fill(color)
        self.canvas.blit(overlay, square_origin(square, self.flipped))

    def draw_board(self):
        for square in chess.SQUARES:
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            x, y = square_origin(square, self.flipped)
            pygame.draw.rect(self.canvas, LIGHT if light else DARK, (x, y, SQUARE, SQUARE))
        if self.board.move_stack:
            last = self.board.peek()
            self.fill_square(last.from_square, LAST_MOVE)
            self.fill_square(last.to_square, LAST_MOVE)
        if self.board.is_check():
            self.fill_square(self.board.king(self.board.turn), CHECK)
        if self.selected is not None:
            self.fill_square(self.selected, SELECTED)

        for i in range(8):
            file_name = chess.FILE_NAMES[7 - i if self.flipped else i]
            rank_name = chess.RANK_NAMES[i if self.flipped else 7 - i]
            self.canvas.blit(self.fonts.small.render(file_name, True, LIGHT if i % 2 == 0 else DARK),
                             (i * SQUARE + SQUARE - 24, BOARD_SIZE - 36))
            self.canvas.blit(self.fonts.small.render(rank_name, True, DARK if i % 2 == 0 else LIGHT),
                             (6, i * SQUARE + 4))

        for square, piece in self.board.piece_map().items():
            self.canvas.blit(self.pieces[piece.symbol()], square_origin(square, self.flipped))

        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        if self.selected is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected:
                    pygame.draw.circle(overlay, SELECTED, square_center(move.to_square, self.flipped), 24)
        # Las flechas solo se ven mientras la última jugada es la del modelo.
        if self.analysis and self.analysis[0] == self.board.ply() - 1:
            for _, prob, move in reversed(self.analysis[2]):
                draw_arrow(overlay, square_center(move.from_square, self.flipped),
                           square_center(move.to_square, self.flipped), prob)
        self.canvas.blit(overlay, (0, 0))

    def draw_bar(self):
        """Barra de ventaja: la parte blanca crece con la probabilidad de ganar de las blancas."""
        pygame.draw.rect(self.canvas, BAR_BLACK, (BOARD_SIZE, 0, BAR_WIDTH, BOARD_SIZE))
        if not self.evaluation:
            return
        white = round(BOARD_SIZE * self.evaluation[1])
        y = 0 if self.flipped else BOARD_SIZE - white
        pygame.draw.rect(self.canvas, BAR_WHITE, (BOARD_SIZE, y, BAR_WIDTH, white))
        pygame.draw.line(self.canvas, MUTED, (BOARD_SIZE, BOARD_SIZE // 2),
                         (PANEL_X - 1, BOARD_SIZE // 2), 2)

    def draw_panel(self):
        pygame.draw.rect(self.canvas, PANEL_BG, (PANEL_X, 0, PANEL_WIDTH, BOARD_SIZE))
        x, y = PANEL_X + 40, 40

        def line(text, font=self.fonts.normal, color=TEXT, gap=56):
            nonlocal y
            self.canvas.blit(font.render(text, True, color), (x, y))
            y += gap

        line(f"Blancas: {self.name(chess.WHITE)}")
        line(f"Negras: {self.name(chess.BLACK)}", gap=88)

        outcome = self.board.outcome(claim_draw=True)
        if outcome:
            line(f"Resultado: {outcome.result()}")
            line(f"({TERMINATIONS.get(outcome.termination, outcome.termination.name.lower())})", gap=88)
        else:
            turn = "blancas" if self.board.turn == chess.WHITE else "negras"
            line(f"Jugada {self.board.fullmove_number}, mueven {turn}")
            line("En pausa" if self.paused else "", color=MUTED, gap=88)

        if self.evaluation and not outcome:
            text, white_win = self.evaluation
            line(f"Ventaja: {text}")
            line(f"Prob. de ganar de las blancas: {white_win * 100:.0f} %", self.fonts.small, MUTED, 88)

        if self.analysis:
            ply, color, top, illegal_mass = self.analysis
            side = "blancas" if color == chess.WHITE else "negras"
            line(f"Modelo ({side}), jugada {ply // 2 + 1}:")
            for san, prob, _ in top:
                line(f"   {san:<8} {prob * 100:5.1f} %")
            line(f"   Prob. en ilegales: {illegal_mass * 100:.1f} %", color=MUTED)

        y = BOARD_SIZE - 180
        if self.saved_path:
            line(f"PGN: {self.saved_path}", self.fonts.small, MUTED, 48)
        line("n nueva   u deshacer   Esc menú", self.fonts.small, MUTED, 44)
        line("espacio pausa (CPU vs CPU)", self.fonts.small, MUTED, 44)

    def draw(self):
        self.draw_board()
        self.draw_bar()
        self.draw_panel()


class App:
    def __init__(self):
        size = (CANVAS_SIZE[0] // UI_SCALE, CANVAS_SIZE[1] // UI_SCALE)
        self.window = pygame.Window("Ajedrez contra el modelo", size, allow_high_dpi=True)
        self.renderer = Renderer(self.window)
        self.texture = Texture(self.renderer, CANVAS_SIZE, streaming=True)
        self.canvas = pygame.Surface(CANVAS_SIZE)
        self.fonts = Fonts()
        self.pieces = load_pieces()
        self.menu = Menu(self.canvas, self.fonts)
        self.evaluator = open_stockfish() if self.menu.has_stockfish else None
        self.game = None

    def start_game(self):
        kinds, elos, model_path = self.menu.config()
        players = {color: make_player(kinds[color], model_path, elos[color]) for color in chess.COLORS}
        self.game = ChessGame(self.canvas, self.fonts, self.pieces, players, self.evaluator)

    def back_to_menu(self):
        self.game.close()
        self.game = None

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = (event.pos[0] * UI_SCALE, event.pos[1] * UI_SCALE)
            if self.game:
                self.game.on_click(pos)
            elif self.menu.on_click(pos):
                self.start_game()
        elif event.type == pygame.KEYDOWN and self.game:
            if event.key == pygame.K_ESCAPE:
                self.back_to_menu()
            else:
                self.game.on_key(event.key)

    def present(self):
        self.texture.update(self.canvas)
        self.renderer.clear()
        self.texture.draw()
        self.renderer.present()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
                    return
                self.handle(event)
            if self.game:
                self.game.draw()
            else:
                self.menu.draw()
            self.present()
            if self.game:
                self.game.cpu_turn()
            clock.tick(FPS)

    def close(self):
        if self.game:
            self.game.close()
        if self.evaluator:
            self.evaluator.quit()


def main():
    pygame.init()
    app = App()
    try:
        app.run()
    finally:
        app.close()
        pygame.quit()


if __name__ == "__main__":
    main()
