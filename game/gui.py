import glob
import io
import os
import shutil
import time
from collections import namedtuple

import chess
import chess.engine
import chess.svg

os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "1")  # suaviza el escalado en pantallas sin Retina
import pygame
import pygame.gfxdraw
from pygame._sdl2.video import Renderer, Texture

from game.match import analysis_comment, build_game, next_move, save_game
from game.players import PLAYER_KINDS, ModelPlayer, make_player, open_stockfish

# Se dibuja en un lienzo al doble de la ventana: en pantallas Retina cada píxel del lienzo es uno real.
UI_SCALE = 2
SQUARE = 160
BOARD_SIZE = 8 * SQUARE
MARGIN = 32
BOARD_Y = MARGIN
BAR_X = MARGIN
BAR_WIDTH = 72
BOARD_X_WITH_BAR = BAR_X + BAR_WIDTH + MARGIN
PANEL_X = BOARD_X_WITH_BAR + BOARD_SIZE + MARGIN
BOARD_X_NO_BAR = (PANEL_X - BOARD_SIZE) // 2  # sin Stockfish no hay barra y el tablero se centra
PANEL_WIDTH = 640
CARD_WIDTH = PANEL_WIDTH - MARGIN
CANVAS_SIZE = (PANEL_X + PANEL_WIDTH, BOARD_SIZE + 2 * MARGIN)
FPS = 30

MODELS_DIR = "models"
ELO_OPTIONS = (1320, 1600, 2000, 2400, 2800, None)  # None = Stockfish sin límite
EVAL_TIME = 0.1
SPEED_DELAYS = (2.0, 1.0, 0.5, 0.25, 0.1, 0.0)  # segundos entre jugadas de la CPU
DEFAULT_SPEED = 2
BAR_EASING = 0.25  # fracción del camino que la barra recorre hacia su valor en cada fotograma

BG = (30, 36, 43)
CARD = (41, 49, 58)
CARD_ROW = (55, 65, 76)
BUTTON = (62, 73, 86)
BUTTON_HOVER = (78, 91, 106)
ACCENT = (92, 160, 230)
TEXT = (232, 236, 240)
MUTED = (140, 152, 164)
BOARD_LIGHT = (222, 227, 230)
BOARD_DARK = (140, 162, 173)
LAST_MOVE = (245, 215, 80, 115)
CHECK = (230, 40, 40, 200)
MARKER = (20, 30, 40, 75)
BAR_WHITE = (244, 246, 248)
BAR_BLACK = (52, 58, 66)
BAR_EDGE = (120, 132, 146)
BAR_TICK = (140, 148, 158)
ARROW_COLORS = ((29, 185, 170), (240, 170, 40), (160, 110, 230))  # 1.º, 2.º y 3.º puesto del modelo
ARROW_ALPHA = 215
BADGE_TEXT = (20, 24, 30)

FONT_NAMES = "avenirnext,helveticaneue,helvetica,arial"
PAD = 28

TERMINATIONS = {
    chess.Termination.CHECKMATE: "jaque mate",
    chess.Termination.STALEMATE: "ahogado",
    chess.Termination.INSUFFICIENT_MATERIAL: "material insuficiente",
    chess.Termination.SEVENTYFIVE_MOVES: "regla de 75 jugadas",
    chess.Termination.FIVEFOLD_REPETITION: "repetición",
    chess.Termination.FIFTY_MOVES: "regla de 50 jugadas",
    chess.Termination.THREEFOLD_REPETITION: "repetición",
}

Evaluation = namedtuple("Evaluation", "label prob white_end")  # white_end: el número va en el extremo blanco


def square_origin(square, flipped, board_x):
    """Esquina superior izquierda de la casilla, en píxeles del lienzo."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    if flipped:
        return board_x + (7 - file) * SQUARE, BOARD_Y + rank * SQUARE
    return board_x + file * SQUARE, BOARD_Y + (7 - rank) * SQUARE


def square_at(pos, flipped, board_x):
    """Casilla bajo el píxel del lienzo, o None si cae fuera del tablero."""
    x, y = pos[0] - board_x, pos[1] - BOARD_Y
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return None
    file, row = x // SQUARE, y // SQUARE
    if flipped:
        return chess.square(7 - file, row)
    return chess.square(file, 7 - row)


def square_center(square, flipped, board_x):
    x, y = square_origin(square, flipped, board_x)
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
    """Ventaja según Stockfish: texto corto para la barra y probabilidad de ganar de las blancas (fórmula de Lichess)."""
    outcome = board.outcome(claim_draw=True)
    if outcome:
        label = {True: "1-0", False: "0-1", None: "½-½"}[outcome.winner]
        return Evaluation(label, {True: 1.0, False: 0.0, None: 0.5}[outcome.winner], outcome.winner is not False)
    score = engine.analyse(board, chess.engine.Limit(time=EVAL_TIME))["score"].white()
    prob = score.wdl(model="lichess").expectation()
    if score.is_mate():
        return Evaluation(f"#{abs(score.mate())}", prob, score.mate() > 0)
    pawns = score.score() / 100
    if abs(pawns) < 0.05:
        return Evaluation("0.0", prob, True)
    return Evaluation(f"{pawns:+.0f}" if abs(pawns) >= 10 else f"{pawns:+.1f}", prob, pawns > 0)


def model_label(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return "Modelo final" if stem.startswith("final") else "Modelo " + stem.replace("_", " ")


def player_label(kind, elo, model_path):
    """Nombre corto para el panel; el PGN conserva el nombre completo del jugador."""
    if kind == "modelo":
        return model_label(model_path)
    if kind == "stockfish":
        return f"Stockfish {elo}" if elo else "Stockfish completo"
    return kind.capitalize()


def mouse_pos():
    x, y = pygame.mouse.get_pos()
    return x * UI_SCALE, y * UI_SCALE


def fit_text(font, text, max_width):
    while font.size(text)[0] > max_width and len(text) > 1:
        text = text[:-2].rstrip() + "…"
    return text


def draw_text(canvas, font, text, color, pos, anchor="topleft", max_width=None):
    if max_width:
        text = fit_text(font, text, max_width)
    surface = font.render(text, True, color)
    rect = surface.get_rect(**{anchor: pos})
    canvas.blit(surface, rect)
    return rect


def circle_sprite(radius, color, width=0, scale=4):
    """Círculo o anillo con bordes suaves, dibujado en grande y reducido."""
    big = pygame.Surface((2 * radius * scale,) * 2, pygame.SRCALPHA)
    pygame.draw.circle(big, color, (radius * scale,) * 2, radius * scale, width * scale)
    return pygame.transform.smoothscale(big, (2 * radius,) * 2)


def glow_sprite(size, color):
    """Degradado radial que se desvanece hacia el borde (jaque)."""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    radius = int(size * 0.62)
    for r in range(radius, 0, -2):
        alpha = round(color[3] * (1 - r / radius) ** 1.4)
        pygame.draw.circle(surface, (*color[:3], alpha), (size // 2, size // 2), r)
    return surface


def badge(canvas, fonts, center, number, color):
    x, y = round(center.x), round(center.y)
    for radius, fill in ((30, BADGE_TEXT), (26, color)):
        pygame.gfxdraw.filled_circle(canvas, x, y, radius, fill)
        pygame.gfxdraw.aacircle(canvas, x, y, radius, fill)
    draw_text(canvas, fonts.bold, str(number), BADGE_TEXT, (x, y), "center")


def draw_arrow(canvas, fonts, start, end, prob, rank, badge_t):
    """Flecha de un solo polígono con bordes suaves; el color y la insignia indican el puesto del modelo."""
    direction = (end - start).normalize()
    normal = pygame.Vector2(-direction.y, direction.x)
    half = (20 + 28 * prob ** 0.5) / 2
    head_half = half + 22
    tip = end - direction * SQUARE * 0.1
    base = tip - direction * 70
    tail = start + direction * SQUARE * 0.18
    points = [tail + normal * half, base + normal * half, base + normal * head_half, tip,
              base - normal * head_half, base - normal * half, tail - normal * half]
    points = [(round(p.x), round(p.y)) for p in points]
    color = (*ARROW_COLORS[rank], ARROW_ALPHA)
    pygame.gfxdraw.filled_polygon(canvas, points, color)
    pygame.gfxdraw.aapolygon(canvas, points, color)
    badge(canvas, fonts, start + (end - start) * badge_t, rank + 1, ARROW_COLORS[rank])


class Fonts:
    def __init__(self):
        def make(size, bold=False):
            return pygame.font.SysFont(FONT_NAMES, size, bold=bold)

        self.title = make(84, True)
        self.heading = make(28, True)
        self.normal = make(38)
        self.bold = make(38, True)
        self.small = make(32)
        self.tiny = make(26, True)


class Buttons:
    """Botones de la pantalla actual: se registran al dibujarlos y se buscan al hacer clic."""

    def __init__(self, canvas, fonts):
        self.canvas = canvas
        self.fonts = fonts
        self.items = []

    def reset(self):
        self.items = []

    def add(self, rect, label, action, on=False, enabled=True, font=None):
        rect = pygame.Rect(rect)
        hover = enabled and rect.collidepoint(mouse_pos())
        color = ACCENT if on else BUTTON_HOVER if hover else BUTTON
        pygame.draw.rect(self.canvas, color, rect, border_radius=16)
        text_color = BADGE_TEXT if on else TEXT if enabled else MUTED
        draw_text(self.canvas, font or self.fonts.normal, label, text_color, rect.center, "center",
                  rect.width - 16)
        if enabled:
            self.items.append((rect, action))

    def click(self, pos):
        """Ejecuta la acción del botón pulsado y devuelve lo que ella devuelva (False si no había botón)."""
        for rect, action in self.items:
            if rect.collidepoint(pos):
                return action()
        return False


def card(canvas, fonts, rect, title=None):
    pygame.draw.rect(canvas, CARD, rect, border_radius=24)
    if title:
        draw_text(canvas, fonts.heading, title.upper(), MUTED, (rect.x + PAD, rect.y + PAD - 4))


class Menu:
    """Pantalla inicial: una tarjeta por color con quién juega y, solo si aplica, la fuerza o el modelo."""

    def __init__(self, canvas, fonts):
        self.canvas = canvas
        self.fonts = fonts
        self.buttons = Buttons(canvas, fonts)
        self.models = sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))
        self.has_stockfish = shutil.which("stockfish") is not None
        self.kinds = {chess.WHITE: "humano",
                      chess.BLACK: "modelo" if self.models else "stockfish" if self.has_stockfish else "humano"}
        self.elos = {chess.WHITE: ELO_OPTIONS[0], chess.BLACK: ELO_OPTIONS[0]}
        self.model_index = 0
        self.play_rect = pygame.Rect(0, 0, 0, 0)

    def available(self, kind):
        return {"modelo": bool(self.models), "stockfish": self.has_stockfish}.get(kind, True)

    def config(self):
        model = self.models[self.model_index] if self.models else None
        return self.kinds, self.elos, model

    def draw_card(self, rect, color, name):
        fonts, buttons = self.fonts, self.buttons
        card(self.canvas, fonts, rect)
        swatch = (250, 250, 250) if color == chess.WHITE else (22, 26, 32)
        center = (rect.x + PAD + 22, rect.y + PAD + 24)
        pygame.gfxdraw.filled_circle(self.canvas, *center, 22, swatch)
        pygame.gfxdraw.aacircle(self.canvas, *center, 22, swatch)
        pygame.gfxdraw.aacircle(self.canvas, *center, 22, BAR_EDGE)
        draw_text(self.canvas, fonts.bold, name, TEXT, (rect.x + PAD + 62, center[1]), "midleft")

        inner = rect.width - 2 * PAD
        segment = (inner - (len(PLAYER_KINDS) - 1) * 10) // len(PLAYER_KINDS)
        top = rect.y + 120
        pygame.draw.rect(self.canvas, BG, (rect.x + PAD - 8, top - 8, inner + 16, 100), border_radius=22)
        for i, kind in enumerate(PLAYER_KINDS):
            buttons.add((rect.x + PAD + i * (segment + 10), top, segment, 84), kind.capitalize(),
                        lambda c=color, k=kind: self.kinds.__setitem__(c, k),
                        on=self.kinds[color] == kind, enabled=self.available(kind))

        y = top + 150
        kind = self.kinds[color]
        if kind == "humano":
            draw_text(self.canvas, fonts.small, "Mueves tú con el ratón.", MUTED, (rect.x + PAD, y))
        elif kind == "azar":
            draw_text(self.canvas, fonts.small, "Juega una jugada legal al azar.", MUTED, (rect.x + PAD, y))
        elif kind == "modelo":
            draw_text(self.canvas, fonts.heading, "MODELO", MUTED, (rect.x + PAD, y))
            step = lambda d: setattr(self, "model_index", (self.model_index + d) % len(self.models))
            buttons.add((rect.x + PAD, y + 50, 84, 84), "‹", lambda: step(-1), enabled=len(self.models) > 1)
            pygame.draw.rect(self.canvas, CARD_ROW, (rect.x + PAD + 96, y + 50, inner - 192, 84), border_radius=16)
            draw_text(self.canvas, fonts.bold, model_label(self.models[self.model_index]), TEXT,
                      (rect.x + rect.width // 2, y + 92), "center", inner - 220)
            buttons.add((rect.right - PAD - 84, y + 50, 84, 84), "›", lambda: step(1), enabled=len(self.models) > 1)
        else:
            draw_text(self.canvas, fonts.heading, "FUERZA (ELO)", MUTED, (rect.x + PAD, y))
            chip = (inner - 2 * 12) // 3
            for i, elo in enumerate(ELO_OPTIONS):
                col, row = i % 3, i // 3
                buttons.add((rect.x + PAD + col * (chip + 12), y + 50 + row * 96, chip, 84),
                            str(elo) if elo else "Completo", lambda c=color, e=elo: self.elos.__setitem__(c, e),
                            on=self.elos[color] == elo)

    def draw(self):
        self.buttons.reset()
        self.canvas.fill(BG)
        width = CANVAS_SIZE[0]
        draw_text(self.canvas, self.fonts.title, "Ajedrez contra el modelo", TEXT, (width // 2, 150), "center")
        card_width = (width - 2 * 140 - 48) // 2
        for i, (color, name) in enumerate(((chess.WHITE, "Blancas"), (chess.BLACK, "Negras"))):
            self.draw_card(pygame.Rect(140 + i * (card_width + 48), 280, card_width, 620), color, name)

        self.play_rect = pygame.Rect(width // 2 - 280, 1000, 560, 130)
        self.buttons.add(self.play_rect, "Jugar", lambda: True, on=True, font=self.fonts.bold)
        notes = []
        if not self.has_stockfish:
            notes.append("Stockfish no está instalado (brew install stockfish): sin él no hay barra de ventaja.")
        if not self.models:
            notes.append(f"No hay modelos .keras en {MODELS_DIR}/.")
        for i, note in enumerate(notes):
            draw_text(self.canvas, self.fonts.small, note, MUTED, (width // 2, 1210 + i * 44), "center")

    def on_click(self, pos):
        """Aplica el botón pulsado; devuelve True si fue «Jugar»."""
        return self.buttons.click(pos) is True


class ChessGame:
    def __init__(self, canvas, fonts, pieces, players, evaluator, labels):
        self.canvas = canvas
        self.fonts = fonts
        self.pieces = pieces
        self.players = players
        self.evaluator = evaluator
        self.labels = labels
        self.flipped = players[chess.WHITE] is not None and players[chess.BLACK] is None
        self.board_x = BOARD_X_WITH_BAR if evaluator else BOARD_X_NO_BAR
        self.has_cpu = any(player is not None for player in players.values())
        self.has_model = any(isinstance(player, ModelPlayer) for player in players.values())
        self.speed_index = DEFAULT_SPEED
        self.show_arrows = True
        self.buttons = Buttons(canvas, fonts)
        self.dot = circle_sprite(22, MARKER)
        self.ring = circle_sprite(SQUARE // 2 - 4, MARKER, width=10)
        self.check_glow = glow_sprite(SQUARE, CHECK)
        self.new_game()

    def name(self, color):
        player = self.players[color]
        return "Humano" if player is None else player.name

    def new_game(self):
        self.board = chess.Board()
        self.comments = []
        self.sans = []
        self.analysis = None  # (ply, color, [(san, prob, move)], ilegales) de la última jugada del modelo
        self.selected = None
        self.paused = False
        self.saved_path = None
        self.bar_shown = 0.5
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
        self.sans.append(self.board.san(move))
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
            self.sans.pop()
            if self.players[self.board.turn] is None or self.players[not self.board.turn] is not None:
                break
        self.analysis = None
        self.selected = None
        self.saved_path = None
        self.refresh_evaluation()

    def speed(self, delta):
        self.speed_index = max(0, min(len(SPEED_DELAYS) - 1, self.speed_index + delta))

    def toggle_pause(self):
        self.paused = not self.paused

    def toggle_arrows(self):
        self.show_arrows = not self.show_arrows

    def on_click(self, pos):
        """Devuelve True si se pulsó «Menú»."""
        if self.buttons.click(pos) is True:
            return True
        if self.players[self.board.turn] is not None or self.board.is_game_over(claim_draw=True):
            return False
        square = square_at(pos, self.flipped, self.board_x)
        if square is None:
            return False
        if self.selected is not None:
            move = human_move(self.board, self.selected, square)
            if move:
                self.push(move)
                return False
        piece = self.board.piece_at(square)
        self.selected = square if piece and piece.color == self.board.turn else None
        return False

    def on_key(self, key):
        if key == pygame.K_n:
            self.new_game()
        elif key == pygame.K_u:
            self.undo()
        elif key == pygame.K_SPACE and self.has_cpu:
            self.toggle_pause()
        elif key == pygame.K_a:
            self.toggle_arrows()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_UP):
            self.speed(1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN):
            self.speed(-1)

    def cpu_turn(self):
        player = self.players[self.board.turn]
        if player is None or self.paused or self.board.is_game_over(claim_draw=True):
            return
        if time.monotonic() - self.last_cpu_move < SPEED_DELAYS[self.speed_index]:
            return
        move, analysis = next_move(player, self.board)
        self.push(move, analysis)
        self.last_cpu_move = time.monotonic()

    def close(self):
        for player in self.players.values():
            if hasattr(player, "close"):
                player.close()

    def origin(self, square):
        return square_origin(square, self.flipped, self.board_x)

    def center(self, square):
        return square_center(square, self.flipped, self.board_x)

    def fill_square(self, square, color):
        overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
        overlay.fill(color)
        self.canvas.blit(overlay, self.origin(square))

    def draw_board(self):
        for square in chess.SQUARES:
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            pygame.draw.rect(self.canvas, BOARD_LIGHT if light else BOARD_DARK, (*self.origin(square), SQUARE, SQUARE))
        if self.board.move_stack:
            last = self.board.peek()
            self.fill_square(last.from_square, LAST_MOVE)
            self.fill_square(last.to_square, LAST_MOVE)
        if self.board.is_check():
            self.canvas.blit(self.check_glow, self.origin(self.board.king(self.board.turn)))
        if self.selected is not None:
            self.fill_square(self.selected, (*ACCENT, 120))

        for i in range(8):
            file_name = chess.FILE_NAMES[7 - i if self.flipped else i]
            rank_name = chess.RANK_NAMES[i if self.flipped else 7 - i]
            x, y = self.board_x + i * SQUARE, BOARD_Y + i * SQUARE
            draw_text(self.canvas, self.fonts.tiny, file_name, BOARD_LIGHT if i % 2 == 0 else BOARD_DARK,
                      (x + SQUARE - 10, BOARD_Y + BOARD_SIZE - 6), "bottomright")
            draw_text(self.canvas, self.fonts.tiny, rank_name, BOARD_DARK if i % 2 == 0 else BOARD_LIGHT,
                      (self.board_x + 10, y + 6))

        for square, piece in self.board.piece_map().items():
            self.canvas.blit(self.pieces[piece.symbol()], self.origin(square))

        if self.selected is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected:
                    sprite = self.ring if self.board.is_capture(move) else self.dot
                    self.canvas.blit(sprite, sprite.get_rect(center=self.center(move.to_square)))

        # Las flechas solo se ven mientras la última jugada es la del modelo.
        if self.show_arrows and self.analysis and self.analysis[0] == self.board.ply() - 1:
            top = self.analysis[2]
            for rank in reversed(range(len(top))):
                _, prob, move = top[rank]
                shared = sum(1 for _, _, other in top[:rank] if other.to_square == move.to_square)
                draw_arrow(self.canvas, self.fonts, self.center(move.from_square), self.center(move.to_square),
                           prob, rank, 0.7 - 0.2 * shared)

        outcome = self.board.outcome(claim_draw=True)
        if outcome:
            self.draw_result(outcome)

    def draw_result(self, outcome):
        winner = {True: "Ganan las blancas", False: "Ganan las negras", None: "Tablas"}[outcome.winner]
        rect = pygame.Rect(0, 0, 780, 250 if self.saved_path else 200)
        rect.center = (self.board_x + BOARD_SIZE // 2, BOARD_Y + BOARD_SIZE // 2)
        shadow = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow, (*CARD, 240), shadow.get_rect(), border_radius=28)
        self.canvas.blit(shadow, rect)
        pygame.draw.rect(self.canvas, ACCENT, rect, width=4, border_radius=28)
        draw_text(self.canvas, self.fonts.title, winner, TEXT, (rect.centerx, rect.y + 80), "center")
        reason = TERMINATIONS.get(outcome.termination, outcome.termination.name.lower())
        draw_text(self.canvas, self.fonts.normal, f"{outcome.result()} · {reason}", MUTED,
                  (rect.centerx, rect.y + 148), "center")
        if self.saved_path:
            draw_text(self.canvas, self.fonts.small, f"PGN: {os.path.basename(self.saved_path)}", MUTED,
                      (rect.centerx, rect.y + 206), "center", rect.width - 2 * PAD)

    def draw_bar(self):
        """Barra de ventaja: la parte blanca crece con la probabilidad de ganar de las blancas, con el número en el extremo del bando que gana."""
        evaluation = self.evaluation
        if not evaluation:
            return
        self.bar_shown += (evaluation.prob - self.bar_shown) * BAR_EASING
        size = (BAR_WIDTH, BOARD_SIZE)
        white = round(BOARD_SIZE * self.bar_shown)
        top = 0 if self.flipped else BOARD_SIZE - white
        bar = pygame.Surface(size, pygame.SRCALPHA)
        bar.fill(BAR_BLACK)
        bar.fill(BAR_WHITE, (0, top, BAR_WIDTH, white))
        for fraction in (0.25, 0.5, 0.75):
            y = round(BOARD_SIZE * fraction)
            length = 28 if fraction == 0.5 else 16
            pygame.draw.line(bar, BAR_TICK, (0, y), (length, y), 3)
            pygame.draw.line(bar, BAR_TICK, (BAR_WIDTH - length, y), (BAR_WIDTH, y), 3)
        mask = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=18)
        bar.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        rect = pygame.Rect(BAR_X, BOARD_Y, *size)
        self.canvas.blit(bar, rect)
        pygame.draw.rect(self.canvas, BAR_EDGE, rect, width=3, border_radius=18)

        at_top = self.flipped == evaluation.white_end
        extent = white if evaluation.white_end else BOARD_SIZE - white
        color = BAR_BLACK if (extent > 70) == evaluation.white_end else BAR_WHITE
        y = rect.y + 18 if at_top else rect.bottom - 18
        draw_text(self.canvas, self.fonts.tiny, evaluation.label, color, (rect.centerx, y),
                  "midtop" if at_top else "midbottom")

    def draw_card_players(self, rect):
        card(self.canvas, self.fonts, rect)
        outcome = self.board.outcome(claim_draw=True)
        for i, color in enumerate((chess.WHITE, chess.BLACK)):
            row = pygame.Rect(rect.x + 12, rect.y + 14 + i * 80, rect.width - 24, 72)
            if self.board.turn == color and not outcome:
                pygame.draw.rect(self.canvas, CARD_ROW, row, border_radius=16)
            center = (row.x + 40, row.centery)
            swatch = (250, 250, 250) if color == chess.WHITE else (22, 26, 32)
            pygame.gfxdraw.filled_circle(self.canvas, *center, 16, swatch)
            pygame.gfxdraw.aacircle(self.canvas, *center, 16, swatch)
            pygame.gfxdraw.aacircle(self.canvas, *center, 16, BAR_EDGE)
            draw_text(self.canvas, self.fonts.bold, self.labels[color], TEXT, (row.x + 76, row.centery), "midleft",
                      row.width - 100)
        if outcome:
            status = f"{outcome.result()} · {TERMINATIONS.get(outcome.termination, outcome.termination.name.lower())}"
        elif self.paused:
            status = "En pausa"
        else:
            status = f"Jugada {self.board.fullmove_number} · mueven {'blancas' if self.board.turn else 'negras'}"
        draw_text(self.canvas, self.fonts.small, status, ACCENT if self.paused else MUTED,
                  (rect.x + PAD, rect.bottom - 44), "midleft", rect.width - 2 * PAD)

    def draw_card_moves(self, rect):
        card(self.canvas, self.fonts, rect, "Jugadas")
        row_height = 42
        top = rect.y + 76
        visible = (rect.bottom - 16 - top) // row_height
        rows = (len(self.sans) + 1) // 2
        first = max(0, rows - visible)
        for r in range(first, rows):
            y = top + (r - first) * row_height
            draw_text(self.canvas, self.fonts.small, f"{r + 1}.", MUTED, (rect.x + PAD + 52, y), "topright")
            for column in (0, 1):
                index = 2 * r + column
                if index >= len(self.sans):
                    break
                x = rect.x + PAD + 76 + column * 220
                if index == len(self.sans) - 1:
                    pygame.draw.rect(self.canvas, CARD_ROW, (x - 10, y - 4, 200, row_height), border_radius=10)
                draw_text(self.canvas, self.fonts.small, self.sans[index], TEXT, (x, y), max_width=180)

    def draw_card_model(self, rect):
        card(self.canvas, self.fonts, rect, "Modelo")
        if not self.analysis:
            draw_text(self.canvas, self.fonts.small, "Aún no ha movido el modelo.", MUTED,
                      (rect.x + PAD, rect.y + 90))
            return
        ply, color, top, illegal_mass = self.analysis
        side = "blancas" if color == chess.WHITE else "negras"
        draw_text(self.canvas, self.fonts.small, f"{side} · jugada {ply // 2 + 1}", MUTED,
                  (rect.right - PAD, rect.y + PAD - 2), "topright")
        track_x, track_width = rect.x + PAD + 190, rect.width - 2 * PAD - 190 - 120
        for i, (san, prob, _) in enumerate(top):
            cy = rect.y + 94 + i * 56
            center = (rect.x + PAD + 20, cy)
            pygame.gfxdraw.filled_circle(self.canvas, *center, 20, ARROW_COLORS[i])
            pygame.gfxdraw.aacircle(self.canvas, *center, 20, ARROW_COLORS[i])
            draw_text(self.canvas, self.fonts.bold, str(i + 1), BADGE_TEXT, center, "center")
            draw_text(self.canvas, self.fonts.bold, san, TEXT, (rect.x + PAD + 52, cy), "midleft", 130)
            pygame.draw.rect(self.canvas, CARD_ROW, (track_x, cy - 9, track_width, 18), border_radius=9)
            pygame.draw.rect(self.canvas, ARROW_COLORS[i], (track_x, cy - 9, max(18, round(track_width * prob)), 18),
                             border_radius=9)
            draw_text(self.canvas, self.fonts.small, f"{prob * 100:.1f} %", TEXT, (rect.right - PAD, cy), "midright")
        draw_text(self.canvas, self.fonts.small, f"En jugadas ilegales: {illegal_mass * 100:.1f} %", MUTED,
                  (rect.x + PAD, rect.bottom - 34), "midleft", rect.width - 2 * PAD)

    def draw_card_controls(self, rect):
        card(self.canvas, self.fonts, rect)
        add = self.buttons.add
        over = self.board.is_game_over(claim_draw=True)
        x, y, width = rect.x + PAD, rect.y + 24, rect.width - 2 * PAD

        def row(labels_actions):
            nonlocal y
            gap = 12
            each = (width - gap * (len(labels_actions) - 1)) // len(labels_actions)
            for i, (label, action, kwargs) in enumerate(labels_actions):
                add((x + i * (each + gap), y, each, 68), label, action, font=self.fonts.small, **kwargs)
            y += 68 + gap

        row([("Nueva", self.new_game, {}), ("Deshacer", self.undo, {}), ("Menú", lambda: True, {})])
        if self.has_cpu:
            second = [("Seguir" if self.paused else "Pausa", self.toggle_pause, {"enabled": not over})]
            if self.has_model:
                second.append(("Flechas", self.toggle_arrows, {"on": self.show_arrows}))
            row(second)
            delay = SPEED_DELAYS[self.speed_index]
            label = "Velocidad máxima" if delay == 0 else f"Velocidad {1 / delay:g}×"
            add((x, y, 80, 68), "−", lambda: self.speed(-1), font=self.fonts.bold, enabled=self.speed_index > 0)
            draw_text(self.canvas, self.fonts.small, label, TEXT, (rect.centerx, y + 34), "center", width - 200)
            add((x + width - 80, y, 80, 68), "+", lambda: self.speed(1), font=self.fonts.bold,
                enabled=self.speed_index < len(SPEED_DELAYS) - 1)

    def draw_panel(self):
        self.buttons.reset()
        x, bottom = PANEL_X, BOARD_Y + BOARD_SIZE
        controls_h = 116 + (2 * 80 if self.has_cpu else 0)
        model_h = 290 if self.has_model else 0
        controls = pygame.Rect(x, bottom - controls_h, CARD_WIDTH, controls_h)
        model = pygame.Rect(x, controls.y - MARGIN // 2 - model_h, CARD_WIDTH, model_h)
        players = pygame.Rect(x, BOARD_Y, CARD_WIDTH, 232)
        moves_bottom = (model.y if self.has_model else controls.y) - MARGIN // 2
        moves = pygame.Rect(x, players.bottom + MARGIN // 2, CARD_WIDTH, moves_bottom - players.bottom - MARGIN // 2)
        self.draw_card_players(players)
        self.draw_card_moves(moves)
        if self.has_model:
            self.draw_card_model(model)
        self.draw_card_controls(controls)

    def draw(self):
        self.canvas.fill(BG)
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
        labels = {color: player_label(kinds[color], elos[color], model_path) for color in chess.COLORS}
        self.game = ChessGame(self.canvas, self.fonts, self.pieces, players, self.evaluator, labels)

    def back_to_menu(self):
        self.game.close()
        self.game = None

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = (event.pos[0] * UI_SCALE, event.pos[1] * UI_SCALE)
            if self.game:
                if self.game.on_click(pos):
                    self.back_to_menu()
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
