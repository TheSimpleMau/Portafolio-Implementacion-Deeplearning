import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 300
plt.rcParams.update({"font.size": 12})
GRAPHICS_DIR = "./graficas"
PIECE_NAMES = ("Peón", "Caballo", "Alfil", "Torre", "Dama", "Rey")


def ensure_graphics_directory():
    os.makedirs(GRAPHICS_DIR, exist_ok=True)


def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, name), dpi=DPI)
    plt.close()


def elo_hist(elo):
    ensure_graphics_directory()
    plt.figure(figsize=(6, 4))
    plt.hist(elo, bins=80, color="steelblue")
    plt.axvline(np.median(elo), color="black", linestyle="--", label=f"Mediana = {np.median(elo):.0f}")
    plt.xlabel("Elo del jugador que mueve")
    plt.ylabel("Posiciones")
    plt.title("Distribución del Elo")
    plt.legend()
    _save("eda_elo.png")


def ply_hist(ply, cut=10):
    ensure_graphics_directory()
    plt.figure(figsize=(6, 4))
    plt.hist(ply, bins=np.arange(0, min(ply.max(), 200) + 2) - 0.5, color="steelblue")
    plt.axvline(cut + 0.5, color="black", linestyle="--", label=f"ply = {cut}")
    plt.xlabel("Ply (medias jugadas desde el inicio)")
    plt.ylabel("Posiciones")
    plt.title("Ply de cada posición")
    plt.legend()
    _save("eda_ply.png")


def legal_moves_hist(n_legal):
    ensure_graphics_directory()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
    left.hist(n_legal, bins=np.arange(1, n_legal.max() + 2) - 0.5, color="steelblue")
    left.axvline(n_legal.mean(), color="black", linestyle="--", label=f"Media = {n_legal.mean():.1f}")
    left.set_xlabel("Jugadas legales en la posición")
    left.set_ylabel("Posiciones")
    left.set_title("Número de jugadas legales")
    left.legend()

    edges = [1, 2, 5, 10, 20, 30, 40, 60, n_legal.max() + 1]
    labels, hit_rate, share = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (n_legal >= low) & (n_legal < high)
        if not mask.any():
            continue
        labels.append(str(low) if high - 1 == low else f"{low}-{high - 1}")
        hit_rate.append((1 / n_legal[mask]).mean() * 100)
        share.append(mask.mean() * 100)
    bars = right.bar(labels, hit_rate, color="darkorange")
    for bar, pct in zip(bars, share):
        right.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 6) + 1.5, f"{pct:.1f}%",
                   ha="center", va="bottom", fontsize=9)
    right.axhline((1 / n_legal).mean() * 100, color="black", linestyle="--",
                  label=f"Promedio = {(1 / n_legal).mean() * 100:.1f}%")
    right.set_xlabel("Jugadas legales")
    right.set_ylabel("Acierto esperado al azar (%)")
    right.set_title("Acierto del azar por jugadas legales")
    right.set_ylim(0, 112)
    right.tick_params(axis="x", rotation=45)
    right.legend()
    _save("eda_jugadas_legales.png")


def class_frequency(counts):
    ensure_graphics_directory()
    present = np.sort(counts[counts > 0])[::-1]
    cumulative = np.cumsum(present) / present.sum()
    fig, left = plt.subplots(figsize=(6.5, 4.5))
    left.plot(np.arange(1, len(present) + 1), present, color="steelblue")
    left.set_yscale("log")
    left.set_xlabel("Rango de la clase (de la más a la menos frecuente)")
    left.set_ylabel("Posiciones (escala log)", color="steelblue")
    right = left.twinx()
    right.plot(np.arange(1, len(present) + 1), cumulative * 100, color="darkorange")
    right.set_ylabel("Porcentaje acumulado", color="darkorange")
    right.set_ylim(0, 100)
    plt.title(f"Frecuencia de las {len(present):,} clases que aparecen")
    _save("eda_frecuencia_clases.png")


def moved_piece_bar(piece):
    ensure_graphics_directory()
    shares = [(piece == code).mean() * 100 for code in range(1, 7)]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(PIECE_NAMES, shares, color="steelblue")
    plt.bar_label(bars, fmt="%.1f%%")
    plt.ylim(0, max(shares) * 1.15)
    plt.ylabel("Porcentaje de las jugadas")
    plt.title("Pieza que se mueve")
    _save("eda_pieza_movida.png")


def capture_bar(captured):
    ensure_graphics_directory()
    values = [0, 1, 3, 5, 9]
    labels = ["Tranquila", "Peón (1)", "Caballo o\nalfil (3)", "Torre (5)", "Dama (9)"]
    shares = [(captured == v).mean() * 100 for v in values]
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(labels, shares, color=["gray"] + ["firebrick"] * 4)
    plt.bar_label(bars, fmt="%.1f%%")
    plt.ylim(0, max(shares) * 1.15)
    plt.ylabel("Porcentaje de las jugadas")
    plt.title("Valor de la pieza capturada")
    _save("eda_capturas.png")


def phase_bar(phase, phase_names):
    ensure_graphics_directory()
    shares = [(phase == code).mean() * 100 for code in range(len(phase_names))]
    plt.figure(figsize=(5.5, 4))
    bars = plt.bar([name.capitalize() for name in phase_names], shares, color="steelblue")
    plt.bar_label(bars, fmt="%.1f%%")
    plt.ylim(0, max(shares) * 1.15)
    plt.ylabel("Porcentaje de las posiciones")
    plt.title("Fase de la partida")
    _save("eda_fases.png")


def repeated_positions_by_ply(repeated_rate, ceiling_rate):
    ensure_graphics_directory()
    plies = np.arange(len(repeated_rate))
    plt.figure(figsize=(8, 4.5))
    plt.plot(plies, repeated_rate * 100, marker=".", label="Posiciones que aparecen en 2 o más partidas")
    plt.plot(plies, ceiling_rate * 100, marker=".", label="Techo de top-1 en esas posiciones")
    plt.xlabel("Ply")
    plt.ylabel("Porcentaje")
    plt.ylim(0, 100)
    plt.title("Posiciones que se repiten entre partidas, por ply")
    plt.legend()
    _save("eda_posiciones_repetidas.png")
