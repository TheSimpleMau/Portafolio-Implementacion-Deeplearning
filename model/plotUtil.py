import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
plt.rcParams.update({"font.size": 12})
GRAPHICS_DIR = "./graficas"
PIECE_NAMES = ("Peón", "Caballo", "Alfil", "Torre", "Dama", "Rey")
# Volúmenes de la Encyclopaedia of Chess Openings.
ECO_GROUP_NAMES = {
    "A": "A: flanco\ny otras", "B": "B: semiabiertas", "C": "C: abiertas\ny Francesa",
    "D": "D: cerradas y\nsemicerradas", "E": "E: indias",
}


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
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(np.arange(1, len(present) + 1), cumulative * 100, color="steelblue")
    for share in (0.5, 0.8, 0.95):
        n_classes = np.searchsorted(cumulative, share) + 1
        plt.plot([0, n_classes, n_classes], [share * 100, share * 100, 0], color="gray", linestyle="--")
        plt.text(n_classes + 20, share * 100 - 6, f"{n_classes} jugadas: {share:.0%}")
    plt.xlim(0, len(present))
    plt.ylim(0, 102)
    plt.xlabel("Jugadas distintas, de la más a la menos común")
    plt.ylabel("% del total de jugadas")
    plt.title("Pocas jugadas concentran casi todo")
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


def eco_group_bar(shares):
    ensure_graphics_directory()
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar([ECO_GROUP_NAMES[g] for g in shares.index], shares.to_numpy() * 100, color="steelblue")
    plt.bar_label(bars, fmt="%.1f%%")
    plt.ylim(0, shares.max() * 100 * 1.15)
    plt.ylabel("Porcentaje de las partidas")
    plt.title("Partidas por grupo de aperturas (ECO)")
    _save("eda_grupos_eco.png")


def opening_families_bar(shares):
    ensure_graphics_directory()
    plt.figure(figsize=(7.5, 7))
    bars = plt.barh(shares.index[::-1], shares.to_numpy()[::-1] * 100, color="steelblue")
    plt.bar_label(bars, fmt="%.1f%%", padding=2)
    plt.xlim(0, shares.max() * 100 * 1.18)
    plt.xlabel("Porcentaje de las partidas")
    plt.title(f"Las {len(shares)} familias de aperturas más jugadas")
    _save("eda_familias_aperturas.png")


def opening_cumulative(counts):
    ensure_graphics_directory()
    present = np.sort(counts)[::-1]
    cumulative = np.cumsum(present) / present.sum()
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(np.arange(1, len(present) + 1), cumulative * 100, color="steelblue")
    marks = [(share, np.searchsorted(cumulative, share) + 1) for share in (0.5, 0.8, 0.95)]
    for share, n_families in marks:
        plt.plot([0, n_families, n_families], [share * 100, share * 100, 0], color="gray", linestyle="--")
        # Las etiquetas van a la derecha de la última marca para que ninguna línea las cruce.
        plt.text(marks[-1][1] + 4, share * 100 - 6, f"{n_families} familias: {share:.0%}")
    plt.xlim(0, len(present))
    plt.ylim(0, 102)
    plt.xlabel("Familias de aperturas, de la más a la menos jugada")
    plt.ylabel("% de las partidas")
    plt.title("Pocas familias concentran casi todas las partidas")
    _save("eda_aperturas_acumulado.png")


def first_moves_bar(first_moves, replies, top=5):
    """Primera jugada de las blancas y respuesta de las negras a 1.e4 y 1.d4 (las demás, en «Otras»)."""
    ensure_graphics_directory()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [(first_moves, "Primera jugada de las blancas", "1.")]
    panels += [(shares, f"Respuesta de las negras a 1.{move}", "1...") for move, shares in replies.items()]
    for ax, (shares, title, prefix) in zip(axes, panels):
        shown = shares[:top]
        values = list(shown.to_numpy() * 100) + [(1 - shown.sum()) * 100]
        bars = ax.bar([prefix + m for m in shown.index] + ["Otras"], values,
                      color=["steelblue"] * len(shown) + ["gray"])
        ax.bar_label(bars, fmt="%.1f%%", fontsize=10)
        ax.set_ylim(0, max(values) * 1.15)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Porcentaje de las partidas")
    _save("eda_primera_jugada.png")


def openings_by_speed(shares, games_per_speed):
    """Grupos ECO dentro de cada ritmo (shares: ritmos × grupos, filas que suman 1)."""
    ensure_graphics_directory()
    speeds = [s for s in games_per_speed.index if s in shares.index]
    width = 0.8 / len(speeds)
    plt.figure(figsize=(8, 4.5))
    for code, speed in enumerate(speeds):
        offset = (code - (len(speeds) - 1) / 2) * width
        plt.bar(np.arange(len(shares.columns)) + offset, shares.loc[speed].to_numpy() * 100, width,
                label=f"{speed} ({games_per_speed[speed]:,} partidas)")
    plt.xticks(np.arange(len(shares.columns)), [ECO_GROUP_NAMES[g] for g in shares.columns])
    plt.ylim(0, shares.to_numpy().max() * 100 * 1.4)
    plt.ylabel("Porcentaje de las partidas del ritmo")
    plt.title("Grupos de aperturas por ritmo de juego")
    plt.legend()
    _save("eda_aperturas_por_ritmo.png")


def termination_bar(shares, games_per_speed):
    """Cómo terminan las partidas de cada ritmo (shares: ritmos × tipo de terminación de Lichess)."""
    ensure_graphics_directory()
    names = {"Normal": "Normal (mate, abandono o tablas)", "Time forfeit": "Tiempo agotado",
             "Unterminated": "Sin terminar"}
    grouped = shares.T.groupby(lambda column: names.get(column, "Otras")).sum().T
    columns = [n for n in list(names.values()) + ["Otras"] if n in grouped.columns]
    speeds = [s for s in games_per_speed.index if s in grouped.index][::-1]
    colors = {"Normal (mate, abandono o tablas)": "steelblue", "Tiempo agotado": "darkorange",
              "Sin terminar": "gray", "Otras": "firebrick"}
    plt.figure(figsize=(8, 4.5))
    left = np.zeros(len(speeds))
    for column in columns:
        values = grouped.loc[speeds, column].to_numpy() * 100
        bars = plt.barh(speeds, values, left=left, label=column, color=colors[column])
        plt.bar_label(bars, labels=[f"{v:.1f}%" if v >= 5 else "" for v in values], label_type="center")
        left += values
    plt.xlim(0, 100)
    plt.xlabel("Porcentaje de las partidas del ritmo")
    plt.title("Cómo terminan las partidas")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _save("eda_terminacion.png")


def game_length_hist(plies):
    ensure_graphics_directory()
    plt.figure(figsize=(6.5, 4))
    plt.hist(plies, bins=np.arange(0, min(plies.max(), 250) + 2, 2) - 0.5, color="steelblue")
    plt.axvline(np.median(plies), color="black", linestyle="--", label=f"Mediana = {np.median(plies):.0f}")
    plt.xlabel("Plies de la partida")
    plt.ylabel("Partidas")
    plt.title("Duración de las partidas")
    plt.legend()
    _save("eda_duracion_partidas.png")


def accuracy_by_opening(results):
    """Top-1 del modelo y del «libro» en val por grupo ECO y por familia de aperturas."""
    ensure_graphics_directory()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 2]})
    for ax, cut, title in ((axes[0], "grupo_eco", "Por grupo ECO"), (axes[1], "familia", "Por familia")):
        names = list(results[cut]["modelo"])
        labels = [ECO_GROUP_NAMES.get(n, n).replace("\n", " ") if cut == "grupo_eco" else n for n in names]
        x = np.arange(len(names))
        for offset, source, color in ((-0.2, "modelo", "steelblue"), (0.2, "libro", "gray")):
            values = [results[cut][source][n]["top1"] * 100 for n in names]
            ax.bar(x + offset, values, 0.4, color=color, label="Modelo" if source == "modelo" else "«Libro»")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylim(0, 55)
        ax.set_title(title)
        ax.legend(loc="upper center", ncol=2)
    axes[0].set_ylabel("Top-1 en validación (%)")
    _save("acierto_por_apertura.png")


def split_comparison(elo, ply, phase, splits, split_names, phase_names):
    """Distribuciones de Elo, ply y fase en cada conjunto (splits = índices de cada uno)."""
    ensure_graphics_directory()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    width = 0.8 / len(splits)
    for code, (idx, name) in enumerate(zip(splits, split_names)):
        axes[0].hist(elo[idx], bins=60, density=True, histtype="step", label=name)
        axes[1].hist(ply[idx], bins=np.arange(0, 151) - 0.5, density=True, histtype="step", label=name)
        shares = np.bincount(phase[idx], minlength=len(phase_names)) / len(idx) * 100
        offset = (code - (len(splits) - 1) / 2) * width
        axes[2].bar(np.arange(len(phase_names)) + offset, shares, width, label=name)
    axes[0].set_xlabel("Elo del que mueve")
    axes[1].set_xlabel("Ply")
    axes[2].set_xticks(np.arange(len(phase_names)), [n.capitalize() for n in phase_names])
    axes[0].set_ylabel("Densidad")
    axes[2].set_ylabel("Porcentaje")
    axes[0].set_title("Elo")
    axes[1].set_title("Ply")
    axes[2].set_title("Fase")
    for ax in axes:
        ax.legend()
    _save("comparacion_train_val.png")


def training_history(history, name):
    """Pérdida y accuracy por época en train y val; la línea marca la época que guarda el checkpoint."""
    ensure_graphics_directory()
    df = pd.DataFrame(history)
    df.index += 1
    best = df["val_loss"].idxmin()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df.plot(y=["loss", "val_loss"], ax=axes[0], marker=".", title="Pérdida por época")
    (df[["accuracy", "val_accuracy"]] * 100).plot(ax=axes[1], marker=".", title="Accuracy (top-1) por época")
    axes[1].set_ylabel("Porcentaje")
    for ax in axes:
        ax.axvline(best, color="black", linestyle="--", label=f"Mejor época = {best}")
        ax.set_xlabel("Época")
        ax.legend()
    _save(f"{name}_curvas.png")
