import random

import chess
import numpy as np

from model import experiment
from model.chessModels import compile_model, get_model
from model.etlProcess import split_data
from model.experiment import batch_generator, progress_line, steps
from model.preprocessing import encode_move, encode_position, expand_planes


def synthetic_data(n, seed=0):
    """Posiciones al azar cuya jugada es su propio índice, para saber de qué fila sale cada lote."""
    rng = np.random.default_rng(seed)
    return {
        "squares": rng.integers(0, 13, size=(n, 64)).astype(np.int8),
        "castling": rng.integers(0, 16, size=n).astype(np.uint8),
        "ep": rng.integers(-1, 8, size=n).astype(np.int8),
        "move": np.arange(n, dtype=np.int16),
    }


def random_positions(n, seed=0):
    """n posiciones distintas de partidas al azar, cada una con una jugada legal al azar."""
    rng = random.Random(seed)
    seen, columns = set(), {"squares": [], "castling": [], "ep": [], "move": []}
    while len(seen) < n:
        board = chess.Board()
        for _ in range(rng.randint(4, 60)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over() or board.fen() in seen:
            continue
        seen.add(board.fen())
        squares, castling, ep = encode_position(board)
        move = rng.choice([m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)])
        for key, value in zip(columns, (squares, castling, ep, encode_move(move, board.turn == chess.BLACK))):
            columns[key].append(value)
    return {key: np.array(values) for key, values in columns.items()}


def test_una_epoca_recorre_cada_indice_una_vez_con_su_tablero():
    data = synthetic_data(20)
    idx = np.arange(3, 13)
    generator = batch_generator(data, idx, batch_size=4, shuffle=True, seed=0)

    seen = []
    for _ in range(steps(idx, 4)):
        x, y = next(generator)
        assert x.dtype == np.float32
        expected = expand_planes(data["squares"][y], data["castling"][y], data["ep"][y])
        np.testing.assert_array_equal(x, expected)
        seen.extend(y.tolist())
    assert sorted(seen) == idx.tolist()


def test_tiempo_restante_usa_la_epoca_anterior_y_su_validacion():
    # Mitad de la época 2 de 3 a 1 s por paso; la época anterior duró 120 s, 20 de ellos de validación.
    line = progress_line(2, 3, step=50, n_steps=100, seconds_per_step=1, epoch_seconds=120, val_seconds=20)
    assert "época 2/3 · paso 50/100 · 1000 ms/paso · época: faltan 1 min · total: faltan 3 min" in line
    # En la primera época todavía no hay validación medida: solo cuentan los pasos.
    line = progress_line(1, 2, step=25, n_steps=100, seconds_per_step=1)
    assert "época: faltan 1 min · total: faltan 3 min" in line


def test_cordura_el_modelo_memoriza_pocas_posiciones():
    """Si no aprende de memoria 500 posiciones, el flujo tablero→jugada está roto."""
    data = random_positions(500)
    idx = np.arange(500)
    model = get_model()
    compile_model(model)
    history = model.fit(batch_generator(data, idx, batch_size=50, shuffle=True, seed=0),
                        steps_per_epoch=steps(idx, 50), epochs=15, shuffle=False, verbose=0)
    assert history.history["accuracy"][-1] > 0.9


def test_final_entrena_con_train_y_val_y_evalua_solo_en_test(tmp_path, monkeypatch):
    """La corrida final no tiene val para detenerse: entrena con train ∪ val, evalúa en test y no va al registro."""
    data = synthetic_data(300)
    data["game_id"] = np.arange(300) // 3  # 100 partidas de 3 posiciones
    train, val, test = split_data(data["game_id"])
    trained, evaluated = [], []

    def recording_generator(data, idx, *args, **kwargs):
        trained.append(idx)
        return batch_generator(data, idx, *args, **kwargs)

    def fake_evaluation(model, data, idx):
        evaluated.append(idx)
        values = {"posiciones": len(idx), "top1": 0.0, "top3": 0.0, "top5": 0.0, "top1_ilegal": 0.0}
        return {"por_corte": {"todas": values}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "load_data", lambda: data)
    monkeypatch.setattr(experiment, "batch_generator", recording_generator)
    monkeypatch.setattr(experiment.evaluation, "evaluate_model", fake_evaluation)
    (tmp_path / "models").mkdir()
    config = {"name": "final", "model": {}, "batch_size": 64, "learning_rate": 0.001, "epochs": 1, "seed": 0,
              "model_path": "models/final.keras"}
    experiment.run_final(config)

    np.testing.assert_array_equal(trained[0], np.sort(np.concatenate([train, val])))
    assert [idx.tolist() for idx in evaluated] == [test.tolist()]
    assert (tmp_path / "models" / "final.keras").exists()
    assert not (tmp_path / "resultados" / "registro_ejecuciones.csv").exists()
