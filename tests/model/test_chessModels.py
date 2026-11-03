import pytest
from tensorflow.keras.layers import Dropout

from model.chessModels import compile_model, get_model


def test_valores_por_defecto_dan_m1():
    model = get_model()
    assert model.count_params() == 12_671_744
    assert not any(isinstance(layer, Dropout) for layer in model.layers)


def test_hiperparametros_llegan_al_modelo():
    model = get_model(filters=(32, 64), dense_units=512)
    compile_model(model, learning_rate=0.0003)
    assert model.get_layer("conv_1").filters == 32
    assert model.get_layer("conv_2").filters == 64
    assert model.get_layer("dense_1").units == 512
    assert float(model.optimizer.learning_rate) == pytest.approx(0.0003)


@pytest.mark.parametrize("after", ["flatten", "dense_1"])
def test_dropout_queda_despues_de_la_capa_pedida(after):
    model = get_model(dropout=0.3, dropout_after=after)
    names = [layer.name for layer in model.layers]
    assert names[names.index(after) + 1] == "dropout"
    assert model.get_layer("dropout").rate == 0.3
    assert model.count_params() == 12_671_744  # Dropout no agrega parámetros
