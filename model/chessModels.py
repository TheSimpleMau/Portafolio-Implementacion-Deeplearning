from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from model.preprocessing import N_CLASSES, N_PLANES


def get_model(input_shape=(8, 8, N_PLANES), n_classes=N_CLASSES, filters=(64, 128), dense_units=1024,
              dropout=0.0, dropout_after="dense_1"):
    """M1: CNN sencilla sobre el tablero one-hot; sin pooling porque el tablero es de 8×8.

    Los valores por defecto dan M1; con dropout > 0 agrega un Dropout justo después de la capa dropout_after.
    """
    layers = [
        Conv2D(filters[0], 3, padding="same", activation="relu", name="conv_1"),
        Conv2D(filters[1], 3, padding="same", activation="relu", name="conv_2"),
        Flatten(name="flatten"),
        Dense(dense_units, activation="relu", name="dense_1"),
        Dense(n_classes, activation="softmax", name="dense_2"),
    ]
    if dropout:
        position = [layer.name for layer in layers].index(dropout_after) + 1
        layers.insert(position, Dropout(dropout, name="dropout"))
    model = Sequential([Input(input_shape), *layers])
    return model


def compile_model(model, learning_rate=0.001):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
