from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Sequential

from model.preprocessing import N_CLASSES, N_PLANES


def get_model(input_shape=(8, 8, N_PLANES), n_classes=N_CLASSES):
    """M1: CNN sencilla sobre el tablero one-hot; sin pooling porque el tablero es de 8×8."""
    model = Sequential([
        Input(input_shape),
        Conv2D(64, 3, padding="same", activation="relu", name="conv_1"),
        Conv2D(128, 3, padding="same", activation="relu", name="conv_2"),
        Flatten(name="flatten"),
        Dense(1024, activation="relu", name="dense_1"),
        Dense(n_classes, activation="softmax", name="dense_2"),
    ])
    return model


def compile_model(model):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
