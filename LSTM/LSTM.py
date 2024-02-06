import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from Helper import DataProcessor


def build_lstm_1(
        dataset: DataProcessor,
        epochs=25,
        batch_size=32,
        lstm_dim: int = 350,
        dense_dim: int = 64):
    """
      Builds, compiles, and fits our Multivariate_LSTM baseline model.
    """

    n_timesteps, n_features, n_outputs = dataset.X_train.shape[1], dataset.X_train.shape[2], dataset.y_train.shape[1]
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True)]
    opt = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(LSTM(lstm_dim, input_shape=(n_timesteps, n_features)))
    model.add(Dense(dense_dim))
    model.add(Dense(n_outputs))

    print("compliling baseline model")
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])

    print("fitting model")
    history = model.fit(dataset.X_train, dataset.y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(dataset.X_test, dataset.y_test), verbose=1)

    return model, history
