import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, InputLayer

tf.random.set_seed(42)
class DNN:
    def __init__(self, input_shape, layers, learning_rate):
        self.model = models.Sequential()
        self.model.add(InputLayer(input_shape=input_shape))
        for units in layers:
            self.model.add(Dense(units, activation='relu'))
        self.model.add(Dense(1, activation='linear'))

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        self.model.summary()

    def get_model(self):
        return self.model
    
    def train(self, X_train, y_train, epochs=50):
        self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, verbose=0)

