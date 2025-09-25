import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, InputLayer

# Fix the random seed for reproducibility (ensures same initialization each run)
tf.random.set_seed(42)

class DNN:
    """
    A wrapper class for building and training a customizable Deep Neural Network (DNN)
    using TensorFlow/Keras.

    Attributes
    ----------
    model : keras.Model
        The underlying Keras Sequential model.

    Methods
    -------
    get_model():
        Returns the compiled Keras model.
    train(X_train, y_train, epochs=50):
        Trains the model on provided training data.
    """

    def __init__(self, input_shape, layers, learning_rate):
        """
        Initializes the DNN model.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input features (e.g., (num_features,)).
        layers : list of int
            List specifying the number of units in each hidden layer.
            Example: [32, 16, 8] creates 3 hidden layers with 32, 16, and 8 neurons.
        learning_rate : float
            Learning rate for the optimizer (SGD).
        """

        # Initialize a Sequential model
        self.model = models.Sequential()

        # Add input layer (expects input with given shape)
        self.model.add(InputLayer(input_shape=input_shape))

        # Add hidden layers based on the provided configuration
        for units in layers:
            self.model.add(Dense(units, activation='relu'))  # ReLU for non-linearity

        # Add output layer (single neuron for regression task, no activation)
        self.model.add(Dense(1, activation='linear'))

        # Define optimizer (Stochastic Gradient Descent with custom learning rate)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        # Compile the model with Mean Squared Error loss (for regression)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

        # Print model summary for verification
        self.model.summary()

    def get_model(self):
        """
        Returns the compiled Keras model.

        Returns
        -------
        keras.Model
            The compiled Sequential model.
        """
        return self.model
    
    def train(self, X_train, y_train, epochs=50):
        """
        Trains the DNN model on training data.

        Parameters
        ----------
        X_train : numpy.ndarray or pandas.DataFrame
            Training features.
        y_train : numpy.ndarray or pandas.Series
            Target values corresponding to X_train.
        epochs : int, optional (default=50)
            Number of training epochs.

        Returns
        -------
        history : keras.callbacks.History
            Training history containing loss values for each epoch.
        """
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=0.1,  # Reserve 10% of training data for validation
            verbose=0              # Suppress training output for cleaner logs
        )
