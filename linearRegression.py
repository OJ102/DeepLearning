from sklearn.linear_model import SGDRegressor

class LinearRegressionModel:
    """
    A wrapper class for training a linear regression model using 
    stochastic gradient descent (SGD) from scikit-learn.

    This class mimics a simple linear regression but allows 
    manual control of the learning rate, making it useful for 
    experimentation and comparison with deep learning models.

    Attributes
    ----------
    model : SGDRegressor
        The underlying scikit-learn SGD regressor model.

    Methods
    -------
    train(X_train, y_train):
        Fits the model on training data.
    predict(X_test):
        Generates predictions for new input data.
    """

    def __init__(self, learning_rate):
        """
        Initializes the Linear Regression model using SGD.

        Parameters
        ----------
        learning_rate : float
            The constant learning rate (eta0) for gradient descent updates.
            Example values: 0.1, 0.01, 0.001
        """

        # SGDRegressor is used here instead of normal LinearRegression
        # to allow explicit learning rate tuning and iterative updates.
        self.model = SGDRegressor(
            learning_rate='constant',  # keep learning rate fixed
            eta0=learning_rate,        # set the step size (learning rate)
            random_state=42            # ensure reproducibility
        )

    def train(self, X_train, y_train):
        """
        Trains the linear regression model using training data.

        Parameters
        ----------
        X_train : numpy.ndarray or pandas.DataFrame
            Training feature matrix.
        y_train : numpy.ndarray or pandas.Series
            Target values corresponding to X_train.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts target values for given input features.

        Parameters
        ----------
        X_test : numpy.ndarray or pandas.DataFrame
            Input features for which predictions are required.

        Returns
        -------
        numpy.ndarray
            Predicted target values.
        """
        return self.model.predict(X_test)
