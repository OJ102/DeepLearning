from sklearn.linear_model import SGDRegressor, LinearRegression

class LinearRegressionModel:
    def __init__(self, learning_rate):
        self.model = SGDRegressor(learning_rate='constant', eta0=learning_rate, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    