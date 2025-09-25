import os
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import tensorflow as tf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data using MSE and R² metrics.

    Parameters
    ----------
    model : object
        Trained model with a .predict() method (e.g., LinearRegression, Keras model).
    X_test : numpy.ndarray or pandas.DataFrame
        Test feature matrix.
    y_test : numpy.ndarray or pandas.Series
        True target values.

    Returns
    -------
    tuple (mse, r2)
        - mse : float, Mean Squared Error
        - r2 : float, R² score
    """
    y_pred = model.predict(X_test)

    # Compute Mean Squared Error
    try:
        mse = mean_squared_error(y_test, y_pred)
    except Exception as e:
        print(f"Error calculating MSE: {e}")
        mse = float('nan')

    # Compute R² Score
    try:
        r2 = r2_score(y_test, y_pred)
    except Exception as e:
        print(f"Error calculating R^2: {e}")
        r2 = float('nan')

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    return mse, r2


def linear_model_performance_plot(y_test, y_pred, model_name, learning_rate):
    """
    Generate a scatter plot of actual vs. predicted values for linear models.

    Parameters
    ----------
    y_test : numpy.ndarray or pandas.Series
        True target values.
    y_pred : numpy.ndarray
        Predicted target values.
    model_name : str
        Name of the model (used in title and filename).
    learning_rate : float
        Learning rate used during training (for logging in title/filename).
    """
    os.makedirs("plots", exist_ok=True)  # Ensure plots folder exists

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', lw=2, label="Perfect Prediction"
    )
    plt.title(f'{model_name} Predictions vs Actual (LR={learning_rate}) - {datetime.now()}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)

    # Save plot with timestamp in filename
    filename = f'plots/{model_name}_performance_lr{learning_rate}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename)
    print(f"Saved performance plot: {filename}")


def model_performance_loss_plot(history, model_name, learning_rate):
    """
    Generate a training vs validation loss curve for DNN models.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history object returned by model.fit().
    model_name : str
        Name of the model (used in title and filename).
    learning_rate : float
        Learning rate used during training (for logging in title/filename).
    """
    os.makedirs("plots", exist_ok=True)  # Ensure plots folder exists
    print(f"Plotting training history. TimeStamp: {datetime.now()}")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss Over Epochs (LR={learning_rate}) - {datetime.now()}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save plot with timestamp
    filename = f'plots/{model_name}_loss_lr{learning_rate}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename)
    print(f"Saved loss plot: {filename}")


def test_model(model_type, model_path, x_test, y_test):
    """
    Load a trained model (Linear Regression or DNN),
    evaluate performance on a given test dataset, and
    return key metrics.

    Parameters
    ----------
    model_type : str
        Type of the model: "linear" or "dnn".
    model_path : str
        Path to the saved model file (.pkl for linear, .h5 for DNN).
    x_test : numpy.ndarray or pandas.DataFrame
        Test feature matrix.
    y_test : numpy.ndarray or pandas.Series
        True target values for the test set.

    Returns
    -------
    tuple
        (mse, r2) evaluation metrics
    """

    # -------------------------------
    # 1. Load trained model
    # -------------------------------
    if model_type == "linear":
        if not model_path.endswith(".pkl"):
            raise ValueError("Linear Regression model must be a .pkl file")
        model = joblib.load(model_path)  # Load sklearn model

    elif model_type == "dnn":
        if not model_path.endswith(".h5"):
            raise ValueError("DNN model must be a .h5 file")
        model = tf.keras.models.load_model(model_path)  # Load Keras model

    else:
        raise ValueError("model_type must be either 'linear' or 'dnn'")

    # -------------------------------
    # 2. Predict and evaluate
    # -------------------------------
    y_pred = model.predict(x_test)  # Predictions (not used directly here)
    mse, r2 = evaluate_model(model, x_test, y_test)

    # -------------------------------
    # 3. Print evaluation summary
    # -------------------------------
    print(f"\n--- Test Results ({model_type.upper()} Model) ---")
    print(f"Model Path: {model_path}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    return mse, r2
