from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        mse = mean_squared_error(y_test, y_pred)
    except Exception as e:
        print(f"Error calculating MSE: {e}")
        mse = float('nan')
    try:
        r2 = r2_score(y_test, y_pred)
    except Exception as e:
        print(f"Error calculating R^2: {e}")
        r2 = float('nan')
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    return mse, r2

def linear_model_performance_plot(y_test, y_pred, model_name):

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'{model_name} Predictions vs Actual - {datetime.now()}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(f'plots/{model_name}_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

def model_performance_loss_plot(history, model_name):
    print(f"TimeStamp: {datetime.now()}")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss Over Epochs - {datetime.now()}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{model_name}_loss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
