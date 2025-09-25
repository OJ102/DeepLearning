"""
Main training script for comparing Linear Regression (SGD) and Deep Neural Networks (DNNs)
on the cancer dataset.

Workflow:
---------
1. Preprocess data (scaling, encoding, log transforms).
2. Train multiple models with different learning rates:
   - Linear Regression (SGD-based)
   - DNNs with different hidden layer configurations
3. Evaluate each model on test data using MSE and R².
4. Save performance plots and log results to a text file.
5. Save the best-performing Linear Regression and DNN models.
"""

import preprocessor
import utility
import linearRegression
import DNN
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib   # for saving linear regression models
import sys
import os

# Fix randomness for reproducibility
tf.random.set_seed(42)

# Load and preprocess dataset
df = preprocessor.load_data('cancer_reg-1.csv')
df_processed = preprocessor.preprocess_data(df)

# Features and target split
X = df_processed.drop(columns=['TARGET_deathRate'])
y = df_processed['TARGET_deathRate']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Candidate learning rates
lrList = [0.1, 0.01, 0.001, 0.0001]
all_results = []

# Track best models across all runs
best_linear_score = -sys.maxsize
best_dnn_score = -sys.maxsize
best_linear_model = None
best_dnn_model = None

# Loop over learning rates
for lr in lrList:
    results = []
    epochs = 100

    # ============================
    # Linear Regression (SGD-based)
    # ============================
    model_name = 'Linear Regression'
    print(f"--- Training {model_name} with LR={lr} ---")

    linear_model = linearRegression.LinearRegressionModel(learning_rate=lr) 
    linear_model.train(X_train, y_train)

    mse, r2 = utility.evaluate_model(linear_model, X_test, y_test)
    try:
        utility.linear_model_performance_plot(
            y_test, linear_model.predict(X_test), model_name, lr
        )
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")

    results.append({'model': model_name, 'mse': mse, 'r2': r2})

    # Save best linear model
    if r2 > best_linear_score:
        best_linear_score = r2
        best_linear_model = linear_model

    # ============================
    # DNN Models with various depths
    # ============================

    # --- DNN with 1 hidden layer ---
    model_name = 'DNN-16'
    print(f"--- Training {model_name} with LR={lr} ---")

    DNN_model_1 = DNN.DNN(input_shape=(X_train.shape[1],), layers=[16], learning_rate=lr)
    history_1 = DNN_model_1.train(X_train, y_train, epochs=epochs)

    mse, r2 = utility.evaluate_model(DNN_model_1.get_model(), X_test, y_test)
    try:
        utility.linear_model_performance_plot(  # same scatter plot works here
            y_test, DNN_model_1.get_model().predict(X_test), model_name, lr
        )
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")

    results.append({'model': model_name, 'mse': mse, 'r2': r2})
    if r2 > best_dnn_score:
        best_dnn_score = r2
        best_dnn_model = DNN_model_1.get_model()

    # --- DNN with 2 hidden layers ---
    model_name = 'DNN-30-8'
    print(f"--- Training {model_name} with LR={lr} ---")

    DNN_model_2 = DNN.DNN(input_shape=(X_train.shape[1],), layers=[30, 8], learning_rate=lr)
    history_2 = DNN_model_2.train(X_train, y_train, epochs=epochs)

    mse, r2 = utility.evaluate_model(DNN_model_2.get_model(), X_test, y_test)
    try:
        utility.model_performance_loss_plot(history_2, model_name, lr)
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")

    results.append({'model': model_name, 'mse': mse, 'r2': r2})
    if r2 > best_dnn_score:
        best_dnn_score = r2
        best_dnn_model = DNN_model_2.get_model()

    # --- DNN with 3 hidden layers ---
    model_name = 'DNN-30-16-8'
    print(f"--- Training {model_name} with LR={lr} ---")

    DNN_model_3 = DNN.DNN(input_shape=(X_train.shape[1],), layers=[30, 16, 8], learning_rate=lr)
    history_3 = DNN_model_3.train(X_train, y_train, epochs=epochs)

    mse, r2 = utility.evaluate_model(DNN_model_3.get_model(), X_test, y_test)
    try:
        utility.model_performance_loss_plot(history_3, model_name, lr)
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")

    results.append({'model': model_name, 'mse': mse, 'r2': r2})
    if r2 > best_dnn_score:
        best_dnn_score = r2
        best_dnn_model = DNN_model_3.get_model()

    # --- DNN with 4 hidden layers ---
    model_name = 'DNN-30-16-8-4'
    print(f"--- Training {model_name} with LR={lr} ---")

    DNN_model_4 = DNN.DNN(input_shape=(X_train.shape[1],), layers=[30, 16, 8, 4], learning_rate=lr)
    history_4 = DNN_model_4.train(X_train, y_train, epochs=epochs)

    mse, r2 = utility.evaluate_model(DNN_model_4.get_model(), X_test, y_test)
    try:
        utility.linear_model_performance_plot(
            y_test, DNN_model_4.get_model().predict(X_test), model_name, lr
        )
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")

    results.append({'model': model_name, 'mse': mse, 'r2': r2})
    if r2 > best_dnn_score:
        best_dnn_score = r2
        best_dnn_model = DNN_model_4.get_model()

    # ============================
    # Print summary for current LR
    # ============================
    print("*" * 50)
    print("\nModel Performance Summary:")
    print("Learning Rate:", lr)
    print("Epochs:", epochs)

    for res in results:
        print(f"{res['model']}: MSE={res['mse']:.4f}, R2={res['r2']:.4f}")
    all_results.append((lr, epochs, results))


# ============================
# Save all results to file
# ============================
output_file = "model_performance_summary.txt"
with open(output_file, 'w') as f:
    for lrr, epochs, results in all_results:
        f.write("*" * 50 + "\n")
        f.write("\nModel Performance Summary:\n")
        f.write(f"Learning Rate: {lrr}\n")
        f.write(f"Epochs: {epochs}\n")
        
        for res in results:
            f.write(f"{res['model']}: MSE={res['mse']:.4f}, R2={res['r2']:.4f}\n")

print(f"\n--- Output successfully saved to '{output_file}' ---")


# ============================
# Save best models
# ============================
os.makedirs("saved_models", exist_ok=True)

if best_linear_model:
    joblib.dump(best_linear_model, "saved_models/best_linear_model.pkl")
    print("Best Linear Regression model saved as 'saved_models/best_linear_model.pkl'")

if best_dnn_model:
    best_dnn_model.save("saved_models/best_dnn_model.h5")
    print("Best DNN model saved as 'saved_models/best_dnn_model.h5'")


# ============================
# Log best models into summary
# ============================
with open(output_file, 'a') as f:
    if best_linear_model:
        f.write("\nBest Linear Regression model saved as 'saved_models/best_linear_model.pkl'\n")
        f.write(f"Best Linear Regression R2: {best_linear_score:.4f}\n")
    if best_dnn_model:
        f.write("\nBest DNN model saved as 'saved_models/best_dnn_model.h5'\n")
        f.write(f"Best DNN R2: {best_dnn_score:.4f}\n")
