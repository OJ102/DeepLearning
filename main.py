import preprocessor
import utility
import linearRegression
import DNN
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys


tf.random.set_seed(42)
preprocessor.load_data('cancer_reg-1.csv')
df = preprocessor.load_data('cancer_reg-1.csv')
# Assuming preprocess_data handles the log transformation of the target
df_processed = preprocessor.preprocess_data(df)

X = df_processed.drop(columns=['TARGET_deathRate'])
y = df_processed['TARGET_deathRate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lrList = [0.1, 0.01, 0.001, 0.0001]
all_results = []

# Loop over different learning rates
for lr in lrList:
    results = []
    epochs = 100
    
    # --- Linear Regression ---
    model_name = 'Linear Regression'
    print(f"--- Training {model_name} with LR={lr} ---")
    linear_model = linearRegression.LinearRegressionModel(learning_rate=lr) 
    linear_model.train(X_train, y_train)
    mse, r2 = utility.evaluate_model(linear_model, X_test, y_test)
    try:
        utility.linear_model_performance_plot(y_test, linear_model.predict(X_test), model_name)
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")
    results.append({'model': model_name, 'mse': mse, 'r2': r2})


    # --- DNN with 1 hidden layer (DNN-16) ---
    model_name = 'DNN-16'
    print(f"--- Training {model_name} with LR={lr} ---")
    DNN_model_1 = DNN.DNN(input_shape=(X_train.shape[1],), layers=[16], learning_rate=lr)
    DNN_model_1.train(X_train, y_train, epochs=epochs)
    mse, r2 = utility.evaluate_model(DNN_model_1.get_model(), X_test, y_test)
    try:
        utility.model_performance_plot(y_test, DNN_model_1.get_model().predict(X_test), model_name)
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")

    results.append({'model': model_name, 'mse': mse, 'r2': r2})


    # --- DNN with 2 hidden layers (DNN-30-8) ---
    model_name = 'DNN-30-8'
    print(f"--- Training {model_name} with LR={lr} ---")
    DNN_model_2 = DNN.DNN(input_shape=(X_train.shape[1],), layers=[30,8], learning_rate=lr)
    DNN_model_2.train(X_train, y_train, epochs=epochs)
    mse, r2 = utility.evaluate_model(DNN_model_2.get_model(), X_test, y_test)
    try:
        utility.model_performance_loss_plot(DNN_model_2.get_model().history, model_name)
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")
    results.append({'model': model_name, 'mse': mse, 'r2': r2})


    # --- DNN with 3 hidden layers (DNN-30-16-8) ---
    model_name = 'DNN-30-16-8'
    print(f"--- Training {model_name} with LR={lr} ---")
    DNN_model_3 = DNN.DNN(input_shape=(X_train.shape[1],), layers=[30,16,8], learning_rate=lr)
    DNN_model_3.train(X_train, y_train, epochs=epochs)
    mse, r2 = utility.evaluate_model(DNN_model_3.get_model(), X_test, y_test)
    try:
        utility.model_performance_loss_plot(DNN_model_3.get_model().history, model_name)
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")
    results.append({'model': model_name, 'mse': mse, 'r2': r2})

    # --- DNN with 4 hidden layers (DNN-30-16-8-4) ---
    model_name = 'DNN-30-16-8-4'
    print(f"--- Training {model_name} with LR={lr} ---")
    DNN_model_4 = DNN.DNN(input_shape=(X_train.shape[1],), layers=[30,16,8,4], learning_rate=lr)
    DNN_model_4.train(X_train, y_train, epochs=epochs)
    mse, r2 = utility.evaluate_model(DNN_model_4.get_model(), X_test, y_test)
    try:
        utility.model_performance_plot(y_test, DNN_model_4.get_model().predict(X_test), model_name)
    except Exception as e:
        print(f"Error generating performance plot for {model_name}: {e}")
    results.append({'model': model_name, 'mse': mse, 'r2': r2})


    # Print all results for the current LR
    print("*"*50)
    print("\nModel Performance Summary:")
    print("Learning Rate:", lr)
    print("Epochs:", epochs)

    for res in results:
        print(f"{res['model']}: MSE={res['mse']:.4f}, R2={res['r2']:.4f}")
    all_results.append((lr, epochs, results))

# Define the file name where you want to save the output
output_file = "model_performance_summary.txt"

# Open the file for writing ('w')
with open(output_file, 'w') as f:
    
    # Loop through the results and write directly to the file object 'f'
    for lrr, epochs, results in all_results:
        f.write("*"*50 + "\n")
        f.write("\nModel Performance Summary:\n")
        f.write(f"Learning Rate: {lrr}\n")
        f.write(f"Epochs: {epochs}\n")
        
        for res in results:
            f.write(f"{res['model']}: MSE={res['mse']:.4f}, R2={res['r2']:.4f}\n")

print(f"\n--- Output successfully saved to '{output_file}' ---")