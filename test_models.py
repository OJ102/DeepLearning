"""
Script to test pre-trained models (Linear Regression and DNN)
on the cancer mortality dataset.

Steps:
1. Load and preprocess the dataset using custom preprocessor module.
2. Split dataset into training and test sets (same split as training).
3. Load saved models (best linear regression and best DNN).
4. Evaluate each model on the test set and print metrics.
"""

import preprocessor
import utility
from sklearn.model_selection import train_test_split
import tensorflow as tf

# -------------------------------
# 1. Fix randomness for reproducibility
# -------------------------------
tf.random.set_seed(42)

# -------------------------------
# 2. Load and preprocess dataset
# -------------------------------
df = preprocessor.load_data('cancer_reg-1.csv')           # Load dataset
df_processed = preprocessor.preprocess_data(df)           # Preprocess features

# -------------------------------
# 3. Split dataset into features and target
# -------------------------------
X = df_processed.drop(columns=['TARGET_deathRate'])       # Features
y = df_processed['TARGET_deathRate']                      # Target

# Ensure the same split ratio as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Test saved models
# -------------------------------

# Evaluate best Linear Regression model
utility.test_model(
    model_type='linear',
    model_path='saved_models/best_linear_model.pkl',
    x_test=X_test,
    y_test=y_test
)

# Evaluate best DNN model
utility.test_model(
    model_type='dnn',
    model_path='saved_models/best_dnn_model.h5',
    x_test=X_test,
    y_test=y_test
)
