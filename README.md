# Cancer Mortality Prediction

This project predicts **cancer mortality rates** using both **Linear Regression** and **Deep Neural Networks (DNNs)**.
It includes data preprocessing, model training, evaluation, and saving the **best performing models** for later testing.

---

## 📂 Project Structure

```
project/
│── preprocessor.py          # Data loading and preprocessing
│── linearRegression.py      # Linear regression model wrapper (SGDRegressor)
│── DNN.py                   # DNN model wrapper (Keras Sequential API)
│── utility.py               # Evaluation metrics and plotting functions
│── train_models.py          # Main script to train and save models
│── test_models.py           # Script to load and test saved models
│── cancer_reg-1.csv         # Dataset
│── saved_models/            # Folder where best models are saved
│   ├── best_linear_model.pkl
│   └── best_dnn_model.h5
│── plots/                   # Performance plots (generated automatically)
│── model_performance_summary.txt # Training summary results
│── README.md                # Project documentation
```

---

## ⚙️ Requirements

We are using an anaconda environment

Create the environment:

```bash
conda env create -f environment.yaml
```

Activate it:

```bash
conda activate dl-hw1
```

---

## 🚀 How to Run

### 1. Train Models

Run the training script to train Linear Regression and multiple DNNs with different learning rates.
The script will:

* Train models.
* Evaluate performance (MSE, R²).
* Save plots in `plots/`.
* Save the **best Linear Regression** and **best DNN** models in `saved_models/`.

```bash
python train_models.py
```

After training, you’ll see:

* `saved_models/best_linear_model.pkl`
* `saved_models/best_dnn_model.h5`
* `model_performance_summary.txt` (training summary).

---

### 2. Test Saved Models

Run the testing script to load the saved models and evaluate them on the **test set**.

```bash
python test_models.py
```

This will print metrics and (optionally) generate performance plots for predictions.

Example output:

```
--- Test Results (LINEAR Model) ---
Model Path: saved_models/best_linear_model.pkl
Mean Squared Error: 240.1287
R^2 Score: 0.6543

--- Test Results (DNN Model) ---
Model Path: saved_models/best_dnn_model.h5
Mean Squared Error: 198.5632
R^2 Score: 0.7012
```

---

## 📊 Results

* Both **Linear Regression** and **DNNs** are trained.
* Best performing models are automatically saved.
* Performance plots are stored in the `plots/` folder.

---