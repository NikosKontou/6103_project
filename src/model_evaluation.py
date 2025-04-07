import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.visualizations import plot_residuals, plot_predicted_vs_actual
'''
Evaluates a fitted regression model by computing RMSE, MAE, and R² on both log-transformed and original scales.
Optionally generates diagnostic plots for predicted vs actual values and residuals.
'''
def evaluate_model(name, model, X_train, X_test, y_train_log, y_test_log, plot=True):
    """
    Evaluates the performance of a fitted model on both log and original scales.

    Parameters:
    - name (str): Model name (used in plots and result dictionary)
    - model (estimator): A fitted scikit-learn model or pipeline
    - X_train (DataFrame): Training feature set (unused here but kept for consistency)
    - X_test (DataFrame): Testing feature set
    - y_train_log (Series or array): Log-transformed training targets (unused here)
    - y_test_log (Series or array): Log-transformed testing targets
    - plot (bool): Whether to plot predicted vs actual and residuals

    Returns:
    - dict: Evaluation metrics on log scale and original scale
    """
    # --- Predict on test ---
    y_pred_log = model.predict(X_test)
    y_test_actual = np.expm1(y_test_log)
    y_pred_actual = np.expm1(y_pred_log)

    # --- Predict on train ---
    y_train_pred_log = model.predict(X_train)
    y_train_actual = np.expm1(y_train_log)
    y_train_pred_actual = np.expm1(y_train_pred_log)

    # Metrics (test)
    # Evaluate prediction performance on the test set (log-transformed and original scale)

    test_metrics = {
        "RMSE (Log)": np.sqrt(mean_squared_error(y_test_log, y_pred_log)),
        "MAE (Log)": mean_absolute_error(y_test_log, y_pred_log),
        "R² (Log)": r2_score(y_test_log, y_pred_log),
        "RMSE ($)": np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)),
        "MAE ($)": mean_absolute_error(y_test_actual, y_pred_actual),
        "R² ($)": r2_score(y_test_actual, y_pred_actual)
    }

    # Metrics (train)
    # Evaluate prediction performance on the training set (log-transformed and original scale)
    train_metrics = {
        "RMSE (Log)": np.sqrt(mean_squared_error(y_train_log, y_train_pred_log)),
        "MAE (Log)": mean_absolute_error(y_train_log, y_train_pred_log),
        "R² (Log)": r2_score(y_train_log, y_train_pred_log),
        "RMSE ($)": np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual)),
        "MAE ($)": mean_absolute_error(y_train_actual, y_train_pred_actual),
        "R² ($)": r2_score(y_train_actual, y_train_pred_actual)
    }
    # Optional diagnostic plots
    if plot:
        plot_predicted_vs_actual(y_test_log, y_pred_log, model_name=name)
        plot_residuals(y_test_log, y_pred_log, model_name=name)

    return {
        "Model": name,
        "Train Metrics": train_metrics,
        "Test Metrics": test_metrics
    }
