import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Evaluation tools for the Ames Housing Regression project.

Includes plotting functions for:
- Lasso feature coefficients
- Predicted vs actual values
- Residual analysis
- Feature importance
"""
def plot_lasso_selected_features(selected_df, threshold=0.005, plot=True):
    """
    Plots the coefficients of features selected by Lasso.

    Parameters:
    - selected_df (pd.DataFrame): Must contain 'Feature' and 'Coefficient' columns
    - threshold (float): Lasso threshold value for context (used in plot title)
    - plot (bool): If False, suppresses plotting

    """
    if not plot:
        return
    
    # Sort by absolute coefficient magnitude
    sorted_df = selected_df.reindex(selected_df["Coefficient"].abs().sort_values(ascending=False).index)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.barh(sorted_df["Feature"], sorted_df["Coefficient"])
    plt.xlabel("Lasso Coefficient")
    plt.title(f"Top Features Selected by Lasso (Threshold = {threshold})")
    plt.gca().invert_yaxis()
    plt.tight_layout()

def plot_predicted_vs_actual(y_true_log, y_pred_log, model_name="Model"):
    """
    Plots predicted vs. actual sale prices on original (exponential) scale.

    Parameters:
    - y_true_log (np.array or pd.Series): Log-transformed actual values
    - y_pred_log (np.array or pd.Series): Log-transformed predicted values
    - model_name (str): Name of the model (used in plot title)
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             color='red', linestyle='--')
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"{model_name} Predictions (Original Scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test_log, y_pred_log, model_name="Model"):
    """
    Plots residuals (Predicted - Actual) in the original dollar scale.

    Parameters:
    - y_test_log (np.array or pd.Series): Log-transformed actual values
    - y_pred_log (np.array or pd.Series): Log-transformed predicted values
    - model_name (str): Name of the model (used in plot title)
    """
    if y_test_log is None or y_pred_log is None:
        print(f"Error: Received None value for residuals data in {model_name}")
        return

    y_test = np.expm1(y_test_log)  # Convert back to original scale
    y_pred = np.expm1(y_pred_log)  # Convert back to original scale

    # Plot the residuals
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=(y_pred - y_test), alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Residuals (Predicted - Actual)")
    plt.title(f"{model_name} Residuals (Original Scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




