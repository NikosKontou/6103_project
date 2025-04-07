"""
Updated model training and evaluation module.
Now supports inverse log-transformation of predictions.
"""
import yaml
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def run_regression_and_evaluate(X_train, X_test, y_train, y_test, model, model_name="Model", 
                                inverse_log=False, transformer=None):
    """
    Train and evaluate a regression model using RMSE, MAE, and R².
    Optionally applies inverse log-transform to predictions and targets.
    Returns predictions and performance metrics.
    """
    # Fit model
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Inverse log transform if specified
    if inverse_log and transformer is not None:
        y_pred_train = transformer.inverse_transform(y_pred_train)
        y_pred_test = transformer.inverse_transform(y_pred_test)
        y_train = transformer.inverse_transform(y_train)
        y_test = transformer.inverse_transform(y_test)

    # Train metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Test metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    # Print main performance (Test only)
    print(f"\n📊 {model_name} Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")

    # Detailed comparison of Train vs Test
    print(f"\nDetailed Train/Test Metrics:")
    print(f"  🔹 Train → RMSE: {rmse_train:.4f} | MAE: {mae_train:.4f} | R²: {r2_train:.4f}")
    print(f"  🔸 Test  → RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # Plot predictions vs actual (Test)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"{model_name} Predictions (Test Set)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "model": model,
        "y_pred": y_pred_test,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }