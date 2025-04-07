"""
Feature selection module using LassoCV.
Ensures selection is done ONLY on training data.
"""
import yaml
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

# Load config.yaml to get the random seed
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Get the random seed from config.yaml
random_seed = config.get('random_seed', 42)

# Set the random seed for reproducibility
np.random.seed(random_seed)
random.seed(random_seed)


def run_lasso_feature_selection(X_train, y_train, feature_names, name="Train Set"):
    """
    Run LassoCV on the training set only and return selected features.
    
    Parameters:
        X_train (np.ndarray): Preprocessed training features
        y_train (np.ndarray): Log-transformed training target
        feature_names (List[str] or np.ndarray): Names of all input features
        name (str): Optional label for printing

    Returns:
        pd.DataFrame: DataFrame with selected feature names and their coefficients
    """
    print(f"🔍 Running LassoCV on {name}...")

    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)

    selected_indices = np.where(lasso.coef_ != 0)[0]
    selected_features = np.array(feature_names)[selected_indices]
    selected_coefs = lasso.coef_[selected_indices]

    print(f"✅ Selected {len(selected_features)} features from {name}.")

    return pd.DataFrame({'Feature': selected_features, 'Coefficient': selected_coefs})
