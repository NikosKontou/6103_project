
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV
from src.transformers import LassoFeatureSelector, FeatureCreator
"""
Centralized definition of regression model pipelines used in the Ames Housing project.
Each model includes preprocessing, optional feature selection, and optional hyperparameter tuning.
"""
def get_model_searches(preprocessor):
    """
    Returns a dictionary of regression models with or without hyperparameter tuning.

    Parameters:
    - preprocessor (Pipeline or ColumnTransformer): Preprocessing steps (encoding, scaling, etc.)

    Returns:
    - dict: Model name → pipeline or GridSearchCV-wrapped estimator
    """
    model_searches = {}

    # Baseline: Linear Regression (no regularization)
    linear_pipeline = Pipeline([
        ("feature_creator", FeatureCreator()),
        ("preprocessing", preprocessor),
        #("lasso_feature_selector", LassoFeatureSelector()),# Removed after testing — excluded domain features and reduced model performance
        ("model", LinearRegression())
    ])

    # Linear Regression (no regularization)
    model_searches["Linear Regression"] = linear_pipeline  # not wrapped in GridSearchCV

    # Polynomial Regression (no regularization)
    poly_pipeline = Pipeline([
        ("feature_creator", FeatureCreator()),
        ("preprocessing", preprocessor),
        #("lasso_feature_selector", LassoFeatureSelector()), # Removed after testing — excluded domain features and reduced model performance
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()) 
    ])

    poly_param_grid = {
        #"lasso_feature_selector__threshold": [0.0, 0.001, 0.01],
        "poly__degree": [1, 2]
    }
    model_searches["Polynomial Regression"] = GridSearchCV(
        poly_pipeline, poly_param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
    )

    # Untuned Ridge Regression
    ridge_untuned_pipeline = Pipeline([
        ("feature_creator", FeatureCreator()),
        ("preprocessing", preprocessor),
        #("lasso_feature_selector", LassoFeatureSelector()), # Removed after testing — excluded domain features and reduced model performance
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42))  # Fixed alpha (no tuning)
    ])
    model_searches["Untuned Ridge"] = ridge_untuned_pipeline

    # Tuned Ridge Regression (Linear Regression + L2 Regularization)
    ridge_pipeline = Pipeline([
        ("feature_creator", FeatureCreator()),
        ("preprocessing", preprocessor),
        #("lasso_feature_selector", LassoFeatureSelector()), # Removed after testing — excluded domain features and reduced model performance
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=42))
    ])

    ridge_param_grid = {
        "poly__degree": [1, 2],
        #"lasso_feature_selector__threshold": [0.0, 0.001, 0.01],
        "model__alpha": np.logspace(-2, 3, 20)  # 0.01 to 1000
    }   
    model_searches["Tuned Ridge"] = GridSearchCV(
        ridge_pipeline, ridge_param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
    )

    # Untuned Lasso Regression
    lasso_untuned_pipeline = Pipeline([
        ("feature_creator", FeatureCreator()),
        ("preprocessing", preprocessor),
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=1.0, random_state=42, max_iter=50000))  # Fixed alpha (no tuning)
    ])
    model_searches["Untuned Lasso"] = lasso_untuned_pipeline

    # Tuned Lasso Regression (Linear Regression + L1 Regularization)
    lasso_pipeline = Pipeline([
        ("feature_creator", FeatureCreator()),
        ("preprocessing", preprocessor),
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Lasso(random_state=42, max_iter=50000)) 
    ])

    lasso_param_grid = {
        "poly__degree": [1],
        "model__alpha": np.logspace(-4, 0, 20)  # 0.0001 to 1
    }
    model_searches["Tuned Lasso"] = GridSearchCV(
        lasso_pipeline, lasso_param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
    )

    return model_searches
