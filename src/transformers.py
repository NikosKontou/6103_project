import yaml
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV
"""
Custom transformers for the Ames Housing regression project.

Includes:
- LassoFeatureSelector: Selects features based on LassoCV coefficients
- FeatureCreator: Generates new domain-informed features
"""
# Load project configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects features based on non-zero LassoCV coefficients.

    Parameters:
    - threshold (float): Minimum absolute coefficient to keep a feature
    - cv (int): Number of cross-validation folds for LassoCV
    - random_state (int): Random seed for reproducibility
    - max_iter (int): Maximum number of iterations for LassoCV

    Attributes:
    - selected_columns_ (Index): Features selected based on the threshold
    """
    def __init__(self, threshold=0.0, cv=5, random_state=42, max_iter=10000):
        self.threshold = threshold
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """
        Fits a LassoCV model and selects features with coefficients above threshold.

        Parameters:
        - X (pd.DataFrame): Input feature matrix
        - y (array-like): Target variable

        Returns:
        - self
        """
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)

        self.lasso_ = LassoCV(
            cv=self.cv,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.lasso_.fit(X, y)

        coefs = self.lasso_.coef_
        mask = (abs(coefs) > self.threshold)
        self.selected_columns_ = X.columns[mask]

        return self

    def transform(self, X):
        """
        Transforms input by keeping only features selected during fit.

        Parameters:
        - X (pd.DataFrame): Input feature matrix

        Returns:
        - pd.DataFrame: Subset of selected features
        """
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        return X.loc[:, self.selected_columns_]


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates domain-informed features for the Ames Housing dataset.
    Assumes missing values have already been handled.
    """
    def __init__(self, high_value_neighborhoods=None):
        if high_value_neighborhoods is None:
            high_value_neighborhoods = ['StoneBr', 'NridgHt', 'NoRidge']
        self.high_value_neighborhoods = high_value_neighborhoods

    def fit(self, X, y=None):
        """
        Fit does nothing (stateless transformer).

        Parameters:
        - X (pd.DataFrame): Input features
        - y (array-like, optional): Target variable

        Returns:
        - self
        """
        return self

    def transform(self, X):
        """
        Adds new engineered features based on domain logic.

        Parameters:
        - X (pd.DataFrame): Input features

        Returns:
        - pd.DataFrame: Feature matrix with new features added
        """
        X = X.copy()

        # Engineered features with fillna(0) to avoid errors
        X['Total_SF'] = (
            X['Gr_Liv_Area'].fillna(0) +
            X['Total_Bsmt_SF'].fillna(0) +
            X['Garage_Area'].fillna(0)
        )

        X['Porch_SF'] = (
            X['Open_Porch_SF'].fillna(0) +
            X['Enclosed_Porch'].fillna(0) +
            X['3Ssn_Porch'].fillna(0) +
            X['Screen_Porch'].fillna(0)
        )

        X['Remod_Age'] = X['Yr_Sold'] - X['Year_Remod_Add']
        
        X['Age'] = X['Yr_Sold'] - X['Year_Built']

        X['Is_Luxury_Home'] = ((X['Overall_Qual'] >= 9) & (X['Total_SF'] > 4000)).astype(int)

        X['High_Value_Neighborhood'] = X['Neighborhood'].isin(self.high_value_neighborhoods).astype(int)

        return X 
