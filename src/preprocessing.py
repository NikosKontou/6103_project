import json
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
"""
Preprocessing pipeline for the Ames Housing Dataset.
- IQR-based outlier removal
- Builds a safe ColumnTransformer
- Handles ordinal, nominal, and numeric features
"""

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Removes outliers based on the IQR method.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - columns (list of str): Column names to check for outliers
    - factor (float): IQR multiplier for bounds (default=1.5)

    Returns:
    - pd.DataFrame: Filtered DataFrame with outliers removed
    """
    initial_shape = df.shape[0]

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    final_shape = df.shape[0]
    print(f"Outlier removal: {initial_shape - final_shape} rows removed.")
    return df

def load_config_and_extract_features(df, config_path, extra_nominal=None, target_col='SalePrice'):
    """
    Loads a configuration file to extract and classify features from a DataFrame
    into ordinal, nominal, and numeric types based on data types and domain rules.

    Automatically identifies:
    - Ordinal features and their category orderings from a config file
    - Nominal features based on data type, excluding known ordinal features
    - Numeric features by dtype, excluding the target variable

    Optionally includes manually specified nominal integer features.

    Useful for preprocessing steps where feature categorization is critical
    (e.g., encoding strategies or pipeline design).
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    ordinal_map = config.get("ordinal_map", {})
    ordinal_features = list(ordinal_map.keys())
    ordinal_categories = list(ordinal_map.values())

    # Auto-detect nominal features
    nominal_features = df.select_dtypes(include=['object', 'category']) \
                         .columns.difference(ordinal_features).tolist()

    # Optionally add some manually known nominal int features
    if extra_nominal:
        for col in extra_nominal:
            if col in df.columns and col not in nominal_features:
                nominal_features.append(col)

    # Numeric (excluding target)
    numeric_features = df.select_dtypes(include='number').columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)

    return {
        "ordinal_map": ordinal_map,
        "ordinal_features": ordinal_features,
        "ordinal_categories": ordinal_categories,
        "nominal_features": nominal_features,
        "numeric_features": numeric_features
    }

def build_preprocessor(nominal_features, ordinal_features, ordinal_categories, num_cols, df_columns, target_col='SalePrice'):
    """
    Builds and returns a ColumnTransformer with pipelines for nominal, ordinal, and numeric features.

    Parameters:
    - nominal_features (list of str): Nominal categorical feature names
    - ordinal_features (list of str): Ordinal categorical feature names
    - ordinal_categories (list of list): Ordered category lists for each ordinal feature
    - num_cols (list of str): All numeric columns (may include SalePrice)
    - df_columns (Index or list): All column names in the DataFrame
    - target_col (str): Name of the target column to exclude from preprocessing

    Returns:
    - ColumnTransformer: Preprocessing pipeline with proper encoders and scalers
    """
    # Defensive filtering based on what's actually in the dataset
    nominal_features = [col for col in nominal_features if col in df_columns]
    filtered_ordinal = [(col, cat) for col, cat in zip(ordinal_features, ordinal_categories) if col in df_columns]

    if filtered_ordinal:
        ordinal_features, ordinal_categories = zip(*filtered_ordinal)
        ordinal_features = list(ordinal_features)
        ordinal_categories = list(ordinal_categories)
    else:
        ordinal_features, ordinal_categories = [], []

    # Filter numeric features to exclude any that are also nominal, ordinal, or the target
    excluded_cols = set(nominal_features) | set(ordinal_features) | {target_col}
    true_numeric = [col for col in num_cols if col in df_columns and col not in excluded_cols]

    # Pipelines
    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine all transformers
    transformers = []
    if nominal_features:
        transformers.append(('nom', nominal_pipeline, nominal_features))
    if ordinal_features:
        transformers.append(('ord', ordinal_pipeline, ordinal_features))
    if true_numeric:
        transformers.append(('num', numerical_pipeline, true_numeric))

    # Final preprocessor
    preprocessor = ColumnTransformer(transformers)

    return preprocessor





