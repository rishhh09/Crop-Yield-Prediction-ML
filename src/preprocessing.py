import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Step 1: Create the Custom Transformer ---
# This class will create our "per hectare" features
# and drop the old "total" features.
class PerHectareTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create fertilizer_per_hectare and 
    pesticide_per_hectare features, then drop the originals.
    """
    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X, y=None):
        # Make a copy to avoid changing the original data
        X_ = X.copy()
        
        # We must have the 'Area' column to do this
        if 'Area' not in X_.columns:
            raise KeyError("The 'Area' column is missing, cannot calculate per-hectare features.")
            
        # Create new features. 
        # We replace 0 area with a tiny number (1e-6) to avoid division by zero.
        X_['fertilizer_per_hectare'] = X_['Fertilizer'] / X_['Area'].replace(0, 1e-6)
        X_['pesticide_per_hectare'] = X_['Pesticide'] / X_['Area'].replace(0, 1e-6)
        
        # Drop the original 'total' columns and 'Area'
        # 'State' is also dropped as it's not a useful feature (all Uttarakhand)
        columns_to_drop = ['Area', 'Fertilizer', 'Pesticide', 'State']
        
        # We drop them only if they exist, safely
        X_ = X_.drop(columns=[col for col in columns_to_drop if col in X_.columns])
        
        return X_

# --- Step 2: Define the Preprocessing Pipeline ---
# This function creates the full "translator"
def create_preprocessor():
    """
    Creates the full preprocessing pipeline.
    
    1. Runs the PerHectareTransformer.
    2. Applies scaling (to new numeric features) and one-hot encoding (to categories).
    """
    
    # Define which features go where *after* the custom transform
    # Note: 'Fertilizer', 'Pesticide', 'Area' are gone
    numeric_features = ['Crop_Year', 'Annual_Rainfall', 'fertilizer_per_hectare', 'pesticide_per_hectare']
    categorical_features = ['Crop', 'Season']

    # Create the standard transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create the main preprocessor that applies transformers
    column_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' 
    )
    
    # This is the MASTER preprocessing pipeline:
    # Step 1: Create 'per_hectare' features, drop old ones
    # Step 2: Apply scaling/encoding to the new, clean features
    full_preprocessor_pipeline = Pipeline(steps=[
        ('create_per_hectare_features', PerHectareTransformer()),
        ('process_and_scale', column_preprocessor)
    ])
    
    return full_preprocessor_pipeline