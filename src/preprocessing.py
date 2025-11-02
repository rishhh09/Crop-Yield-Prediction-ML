import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Define which columns are which
# These are the features we will feed *into* the model
NUMERIC_FEATURES = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
CATEGORICAL_FEATURES = ['Crop', 'Season']

# NOTE: We are *not* including 'Area', 'Production', or 'State'.
# 'Area' and 'Production' are leaky, and 'State' is always the same.
# By *not* including them, our 'ColumnTransformer' will automatically drop them.

def create_preprocessor():
    """
    Creates the "translator" (ColumnTransformer) that will convert
    raw data into data ready for the model.
    """
    
    # 2. Create a "processing pipeline" for numeric features
    # This will fill missing values with the median, then scale them.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 3. Create a "processing pipeline" for categorical features
    # This will fill missing values with the most common, then one-hot encode.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. Bundle them together with ColumnTransformer
    # This applies the right transformer to the right column
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # This drops all columns we didn't specify (like 'Area', 'Production')
    )
    
    return preprocessor
