import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load csv file
df = pd.read_csv('uttarakhand_crop_yield.csv')

# Normalize column names
df.columns = df.columns.str.strip()

# Basic preview
print(df.head())
print(df.info())
print("Columns:", df.columns.tolist())

# Identify target
target = 'Yield'
if target not in df.columns:
    raise KeyError(f"Expected target column '{target}' not found. Available: {df.columns.tolist()}")

# Convert numeric-like columns to numeric (safe)
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
# also try to coerce any numeric looking object columns
for col in df.columns:
    if col not in numeric_cols:
        coerced = pd.to_numeric(df[col], errors='coerce')
        if coerced.notna().sum() / len(df) > 0.5 and coerced.dtype != object:
            df[col] = coerced
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude target from features lists
numeric_features = [c for c in numeric_cols if c != target]

# Missing values summary
print("Missing values per column:\n", df.isnull().sum())

# Impute numeric columns with median
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", categorical_cols)

#clean categorical values (strip whitespace)
for c in categorical_cols:
    df[c] = df[c].astype(str).str.strip()
# recompute in case types changed
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns after strip:", categorical_cols)


# Feature engineering: create sensible ratios if base columns exist
if 'Fertilizer' in df.columns and 'Area' in df.columns:
    df['fertilizer_per_area'] = df['Fertilizer'] / (df['Area'] + 1)
if 'Pesticide' in df.columns and 'Area' in df.columns:
    df['pesticide_per_area'] = df['Pesticide'] / (df['Area'] + 1)
if 'Production' in df.columns and 'Area' in df.columns:
    df['production_per_area'] = df['Production'] / (df['Area'] + 1)

# Update numeric features list after new features
numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target]

# Quick EDA plots (numerical distributions)
df[numeric_features].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# Correlation heatmap for numeric columns (include target)
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_features + [target]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

# Scatter plots of numeric features vs target
for feature in numeric_features:
    if feature == target:
        continue
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[feature], y=df[target])
    plt.title(f'{feature} vs {target}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()

# One-hot encode categorical columns (if any)
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# convert boolean dummies to 0/1 so CSV is numeric
bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    df[bool_cols] = df[bool_cols].astype(int)

# scaling: only scale continuous numeric features (exclude target and binaries) 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
continuous_cols = [c for c in num_cols if c != target and df[c].nunique() > 2]
if continuous_cols:
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# Save processed dataset
df.to_csv('processed_dataset.csv', index=False)
print("Saved processed_dataset.csv with shape:", df.shape)