import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# ---------- Paths ----------
# Make sure data folder exists
os.makedirs("data", exist_ok=True)
RAW_DATA_PATH = os.path.join("data", "uttarakhand_crop_yield.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed_dataset.csv")

# ---------- Load csv file ----------
df = pd.read_csv(RAW_DATA_PATH)

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
for col in df.columns:
    if col not in numeric_cols:
        coerced = pd.to_numeric(df[col], errors='coerce')
        if coerced.notna().sum() / len(df) > 0.5 and coerced.dtype != object:
            df[col] = coerced
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude target from features list
numeric_features = [c for c in numeric_cols if c != target]

# Missing values summary
print("Missing values per column:\n", df.isnull().sum())

# Impute numeric columns with median
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Handle categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", categorical_cols)

# Clean categorical values (strip whitespace)
for c in categorical_cols:
    df[c] = df[c].astype(str).str.strip()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns after strip:", categorical_cols)

# ---------- Feature Engineering ----------
if 'Fertilizer' in df.columns and 'Area' in df.columns:
    df['fertilizer_per_area'] = df['Fertilizer'] / (df['Area'] + 1)
if 'Pesticide' in df.columns and 'Area' in df.columns:
    df['pesticide_per_area'] = df['Pesticide'] / (df['Area'] + 1)
if 'Production' in df.columns and 'Area' in df.columns:
    df['production_per_area'] = df['Production'] / (df['Area'] + 1)

# Update numeric features list
numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target]

# ---------- Quick EDA Plots ----------
df[numeric_features].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_features + [target]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

for feature in numeric_features:
    if feature == target:
        continue
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[feature], y=df[target])
    plt.title(f'{feature} vs {target}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()

# ---------- Encoding ----------
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Convert boolean dummies to 0/1
bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    df[bool_cols] = df[bool_cols].astype(int)

# ---------- Scaling ----------
scaler = StandardScaler()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
continuous_cols = [c for c in num_cols if c != target and df[c].nunique() > 2]
if continuous_cols:
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# ---------- Save processed dataset ----------
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"✅ Saved processed dataset to {PROCESSED_DATA_PATH} with shape: {df.shape}")
