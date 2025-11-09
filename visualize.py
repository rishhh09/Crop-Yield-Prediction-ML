import pandas as pd
import joblib                           #To load .pkl file
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
import os
print("All libraries imported successfully.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# --- 1. Define File Paths ---
PIPELINE_PATH = 'models/final_pipeline.pkl'
DATA_PATH = 'data/uttarakhand_crop_yield.csv'

# --- 2. Load Pipeline and Data ---
print(f"Loading pipeline from {PIPELINE_PATH}...")
pipeline = joblib.load(PIPELINE_PATH)

print(f"Loading raw data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

print("Files loaded.")

# --- 3. Prepare and Split Data ---
# We must clean column names just like the training script did
df.columns = df.columns.str.strip()

# Define the target variable
TARGET_COL = 'Yield'

# Separate our features (X) from our target (y)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# THIS IS THE MOST IMPORTANT LINE
# We use the *exact same* test_size and random_state as the training script
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data split into {len(X_test)} test samples.")

# --- 4. Make Predictions ---
print("Making predictions on the test set...")
# The pipeline handles ALL preprocessing and prediction in one step
y_pred = pipeline.predict(X_test)

print("Predictions complete.")

# --- 5. Visualize the Results ---
print("Creating visualization...")

plt.figure(figsize=(10, 6))

# Create the main scatter plot
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, s=50)

# Add the "perfect prediction" line
# We get the min/max of all values to draw a perfect diagonal line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) # 'r--' is red dashed line

# Add labels and title
plt.title('Actual Yield vs. Predicted Yield', fontsize=16)
plt.xlabel('Actual Yield (tons per hectare)', fontsize=12)
plt.ylabel('Predicted Yield (tons per hectare)', fontsize=12)
plt.grid(True)

# Save the plot to a file
plt.savefig('actual_vs_predicted.png')

print("Plot saved as 'actual_vs_predicted.png'")

# Show the plot
plt.show()