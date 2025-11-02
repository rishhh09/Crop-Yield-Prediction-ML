import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Define the path to our trained pipeline
MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "final_pipeline.pkl")

@st.cache_resource  # <-- This is a magic Streamlit command!
def load_pipeline():
    """Loads the saved pipeline file only once."""
    print("Loading pipeline...") # This will only print the first time the app runs
    pipeline = joblib.load(PIPELINE_PATH)
    print("Pipeline loaded successfully.")
    return pipeline

# Load the pipeline
pipeline = load_pipeline()

print("App is running...")

# --- 1. Set up the Page Title ---
st.title('🌱 Uttarakhand Crop Yield Prediction')

# --- 2. Set up the Sidebar ---
st.sidebar.header('Enter Farm Conditions:')

# --- 3. Get User Inputs from the Sidebar ---

# (Keep the code for loading df_raw and getting all_crops/all_seasons)
df_raw = pd.read_csv('data/uttarakhand_crop_yield.csv')
df_raw.columns = df_raw.columns.str.strip()
all_crops = sorted(df_raw['Crop'].unique())
all_seasons = sorted(df_raw['Season'].unique())
# ---

# Create dropdowns for categorical features
in_crop = st.sidebar.selectbox('Select Crop:', all_crops)
in_season = st.sidebar.selectbox('Select Season:', all_seasons)
in_year = st.sidebar.number_input('Enter Crop Year:', min_value=2010, max_value=2030, value=2025)

# --- THIS IS THE NEW PART ---
# We add Area and labels for "per hectare"
in_area = st.sidebar.number_input('Enter Area (in hectares):', min_value=1.0, value=1.0)
in_rainfall = st.sidebar.number_input('Enter Annual Rainfall (in mm):', min_value=0.0, value=1400.0)
in_fertilizer_ph = st.sidebar.number_input('Enter Fertilizer (in kg/hectare):', min_value=0.0, value=100.0) 
in_pesticide_ph = st.sidebar.number_input('Enter Pesticide (in kg/hectare):', min_value=0.0, value=50.0)
# --- END OF NEW PART ---

predict_button = st.sidebar.button('Predict Yield')

# --- 4. Prediction Logic ---
if predict_button:
    
    # --- THIS IS THE NEW PART ---
    # We do the "secret math" to get the TOTALS
    total_fertilizer = in_fertilizer_ph * in_area
    total_pesticide = in_pesticide_ph * in_area
    
    # The new pipeline's preprocessor looks for the *original* column names
    raw_input_data = {
        'Crop': [in_crop],
        'Season': [in_season],
        'Crop_Year': [in_year],
        'Annual_Rainfall': [in_rainfall],
        'Area': [in_area],                 # It needs Area
        'Fertilizer': [total_fertilizer],  # It needs the TOTAL Fertilizer
        'Pesticide': [total_pesticide],    # It needs the TOTAL Pesticide
        'State': ['Uttarakhand']           # Add this for safety
    }
    # --- END OF NEW PART ---

    # This part is the same as before
    input_df = pd.DataFrame(raw_input_data)
    prediction = pipeline.predict(input_df)

    st.success(f"Prediction successful for {in_area} hectare(s):")
    st.header(f"Predicted Yield: {prediction[0]:.2f} tons per hectare")