from fastapi import FastAPI
import pickle
import pandas as pd
import mlflow
from pydantic import BaseModel
from fastapi import HTTPException

# FastAPI instance
app = FastAPI(
    title="Mushroom Prediction",
    description="An API to predict the mushroom type"
)

# Set the MLflow tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "srikanth57-coder"
repo_name = "mushroom_prediction_new"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Load the latest model from MLflow
def load_model():
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("Best Model", stages=["Production"])
        if not versions:
            raise Exception("No production version of the model found.")
        run_id = versions[0].run_id
        return mlflow.pyfunc.load_model(f"runs:/{run_id}/Best Model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load the model at the start
model = load_model()

# Pydantic model for mushroom data
class Mushroom(BaseModel):
    cap_shape: float
    cap_surface: float
    cap_color: float
    bruises: float
    odor: float
    gill_attachment: float
    gill_spacing: float
    gill_size: float
    gill_color: float
    stalk_shape: float
    stalk_root: float
    stalk_surface_above_ring: float
    stalk_surface_below_ring: float
    stalk_color_above_ring: float
    stalk_color_below_ring: float
    veil_type: float
    veil_color: float
    ring_number: float
    ring_type: float
    spore_print_color: float
    population: float
    habitat: float

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to the Mushroom Prediction API"}

# Prediction endpoint
@app.post("/predict")
def predict(mushroom: Mushroom):
    try:
        # Prepare the data for prediction
        mushroom_data = mushroom.dict()
        
        # Convert the data to DataFrame and rename columns to match model expectations
        df = pd.DataFrame([mushroom_data])
        column_mapping = {
            'cap_shape': 'cap-shape',
            'cap_surface': 'cap-surface',
            'cap_color': 'cap-color',
            'bruises': 'bruises',
            'odor': 'odor',
            'gill_attachment': 'gill-attachment',
            'gill_spacing': 'gill-spacing',
            'gill_size': 'gill-size',
            'gill_color': 'gill-color',
            'stalk_shape': 'stalk-shape',
            'stalk_root': 'stalk-root',
            'stalk_surface_above_ring': 'stalk-surface-above-ring',
            'stalk_surface_below_ring': 'stalk-surface-below-ring',
            'stalk_color_above_ring': 'stalk-color-above-ring',
            'stalk_color_below_ring': 'stalk-color-below-ring',
            'veil_type': 'veil-type',
            'veil_color': 'veil-color',
            'ring_number': 'ring-number',
            'ring_type': 'ring-type',
            'spore_print_color': 'spore-print-color',
            'population': 'population',
            'habitat': 'habitat'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Make prediction using the model
        prediction = model.predict(df)

        # Returning the prediction (assuming it returns a numeric value or class)
        return {"prediction": prediction.tolist()}  # Convert prediction to list for easy serialization
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
