from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# model_uri = "models:/fraud_detection_model/4"
# model = mlflow.pyfunc.load_model(model_uri)
model = mlflow.pyfunc.load_model("./models/artifacts")

class FraudFeatures(BaseModel):
    Hour: float
    Day: float
    Boundary: float
    Suspicious_car_rental: float
    Suspicious_fuel: float
    Cumulative_type_percent: float
    Cumulative_Unique_Locations: float
    Days_since_last: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(features: FraudFeatures):
    try:
        input_df = pd.DataFrame([features.dict()])
        prediction = model.predict(input_df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

