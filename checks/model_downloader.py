import mlflow

model_uri = "models:/fraud_detection_model/4"

local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

print(f"Model downloaded to: {local_path}")