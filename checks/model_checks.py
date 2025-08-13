from mlflow.tracking import MlflowClient

client = MlflowClient()
models = client.search_registered_models()
for m in models:
    print(f"Model name: {m.name}")
    for v in m.latest_versions:
        print(f"  - version: {v.version}, stage: {v.current_stage}")
