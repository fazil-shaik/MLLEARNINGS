import mlflow

print("Tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run():
    print("Hello MLflow")