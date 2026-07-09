import mlflow

# Connect to tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create (or use existing) experiment
mlflow.set_experiment("MLflow Beginner Course")

with mlflow.start_run(run_name="Second Run"):

    # Parameters
    mlflow.log_param("learning_rate", 0.02)
    mlflow.log_param("epochs", 100)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("optimizer", "SGD")

    # Metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.22)
    mlflow.log_metric("precision", 0.94)
    mlflow.log_metric("recall", 0.91)

    # Tags
    mlflow.set_tag("developer", "Shaik Fazil")
    mlflow.set_tag("course", "MLflow Mastery")
    mlflow.set_tag("environment", "Local")

    # Artifact
    mlflow.log_artifact("notes.txt")

print("Experiment Logged Successfully!")