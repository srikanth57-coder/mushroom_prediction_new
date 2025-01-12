import mlflow
from mlflow.tracking import MlflowClient
import os

# Ensure the DAGSHUB_TOKEN environment variable is set
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# Set MLFlow tracking credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Define the MLflow server URL and repository details
dagshub_url = "https://dagshub.com"
repo_owner = "srikanth57-coder"
repo_name = "mushroom_prediction_new"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
model_name = "Best Model"  # Update to the actual model name you're using


def promote_model_to_production():
        """promote the latest model in staging to productionand archive the current production model"""
        # Initialize the MLflow client to interact with the server
        client = MlflowClient()

        # Retrieve the latest versions of the model in the 'Staging' stage
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        # If no versions are found, fail the test and skip the model loading part
        if not versions:
            print("no model found in the'staging' stage.")
            return

        # Get the version details of the latest model in the 'Staging' stage
        latest_version = versions[0]
        version_number = latest_version.version

        # Get the current production model ,if any
        