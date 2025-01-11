import unittest
import mlflow
from mlflow.tracking import MlflowClient
import os
import pandas as pd

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


class TestModelLoading(unittest.TestCase):

    def test_model_in_staging(self):
        """
        Test if the model exists in the 'Staging' stage.
        """
        # Initialize the MLflow client to interact with the MLflow server
        client = MlflowClient()

        # Retrieve the latest versions of the model in the 'Staging' stage
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        # Assert that at least one version of the model exists in the 'Staging' stage
        self.assertGreater(len(versions), 0, "No model found in the 'Staging' stage.")

    def test_model_loading(self):
        """
        Test if the model can be loaded properly from the 'Staging' stage.
        """
        # Initialize the MLflow client to interact with the server
        client = MlflowClient()

        # Retrieve the latest versions of the model in the 'Staging' stage
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        # If no versions are found, fail the test and skip the model loading part
        if not versions:
            self.fail("No model found in the 'Staging' stage, skipping model loading test.")

        # Get the version details of the latest model in the 'Staging' stage
        latest_version = versions[0].version
        run_id = versions[0].run_id

        # Load the model from the 'Staging' stage
        model_uri = f"runs/{run_id}/{model_name}"
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model version {latest_version} loaded successfully from {model_uri}.")
        except Exception as e:
            # Fail the test if an exception occurs during model loading
            self.fail(f"Failed to load the model: {e}")

        # Assert that the loaded model is not None
        self.assertIsNotNone(loaded_model, "The loaded model is None.")
        print(f"Model successfully loaded from {model_uri}.")


if __name__ == "__main__":
    unittest.main()

