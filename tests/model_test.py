import unittest
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tenacity import retry, stop_after_attempt, wait_fixed

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

import mlflow
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(10))
def load_model_with_retry(model_uri):
    return mlflow.pyfunc.load_model(model_uri)



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

        # Correct URI for loading the model from the Staging stage
        model_uri = f"runs:/{run_id}/{model_name}"  # Correct format for staging alias
        try:
            # Using the retry function to load the model with retries
            loaded_model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            # Fail the test if an exception occurs during model loading
            self.fail(f"Failed to load the model after retries: {e}")

        # Assert that the loaded model is not None
        self.assertIsNotNone(loaded_model, "The loaded model is None.")
        print(f"Model successfully loaded from {model_uri}.")
    def test_model_performance(self):
        """Test the performance of the model on the test data."""
        # Initialize the MLflow client to interact with the server
        client = MlflowClient()
        # Retrieve the latest versions of the model in the 'Staging' stage
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        # If no versions are found, fail the test and skip the model loading part
        if not versions:
            self.fail("No model found in the 'Staging' stage, skipping model loading test.")

        # Get the version details of the latest model in the 'Staging' stage
        run_id = versions[0].run_id

        # Correct URI for loading the model from the Staging stage
        model_uri = f"runs:/{run_id}/{model_name}"

        loaded_model = mlflow.pyfunc.load_model(model_uri)

        test_data_path = "./data/processed/test_processed_scaler.csv"

        # Ensure test data exists
        if not os.path.exists(test_data_path):
            self.fail(f"Test data not found at {test_data_path}")

        # Load the test data
        test_data = pd.read_csv(test_data_path)

        x_test = test_data.drop(columns=['class'], axis=1)
        y_test = test_data['class']

        predictions = loaded_model.predict(x_test)

          # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="binary")
        recall = recall_score(y_test, predictions, average="binary")
        f1 = f1_score(y_test, predictions, average="binary")

        # Print metrics
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Assert metrics meet thresholds
        self.assertGreaterEqual(accuracy, 0.3, "Accuracy is below the threshold.")
        self.assertGreaterEqual(precision, 0.3, "Precision is below the threshold.")
        self.assertGreaterEqual(recall, 0.3, "Recall is below the threshold.")
        self.assertGreaterEqual(f1, 0.3, "F1 Score is below the threshold.")

if __name__ == "__main__":
    unittest.main()
