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
        staging_version_number = latest_version.version

        # Get the current production model ,if any
        production_versions = client.get_latest_versions(model_name,stages=["production"])

        if production_versions:
             current_production_versions = production_versions[0]
             production_version_number = current_production_versions.version

             #transition the current production model to archived
             client.transition_model_version_stage(
                  name=model_name,
                  version=production_version_number,
                  stage ="Archived",
                  archive_existing_versions=False,
             )

             print(f"Archived model version {production_version_number} in 'Production'.")
        else:
             print("no model currently in 'Production'.")


        # transition the latest staging model to production
        client.transition_model_version_stage(
                  name=model_name,
                  version=staging_version_number,
                  stage ="Production",
                  archive_existing_versions=False,
             )
        print(f"promoted model version {staging_version_number} to 'Production'.")

if __name__ == "__main__":
    promote_model_to_production()

