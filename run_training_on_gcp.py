import os
from google.cloud import aiplatform

# --- Configuration ---
# Replace with your GCP project ID and region
PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"  # Or your preferred region
# Replace with a GCS bucket for staging (e.g., "gs://your-bucket-name")
# This bucket will be used to stage your code and model artifacts.
STAGING_BUCKET = "gs://your-vertex-ai-staging-bucket"

# Initialize the AI Platform client
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

def submit_training_job_to_gcp():
    """
    Submits a custom training job to Google Cloud Vertex AI.
    """
    print("Submitting training job to GCP Vertex AI...")

    # Define the custom job
    job = aiplatform.CustomJob(
        display_name="hydrai-swe-training",
        python_package_gcs_uri=None,  # We will build and upload the package
        python_module_name="src.models.train", # The module to run
        container_uri="gcr.io/cloud-aiplatform/training/tf-cpu.2-11.py310", # Or a GPU image if using GPU
        # Machine configuration
        machine_type="n1-standard-4", # Or a more powerful machine type
        accelerator_type=None, # Or "NVIDIA_TESLA_T4" if using GPU
        accelerator_count=0, # Or 1 if using GPU
        # Environment variables for the training job (optional)
        # env_variables={"MY_ENV_VAR": "my_value"},
        # Arguments passed to the training script (optional)
        # args=["--epochs", "50"],
    )

    # Run the job
    # This will automatically package your local code and upload it to STAGING_BUCKET
    # The 'requirements_path' argument tells Vertex AI to install dependencies from this file.
    job.run(
        service_account=None, # Use default service account or specify one
        requirements_path="requirements.txt",
        sync=True, # Set to False to run asynchronously
    )

    print("Training job submitted to GCP Vertex AI.")
    print(f"Job state: {job.state}")
    print(f"Job resource name: {job.resource_name}")
    print(f"Job logs: {job.log_url}")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Before running this script:
    # 1. Ensure you have authenticated to GCP (e.g., `gcloud auth application-default login`).
    # 2. Replace PROJECT_ID, REGION, and STAGING_BUCKET with your actual values.
    # 3. Make sure your `requirements.txt` is up-to-date.
    # 4. Ensure your `src/models/train.py` is ready for training.
    submit_training_job_to_gcp()
