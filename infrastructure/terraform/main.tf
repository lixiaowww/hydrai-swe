# This is a placeholder for the Terraform configuration.
# In a real application, this would define the cloud infrastructure
# required for the project (e.g., VPC, database, Kubernetes cluster, etc.).

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

resource "google_storage_bucket" "data_bucket" {
  name     = "${var.gcp_project_id}-data"
  location = var.gcp_region
  storage_class = "STANDARD"
}
