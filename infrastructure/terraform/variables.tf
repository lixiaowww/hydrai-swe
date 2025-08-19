variable "gcp_project_id" {
  description = "The GCP project ID to deploy the infrastructure to."
  type        = string
}

variable "gcp_region" {
  description = "The GCP region to deploy the infrastructure to."
  type        = string
  default     = "us-central1"
}
