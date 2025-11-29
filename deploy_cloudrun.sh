#!/bin/bash

# HydrAI-SWE Cloud Run Deployment Script

set -e

# Configuration
PROJECT_ID="forward-script-479715-d6"
SERVICE_NAME="hydrai-swe-service"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check gcloud
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed."
    exit 1
fi

# Set project
log_info "Setting GCP project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable APIs
log_info "Enabling necessary APIs..."
gcloud services enable run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com

# Build and Push Image
log_info "Building and pushing Docker image to GCR..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Deploy to Cloud Run
log_info "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 1

log_success "Deployment complete!"
log_info "Service URL:"
gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)'

# --- Cloud Scheduler Setup ---
log_info "Configuring Cloud Scheduler for 9:00-21:00 availability..."

# Enable Scheduler API
gcloud services enable cloudscheduler.googleapis.com

# Create Service Account for Scheduler
SA_NAME="scheduler-invoker"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe ${SA_EMAIL} &> /dev/null; then
    log_info "Creating service account ${SA_NAME}..."
    gcloud iam service-accounts create ${SA_NAME} --display-name "Cloud Scheduler Invoker"
fi

# Grant permission to invoke Cloud Run
gcloud run services add-iam-policy-binding ${SERVICE_NAME} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.admin" \
    --region=${REGION}

# Create Start Job (9:00 AM) - Sets max-instances to 10 (or desired default)
log_info "Creating Start Job (9:00 AM)..."
gcloud scheduler jobs create http ${SERVICE_NAME}-start \
    --schedule="0 9 * * *" \
    --time-zone="America/Chicago" \
    --uri="https://${REGION}-run.googleapis.com/apis/serving.knative.dev/v1/namespaces/${PROJECT_ID}/services/${SERVICE_NAME}" \
    --http-method=PUT \
    --oauth-service-account-email=${SA_EMAIL} \
    --headers="Content-Type=application/json,User-Agent=Google-Cloud-Scheduler" \
    --message-body='{"apiVersion":"serving.knative.dev/v1","kind":"Service","metadata":{"name":"'${SERVICE_NAME}'"},"spec":{"template":{"spec":{"containers":[{"image":"'${IMAGE_NAME}'"}]},"metadata":{"annotations":{"autoscaling.knative.dev/maxScale":"10"}}}}}' \
    --location=${REGION} \
    --quiet || log_info "Job ${SERVICE_NAME}-start might already exist, skipping..."

# Create Stop Job (21:00 PM) - Sets max-instances to 0
log_info "Creating Stop Job (21:00 PM)..."
gcloud scheduler jobs create http ${SERVICE_NAME}-stop \
    --schedule="0 21 * * *" \
    --time-zone="America/Chicago" \
    --uri="https://${REGION}-run.googleapis.com/apis/serving.knative.dev/v1/namespaces/${PROJECT_ID}/services/${SERVICE_NAME}" \
    --http-method=PUT \
    --oauth-service-account-email=${SA_EMAIL} \
    --headers="Content-Type=application/json,User-Agent=Google-Cloud-Scheduler" \
    --message-body='{"apiVersion":"serving.knative.dev/v1","kind":"Service","metadata":{"name":"'${SERVICE_NAME}'"},"spec":{"template":{"spec":{"containers":[{"image":"'${IMAGE_NAME}'"}]},"metadata":{"annotations":{"autoscaling.knative.dev/maxScale":"0"}}}}}' \
    --location=${REGION} \
    --quiet || log_info "Job ${SERVICE_NAME}-stop might already exist, skipping..."

log_success "Scheduling configured!"
