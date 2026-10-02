#!/usr/bin/env bash
# One-time GCP setup for the ALSPubsChat Cloud Run deployment.
# Run from the repo root (src/) with gcloud logged in as a project owner:
#   DEPLOYER_SA=<email-of-the-GCP_SA_KEY-account> ./deploy/setup_gcp.sh
# Optional: INVOKERS="user:someone@lbl.gov,group:team@lbl.gov"
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-als-user-office-software}"
REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-rag-repo}"
BUCKET="${BUCKET:-${PROJECT_ID}-pubschat-data}"
UI_SERVICE="${UI_SERVICE:-text2sql-ui}"
API_SA="pubschat-api@${PROJECT_ID}.iam.gserviceaccount.com"
UI_SA="pubschat-ui@${PROJECT_ID}.iam.gserviceaccount.com"
: "${DEPLOYER_SA:?Set DEPLOYER_SA to the client_email in the GCP_SA_KEY JSON}"
INVOKERS="${INVOKERS:-user:$(gcloud config get-value account 2>/dev/null)}"

gcloud config set project "$PROJECT_ID"

echo "== Enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  iam.googleapis.com storage.googleapis.com logging.googleapis.com \
  aiplatform.googleapis.com iap.googleapis.com
# Make sure the IAP service agent exists (it forwards signed-in users to Cloud Run)
gcloud beta services identity create --service=iap.googleapis.com --project="$PROJECT_ID" >/dev/null

echo "== Artifact Registry repo"
gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPOSITORY" --repository-format=docker \
       --location="$REGION" --description="ALSPubsChat images"

echo "== Runtime service accounts"
for sa in pubschat-api pubschat-ui; do
  gcloud iam service-accounts describe "${sa}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1 \
    || gcloud iam service-accounts create "$sa" --display-name="ALSPubsChat ${sa#pubschat-}"
done

echo "== API service account may call Gemini on Vertex AI"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${API_SA}" --role=roles/aiplatform.user --condition=None >/dev/null

echo "== Deployer permissions for ${DEPLOYER_SA}"
for role in roles/run.admin roles/artifactregistry.writer roles/logging.viewer roles/iap.admin; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${DEPLOYER_SA}" --role="$role" --condition=None >/dev/null
done
for sa in "$API_SA" "$UI_SA"; do
  gcloud iam service-accounts add-iam-policy-binding "$sa" \
    --member="serviceAccount:${DEPLOYER_SA}" --role=roles/iam.serviceAccountUser >/dev/null
done

echo "== Data bucket gs://${BUCKET}"
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" \
       --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${DEPLOYER_SA}" --role=roles/storage.objectViewer >/dev/null
gcloud storage cp database.db "gs://${BUCKET}/database.db"

echo "== Done. After the first successful workflow run, grant UI access with:"
IFS=',' read -ra members <<< "$INVOKERS"
for m in "${members[@]}"; do
  echo "  gcloud run services add-iam-policy-binding $UI_SERVICE --region=$REGION --member='$m' --role=roles/run.invoker"
done
