#!/usr/bin/env bash
# One-time setup of the Cloud SQL (PostgreSQL) reporting database for the POC.
# Run from src/ with gcloud logged in as a project owner:
#   ./deploy/setup_cloudsql.sh
# Then load data with the command printed at the end.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-als-user-office-software}"
REGION="${REGION:-us-central1}"
INSTANCE="${INSTANCE:-pubs-reporting}"
DB_NAME="${DB_NAME:-pubsreporting}"
TIER="${TIER:-db-f1-micro}"            # ~$8/month compute; db-g1-small for more headroom
SECRET="${SECRET:-pubs-reporting-postgres-password}"
API_SA="pubschat-api@${PROJECT_ID}.iam.gserviceaccount.com"
API_DB_USER="pubschat-api@${PROJECT_ID}.iam"   # IAM DB user name = SA email minus .gserviceaccount.com

gcloud config set project "$PROJECT_ID"

echo "== Enabling APIs"
gcloud services enable sqladmin.googleapis.com secretmanager.googleapis.com

echo "== Cloud SQL instance ${INSTANCE} (${TIER}, PostgreSQL 16) — first creation takes ~5-10 min"
if ! gcloud sql instances describe "$INSTANCE" >/dev/null 2>&1; then
  gcloud sql instances create "$INSTANCE" \
    --database-version=POSTGRES_16 --edition=ENTERPRISE --tier="$TIER" \
    --region="$REGION" --storage-type=SSD --storage-size=10GB \
    --database-flags=cloudsql.iam_authentication=on \
    --backup-start-time=10:00 --retained-backups-count=7
fi

echo "== Database ${DB_NAME}"
gcloud sql databases describe "$DB_NAME" --instance="$INSTANCE" >/dev/null 2>&1 \
  || gcloud sql databases create "$DB_NAME" --instance="$INSTANCE"

echo "== Admin password for 'postgres' (stored in Secret Manager: ${SECRET})"
if ! gcloud secrets describe "$SECRET" >/dev/null 2>&1; then
  PW="$(openssl rand -base64 24 | tr -d '/+=')"
  printf '%s' "$PW" | gcloud secrets create "$SECRET" --data-file=- --replication-policy=automatic
  gcloud sql users set-password postgres --instance="$INSTANCE" --password="$PW"
fi

echo "== IAM database user for the API service account (no password)"
gcloud sql users describe "$API_DB_USER" --instance="$INSTANCE" >/dev/null 2>&1 \
  || gcloud sql users create "$API_DB_USER" --instance="$INSTANCE" --type=cloud_iam_service_account
for role in roles/cloudsql.client roles/cloudsql.instanceUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${API_SA}" --role="$role" --condition=None >/dev/null
done

CONN="$(gcloud sql instances describe "$INSTANCE" --format='value(connectionName)')"
cat <<MSG

== Done. Instance connection name: ${CONN}

Load (or reload) the data from your laptop:
  gcloud auth application-default login
  PGPASSWORD="\$(gcloud secrets versions access latest --secret=${SECRET})" \\
    python load_data.py postgres --data-dir ../data --instance ${CONN} \\
      --user postgres --dbname ${DB_NAME} --reader "${API_DB_USER}"

Point the API at it (GitHub → Settings → Secrets and variables → Actions → Variables):
  DB_BACKEND = postgres
then re-run the Deploy workflow. Set DB_BACKEND = sqlite to switch back.

To pause compute charges:  gcloud sql instances patch ${INSTANCE} --activation-policy=NEVER
To resume:                 gcloud sql instances patch ${INSTANCE} --activation-policy=ALWAYS
MSG
