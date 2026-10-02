#!/usr/bin/env bash
# Let everyone with a lab Google account open the chat UI in a browser (via IAP).
# Run once after the first deploy with IAP, as a project owner:
#   ./deploy/grant_lab_access.sh                       # default: domain:lbl.gov
#   MEMBERS="group:als-pubs-chat@lbl.gov" ./deploy/grant_lab_access.sh   # narrower: one group
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-als-user-office-software}"
REGION="${REGION:-us-central1}"
UI_SERVICE="${UI_SERVICE:-text2sql-ui}"
MEMBERS="${MEMBERS:-domain:lbl.gov}"

gcloud config set project "$PROJECT_ID" >/dev/null

# IAP's built-in Google sign-in only works for accounts inside the project's
# organization. A project outside the lbl.gov organization needs a custom
# OAuth client, set up once in the console.
if ! gcloud projects get-ancestors "$PROJECT_ID" --format='value(type)' | grep -qx organization; then
  echo "WARNING: ${PROJECT_ID} is not inside a Google Cloud organization."
  echo "  Enable IAP once in the console (Cloud Run > ${UI_SERVICE} > Security > Identity-Aware Proxy)"
  echo "  so it creates the OAuth client, then re-run this script."
fi

IFS=',' read -ra list <<< "$MEMBERS"
for m in "${list[@]}"; do
  echo "== Granting IAP access on ${UI_SERVICE} to ${m}"
  gcloud iap web add-iam-policy-binding \
    --member="$m" --role=roles/iap.httpsResourceAccessor \
    --region="$REGION" --resource-type=cloud-run --service="$UI_SERVICE"
done

URL="$(gcloud run services describe "$UI_SERVICE" --region="$REGION" --format='value(status.url)')"
echo
echo "Done. Lab users open ${URL} and sign in with their lab Google account."
echo "Anyone else is refused by IAP before the app sees the request."
