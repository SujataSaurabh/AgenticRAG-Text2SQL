# Deploying ALSPubsChat to Cloud Run

Two private Cloud Run services, deployed by `.github/workflows/deploy.yml` on every push to `main`:

| Service | Image | Size | Access |
|---|---|---|---|
| `text2sql-rag-service` | `Dockerfile` (FastAPI + Qwen2.5-Coder-3B, weights baked in) | 16 GiB / 4 vCPU, concurrency 1 | only the `pubschat-ui` service account |
| `text2sql-ui` | `Dockerfile.ui` (Streamlit) | 1 GiB / 1 vCPU | users/groups you grant `roles/run.invoker` |

`database.db` is not in git (public repo). The workflow copies it from
`gs://als-user-office-software-pubschat-data/database.db` before building.

## One-time setup
```bash
DEPLOYER_SA=<client_email from the GCP_SA_KEY JSON> ./deploy/setup_gcp.sh
```

## Refresh the data
```bash
gcloud storage cp database.db gs://als-user-office-software-pubschat-data/database.db
# then re-run the workflow (Actions → Deploy → Run workflow)
```

## Open the UI (private service)
```bash
gcloud run services proxy text2sql-ui --region=us-central1 --port=8501
# browse http://localhost:8501
```

## Call the API directly
```bash
URL=$(gcloud run services describe text2sql-rag-service --region=us-central1 --format='value(status.url)')
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" -H 'Content-Type: application/json' \
  -d '{"question":"How many publications in 2019?","history":[]}' "$URL/query"
```
(Your user needs `roles/run.invoker` on the API service for this.)
