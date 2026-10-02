# Deploying ALSPubsChat to Cloud Run

Two private Cloud Run services, deployed by `.github/workflows/deploy.yml` on every push to `main`:

| Service | Image | Size | Access |
|---|---|---|---|
| `text2sql-rag-service` | `Dockerfile` (FastAPI + Gemini Flash-Lite on Vertex AI) | 512 MiB / 1 vCPU, concurrency 4 | only the `pubschat-ui` service account |
| `text2sql-ui` | `Dockerfile.ui` (Streamlit) | 1 GiB / 1 vCPU | lab Google accounts, through IAP (`deploy/grant_lab_access.sh`) |

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

## Open the UI (lab accounts only)
The UI sits behind Identity-Aware Proxy (IAP). Lab users open the service URL in a
browser and sign in with their lbl.gov Google account; anyone else is refused before
the request reaches the app. One-time, after the first IAP deploy:
```bash
./deploy/grant_lab_access.sh                                  # all of domain:lbl.gov
MEMBERS="group:<team>@lbl.gov" ./deploy/grant_lab_access.sh   # or one Google group
gcloud run services describe text2sql-ui --region=us-central1 --format='value(status.url)'
```
To remove someone's access, remove their binding with `gcloud iap web remove-iam-policy-binding`
(same flags as in the script).

## Call the API directly
```bash
URL=$(gcloud run services describe text2sql-rag-service --region=us-central1 --format='value(status.url)')
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" -H 'Content-Type: application/json' \
  -d '{"question":"How many publications in 2019?","history":[]}' "$URL/query"
```
(Your user needs `roles/run.invoker` on the API service for this.)

## LLM backend
- Cloud Run: `LLM_BACKEND=vertex`, model set by `GEMINI_MODEL` in `deploy.yml` (the API service account has `roles/aiplatform.user`).
- Local, with Gemini: `gcloud auth application-default login`, then `LLM_BACKEND=vertex uvicorn main:app --port 8080`.
- Local, offline Qwen: `pip install -r requirements.txt`, then `LLM_BACKEND=local uvicorn main:app --port 8080`.

## Database backend (POC: SQLite or Cloud SQL PostgreSQL)
What goes into the reporting database is defined in `schema_catalog.py`: only the
tables and columns listed there are loaded (`load_data.py` drops anything else),
and the API rejects SQL that touches other tables or isn't a single SELECT.

| `DB_BACKEND` | Where the data is | How to refresh it |
|---|---|---|
| `sqlite` (default) | `database.db` bundled in the API image | `python load_data.py sqlite --data-dir ../data`, upload to the GCS bucket, re-run the workflow |
| `postgres` | Cloud SQL instance `pubs-reporting`, schema `reporting` | `python load_data.py postgres ...` (see below); no redeploy needed |

One-time Cloud SQL setup: `./deploy/setup_cloudsql.sh` (prints the load command).
Switch backends: GitHub repo → Settings → Secrets and variables → Actions → Variables →
`DB_BACKEND` = `postgres` or `sqlite`, then re-run the Deploy workflow.
Local API against Cloud SQL: `DB_BACKEND=postgres CLOUDSQL_INSTANCE=<conn-name> DB_USER=<you>@lbl.gov uvicorn main:app`
(after granting your IAM DB user read access with `--reader`).
