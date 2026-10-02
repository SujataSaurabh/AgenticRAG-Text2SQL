# ---- Text2SQL RAG API (FastAPI + Gemini on Vertex AI) ----
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY main.py llm.py db.py schema_catalog.py load_data.py database.db ./

ENV LLM_BACKEND=vertex \
    GOOGLE_CLOUD_LOCATION=global \
    DB_PATH=/src/database.db \
    PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
