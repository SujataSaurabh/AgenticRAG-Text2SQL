# ---- Text2SQL RAG API (FastAPI + Qwen on CPU) ----
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    ANONYMIZED_TELEMETRY=False

# C++ toolchain in case a ChromaDB extension has no prebuilt wheel
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# CPU-only PyTorch. --extra-index-url lets pip fetch torch's own dependencies
# (setuptools, sympy, ...) from PyPI when the PyTorch index lacks a new enough copy.
ARG TORCH_VERSION=2.14.0
RUN pip install "torch==${TORCH_VERSION}" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Bake model weights into the image so cold starts don't download ~6 GB
ARG MODEL_ID=Qwen/Qwen2.5-Coder-3B-Instruct
ARG EMBED_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from huggingface_hub import snapshot_download as d; \
d('${MODEL_ID}', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.py']); \
d('${EMBED_MODEL_ID}', allow_patterns=['*.json','*.safetensors','*.txt','1_Pooling/*'])"

COPY main.py datapreprocess.py database.db ./

ENV MODEL_ID=${MODEL_ID} \
    EMBED_MODEL_ID=${EMBED_MODEL_ID} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DB_PATH=/src/database.db \
    PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
