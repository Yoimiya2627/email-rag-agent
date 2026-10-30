FROM python:3.11-slim

WORKDIR /app

# Install system deps for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy an explicit source allowlist; runtime mail and OAuth files are mounted.
COPY agents ./agents
COPY api ./api
COPY config ./config
COPY core ./core
COPY models ./models
COPY scripts ./scripts
COPY mcp_server.py ./
RUN mkdir -p /app/data

# Model downloads are an explicit build option; never put model/API tokens in ARG.
ARG PRELOAD_EMBEDDING_MODEL=false
ARG EMBEDDING_MODEL=BAAI/bge-m3
ARG EMBEDDING_MODEL_REVISION=
ARG HF_ENDPOINT=https://huggingface.co
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV EMBEDDING_MODEL_REVISION=${EMBEDDING_MODEL_REVISION}
ENV HF_ENDPOINT=${HF_ENDPOINT}
RUN if [ "$PRELOAD_EMBEDDING_MODEL" = "true" ]; then python scripts/preload_model.py; \
    elif [ "$PRELOAD_EMBEDDING_MODEL" != "false" ]; then echo "PRELOAD_EMBEDDING_MODEL must be true or false" >&2; exit 2; fi

EXPOSE 8000

ENV API_HOST=0.0.0.0
ENV API_PORT=8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
