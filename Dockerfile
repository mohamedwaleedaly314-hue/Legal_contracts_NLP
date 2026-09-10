# Mizan — Streamlit + hybrid RAG over the Egyptian legal corpus.
#
#   docker build -t mizan .
#   docker run -p 8501:8501 --env-file .env mizan
#
# Two stages so the 2.2 GB embedding model is downloaded once at build time and
# copied into the final image. Downloading it at first request instead means
# the first user waits ten minutes and most platforms time the request out
# before it finishes.

# --------------------------------------------------------------------------
# Stage 1 - fetch the embedding model
# --------------------------------------------------------------------------
FROM python:3.11-slim AS models

ARG EMBEDDING_MODEL=BAAI/bge-m3
ENV HF_HOME=/opt/models \
    PIP_NO_CACHE_DIR=1

# huggingface_hub alone, not sentence-transformers: the download needs no torch
# and this stage is discarded, so keeping it small keeps the build quick.
RUN pip install --no-cache-dir "huggingface_hub>=0.20"

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${EMBEDDING_MODEL}', \
    allow_patterns=['*.json','*.txt','*.model','*.safetensors','1_Pooling/*'], \
    ignore_patterns=['*.onnx','*.h5','*.msgpack','pytorch_model.bin'])"

# --------------------------------------------------------------------------
# Stage 2 - the application
# --------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    # No network call to check for a newer revision on every load.
    HF_HUB_OFFLINE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# tesseract-ocr-ara is what makes a scanned Arabic contract readable at all.
# Poppler is gone: PDF pages are rasterised in-process by PyMuPDF now, so
# pdf2image is only a fallback and the system package is dead weight.
# libgl1/libglib are OpenCV's runtime dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       tesseract-ocr-ara \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Requirements before source, so a code change does not reinstall torch.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --from=models /opt/models /opt/models

# .dockerignore keeps .env, backup/, outputs/ and the notebooks out of this.
COPY . /app

EXPOSE 8501

# Fails fast and legibly when the index is missing, the collection is empty,
# or the API key is absent - instead of starting and answering every clause
# with "no legal articles found".
HEALTHCHECK --interval=30s --timeout=20s --start-period=120s --retries=3 \
    CMD python -m src.preflight --json --no-qdrant || exit 1

# PORT is what Hugging Face Spaces and most PaaS hand the container.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false"]
