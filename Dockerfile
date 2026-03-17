# ────────────────────────────────────────────────
# Video Animator AI – Dockerfile
# Ubuntu 22.04 + CUDA 11.7 + Python 3.11
# Ollama, FFmpeg, Blender, and all Python deps
# ────────────────────────────────────────────────
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# ── System packages ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg \
        blender \
        git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Ollama ────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Python environment ────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --upgrade pip \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python3.11 -m pip install --no-cache-dir -e .

# ── Expose Ollama port ────────────────────────
EXPOSE 11434

# ── Default entrypoint ────────────────────────
# Start Ollama in the background, then run the CLI.
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--help"]
