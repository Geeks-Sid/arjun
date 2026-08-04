# Development image (CUDA). Base image pinned by immutable version tag;
# pin by digest for release builds: `docker buildx imagetools inspect
# nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` and append @sha256:<digest>.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/medfm-venv \
    PATH="/opt/medfm-venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /usr/local/bin/

RUN useradd --create-home --shell /bin/bash medfm
WORKDIR /workspace

COPY --chown=medfm:medfm pyproject.toml uv.lock README.md ./
COPY --chown=medfm:medfm medfm ./medfm

USER medfm
RUN uv python install 3.13 \
    && uv sync --frozen --no-dev --extra medical --extra hf --extra cuda

COPY --chown=medfm:medfm . .

CMD ["python", "-m", "medfm.tools.doctor", "--backend", "cuda"]
