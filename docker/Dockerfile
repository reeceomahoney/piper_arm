FROM python:3.10-slim

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswscale-dev git linux-libc-dev build-essential

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /work

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --no-install-project --link-mode=copy

COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
