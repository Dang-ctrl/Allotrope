# One image, two entry points (station service, TimescaleDB bridge) selected
# by the command passed at `docker run` / in docker-compose.yml. Both are thin
# Python processes over the same `allotrope` package, so one image is simpler
# to build and version than two near-identical ones.
FROM python:3.11-slim

# libgomp1 satisfies OpenDSS's OpenMP runtime dependency; the rest are ordinary
# build tools for the C-extension wheels PyPSA and psycopg pull in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch, installed before the project's own files are copied in. The
# control policy is small (128-unit MLPs), so a GPU buys nothing here and
# would roughly 10x the image size for no benefit. Kept in its own layer,
# ahead of the COPY below: torch is a slow, heavy download, and putting it
# after pyproject.toml meant any application code change forced a full torch
# re-download on every rebuild.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml ./
COPY allotrope ./allotrope
COPY scripts ./scripts

# Both entry points below need more than the base dependency set: the station
# service publishes over MQTT and, with --checkpoint, loads a torch agent
# (rl); the bridge talks to TimescaleDB via psycopg (deploy). Installing only
# `pip install -e .` here silently omits paho-mqtt/psycopg/torch and both
# scripts fail at import time inside the container -- caught by actually
# building and running this image, not by the (torch-free) test suite.
RUN pip install --no-cache-dir -e ".[rl,mqtt,deploy]"

ENTRYPOINT ["python"]
CMD ["scripts/run_station_service.py", "--help"]
