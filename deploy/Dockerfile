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
COPY pyproject.toml ./
COPY allotrope ./allotrope
COPY scripts ./scripts

# CPU-only torch: the control policy is small (128-unit MLPs), so a GPU buys
# nothing here and would roughly 10x the image size for no benefit.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -e .

ENTRYPOINT ["python"]
CMD ["scripts/run_station_service.py", "--help"]
