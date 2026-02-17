FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config.yaml ./config.yaml
COPY tests ./tests
COPY infra ./infra

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python3", "-m", "lie_engine.cli"]
