# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

COPY pyproject.toml requirements.txt README.md LICENSE openenv.yaml ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "uvicorn", "envs.supply_chain_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
