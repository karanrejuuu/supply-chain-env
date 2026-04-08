FROM python:3.11-slim

WORKDIR /app

# Ensure we have required backend dependencies including FastAPI and Uvicorn
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# Move into source code scope
COPY . .

# Environment variables for configuration scaling 
ENV WORKERS=4
ENV MAX_CONCURRENT_ENVS=100

CMD ["uvicorn", "src.envs.supply_chain_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
