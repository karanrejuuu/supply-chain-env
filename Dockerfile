# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command: run the rule-based inference script
CMD ["python", "scripts/run_inference.py"]
