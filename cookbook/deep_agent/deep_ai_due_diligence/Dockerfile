FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir omnicoreagent
# RUN pip install uv && uv init && uv add omnicoreagent

# COPY requirements.txt .
# RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["omniserve", "quickstart"]
