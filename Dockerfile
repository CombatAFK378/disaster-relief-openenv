# ─────────────────────────────────────────────
# Disaster Relief Coordinator — OpenEnv
# Meta PyTorch x Hugging Face OpenEnv AI Hackathon 2026
# ─────────────────────────────────────────────

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Disaster Relief Coordinator — OpenEnv" \
      org.opencontainers.image.description="OpenEnv-compliant RL environment for training AI disaster relief coordinators. Meta PyTorch x Hugging Face Hackathon 2026." \
      org.opencontainers.image.version="1.0.0"

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]