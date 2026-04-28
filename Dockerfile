# ─── Stage 1: dependency builder ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Copy requirements (optional, but cleaner)
COPY requirements.txt .

RUN pip install --upgrade pip --root-user-action=ignore && \
    pip install --prefix=/install/deps --no-cache-dir --root-user-action=ignore \
        flask==3.0.0 \
        gunicorn==22.0.0 \
        pillow==10.4.0 \
        torchvision==0.18.1 \
        onnxruntime==1.18.0

# ─── Stage 2: lean runtime image ───────────────────────────────────────────
FROM python:3.11-slim

# Security: non-root user
RUN useradd --create-home appuser
WORKDIR /app

# Copy installed dependencies
COPY --from=builder /install/deps /usr/local

# Copy app + ONNX model
COPY app.py .
COPY fallehi_student.onnx .
COPY fallehi_student.onnx.data .

# Env optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2

USER appuser
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Run server
CMD ["sh", "-c", "gunicorn app:app --workers 2 --threads 2 --bind 0.0.0.0:${PORT:-5000} --timeout 60 --access-logfile - --error-logfile -"]