FROM python:3.11-slim

# Security user
RUN useradd --create-home appuser
WORKDIR /app

# Install only required dependencies
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    gunicorn==22.0.0 \
    pillow==10.4.0 \
    numpy==1.26.4 \
    onnxruntime==1.18.0

# Copy app + model
COPY app.py .
COPY fallehi_student.onnx .
COPY fallehi_student.onnx.data .

# Railway-safe env
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    ORT_DISABLE_MEMORY_ARENA=1

USER appuser

EXPOSE 5000

# IMPORTANT: Railway requires $PORT
CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT"]