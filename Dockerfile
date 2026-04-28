FROM python:3.11-slim

RUN useradd --create-home appuser
WORKDIR /app

# Install only what is needed
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

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

USER appuser
EXPOSE 5000

CMD ["gunicorn", "app:app", "--workers", "2", "--threads", "2", "--bind", "0.0.0.0:5000"]