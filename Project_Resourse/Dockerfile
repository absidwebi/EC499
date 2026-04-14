# EC499 — Stage 4 Inference Demo Container
# Python 3.10 slim — CPU-only inference (no CUDA needed for demo)
FROM python:3.10-slim

# System libraries required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer cache efficiency
COPY requirements.txt .

# Install Python packages — CPU-only torch to keep image under 1 GB
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        flask>=2.0 \
        pefile==2023.2.7 \
        opencv-python-headless \
        Pillow \
        numpy==1.26.4 && \
    pip install --no-cache-dir \
        torch==2.2.2+cpu \
        torchvision==0.17.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY config.py .
COPY models.py .
COPY convert_to_malimg.py .
COPY inference.py .
COPY app.py .
COPY templates/ ./templates/

# Copy trained model weights
COPY models/3c2d_malex_clean_vulnerable.pth /app/models/
COPY models/3c2d_malex_adversarially_trained.pth /app/models/

# Environment defaults
ENV MODEL_TYPE=clean
ENV PORT=5000

EXPOSE 5000

CMD ["python", "app.py"]
