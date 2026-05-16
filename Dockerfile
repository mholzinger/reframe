# Use an official Python image as the base
FROM python:3.9-slim

# Install system dependencies for building dlib/face_recognition
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        libx11-dev \
        python3-dev \
        && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# Default settings — override via -e (or Synology Container Manager Environment tab)
ENV REFERENCE_FOLDER=/app/reference_photos
ENV PHOTO_FOLDER=/app/photo_folder
ENV OUTPUT_FOLDER=/app/similar_photos
ENV SIMILARITY_THRESHOLD=0.5
ENV DATE_RANGE_START=
ENV DATE_RANGE_END=
ENV NEIGHBOR_WINDOW=20
ENV NEIGHBOR_SIZE_RATIO=0.5
# WORKERS unset → defaults to cpu_count() at runtime. Set explicitly to tune.
ENV WORKERS=
# Regex(es) to skip by filename. Default catches 4K Stogram (Instagram archive) files.
ENV SKIP_FILENAME_PATTERNS=^FILE\\d+\\.JPG$
# Filename skip only triggers if the file is <= this many bytes (real camera
# photos are always larger). Protects EaseUS-recovered photos sharing the
# FILE<num>.JPG naming pattern. Set 0 to disable size guard (dangerous).
ENV SKIP_FILENAME_MAX_SIZE=500000

# Set work directory
WORKDIR /app

# Copy requirements.txt if you have one, else install directly
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy your code into the container
COPY . .

# Default command (can be overridden)
CMD ["python", "find_photos.py"]
