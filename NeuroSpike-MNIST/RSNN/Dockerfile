# Use CUDA base image
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY environment.yml .
COPY requirements.txt .

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda env create -f environment.yml

# Copy project files
COPY . .

# Set entrypoint
ENTRYPOINT ["conda", "run", "-n", "rsnn", "python", "main.py"]