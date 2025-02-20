#!/bin/bash

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ssnn

# Create necessary directories
mkdir -p results
mkdir -p data
mkdir -p logs
mkdir -p tests

# Install development dependencies
pip install pytest pytest-cov wandb

# Setup pre-commit hooks
pre-commit install

# Initialize wandb
wandb login