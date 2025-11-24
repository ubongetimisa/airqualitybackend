#!/bin/bash
# Ensure pip, setuptools, and wheel are up to date, then install deps
pip install --upgrade pip setuptools wheel

# Install all dependencies (add --no-cache-dir to save space if needed)
pip install -r requirements.txt
