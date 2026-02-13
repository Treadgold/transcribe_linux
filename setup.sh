#!/bin/bash
# Quick setup script for Whisper Transcription Tool

set -e

echo "=== Whisper Transcription Tool - Setup Script ==="
echo ""

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# 2. Check for System Dependencies
echo "Checking system dependencies..."
MISSING=""
if ! command -v ffmpeg &> /dev/null; then MISSING="ffmpeg $MISSING"; fi
if ! command -v parec &> /dev/null; then MISSING="pulseaudio-utils $MISSING"; fi

if [ ! -z "$MISSING" ]; then
    echo "Missing system packages: $MISSING"
    echo "Please run: sudo apt update && sudo apt install -y $MISSING"
    exit 1
fi

# 3. Create virtual environment
if [ ! -d "whisper_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv whisper_env
    echo "✓ Virtual environment created"
fi

# 4. Install Python Dependencies
echo "Installing Python dependencies (rich, whisper, sounddevice, torch)..."
source whisper_env/bin/activate
pip install --upgrade pip -q
pip install openai-whisper rich sounddevice numpy torch -q

# 5. Make scripts executable
chmod +x transcribe.py
if [ -f "run.sh" ]; then chmod +x run.sh; fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start transcribing:"
echo "  ./run.sh"
echo ""