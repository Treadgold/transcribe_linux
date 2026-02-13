#!/bin/bash
# Convenience script to run the whisper transcription tool

# Check if venv exists
if [ ! -d "whisper_env" ]; then
    echo "Virtual environment not found. Running setup.sh first..."
    ./setup.sh
fi

# Activate virtual environment
source whisper_env/bin/activate

# Run the script with any passed arguments (e.g., ./run.sh -m base)
python3 transcribe.py "$@"