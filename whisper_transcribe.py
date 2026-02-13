#!/usr/bin/env python3
"""
Whisper Real-time Transcriber - Pro Version
- Informative Help CLI
- Auto-incrementing filenames in 'transcripts/' folder
- Modern Rich UI with live VU meter
- WSL2 & Native Linux support
"""

import argparse
import os
import sys
import threading
import subprocess
import time
import warnings
import numpy as np
import whisper
from pathlib import Path

# UI Imports
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress_bar import ProgressBar

# Silence noisy warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["PYTHONWARNINGS"] = "ignore"

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

console = Console()

def is_wsl() -> bool:
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False

def get_unique_path(base_folder: str, filename: str) -> str:
    """Ensures a unique filename by appending _1, _2, etc. inside transcripts/ folder."""
    folder = Path(base_folder)
    folder.mkdir(parents=True, exist_ok=True)
    
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    
    target_path = folder / filename
    counter = 1
    while target_path.exists():
        target_path = folder / f"{stem}_{counter}{suffix}"
        counter += 1
        
    return str(target_path)

# ============================================================
# Recorders
# ============================================================

class PulseAudioRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.process = None

    def start(self):
        cmd = ["parec", "--format=float32le", "--rate", str(self.sample_rate), "--channels=1", "--latency-msec=100"]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self, num_samples: int) -> np.ndarray:
        if not self.process or not self.process.stdout:
            return np.array([], dtype=np.float32)
        raw = self.process.stdout.read(num_samples * 4)
        if not raw: return np.array([], dtype=np.float32)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except:
                self.process.kill()

class SoundDeviceRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.stream = None

    def start(self):
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32')
        self.stream.start()

    def read(self, num_samples: int) -> np.ndarray:
        data, _ = self.stream.read(num_samples)
        return data.flatten().copy()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

# ============================================================
# UI Generator
# ============================================================

def make_layout(vu_level: float, transcript: str, status: str, filename: str) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="upper", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )

    level = min(vu_level * 15, 1.0)
    bar = ProgressBar(total=1.0, completed=level, width=None)
    layout["upper"].update(Panel(bar, title=f"[bold cyan]Mic Volume", border_style="cyan"))
    
    layout["main"].update(Panel(
        Text(transcript), 
        title=f"[bold green]Transcript ({os.path.basename(filename)})", 
        border_style="green",
        padding=(1, 2)
    ))
    
    layout["footer"].update(Panel(Text(status, justify="center"), border_style="yellow"))
    return layout

# ============================================================
# Application Logic
# ============================================================

class RealTimeWhisper:
    def __init__(self, model_size: str, requested_filename: str, chunk_sec: int):
        self.chunk_sec = chunk_sec
        self.sample_rate = 16000
        self.running = False
        self.full_transcript = ""
        self.current_vu = 0.0
        self.status = "Initializing..."
        self.output_path = get_unique_path("transcripts", requested_filename)

        if is_wsl():
            self.recorder = PulseAudioRecorder(self.sample_rate)
        else:
            self.recorder = SoundDeviceRecorder(self.sample_rate)

        with console.status(f"[bold green]Loading Whisper '{model_size}' model...") as s:
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            try:
                self.model = whisper.load_model(model_size)
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr

    def worker(self):
        num_samples_per_chunk = self.sample_rate * self.chunk_sec
        step_samples = int(self.sample_rate * 0.2)
        self.recorder.start()
        self.status = "LISTENING (Press ENTER to stop)"

        while self.running:
            audio_buffer = []
            collected = 0
            while collected < num_samples_per_chunk and self.running:
                chunk = self.recorder.read(step_samples)
                if len(chunk) > 0:
                    audio_buffer.append(chunk)
                    collected += len(chunk)
                    self.current_vu = np.sqrt(np.mean(chunk**2))
                else:
                    time.sleep(0.01)

            if not self.running: break

            self.status = "TRANSCRIBING..."
            full_audio = np.concatenate(audio_buffer)
            result = self.model.transcribe(full_audio.copy(), fp16=False, language="en")
            text = result["text"].strip()

            if text:
                self.full_transcript += text + " "
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(text + " ")
                    f.flush()
            self.status = "LISTENING (Press ENTER to stop)"

        self.recorder.stop()

    def run(self):
        self.running = True
        t_worker = threading.Thread(target=self.worker, daemon=True)
        t_worker.start()

        def wait_input():
            sys.stdin.readline()
            self.running = False

        threading.Thread(target=wait_input, daemon=True).start()

        try:
            with Live(make_layout(0, "", "", self.output_path), screen=True, refresh_per_second=10) as live:
                while self.running:
                    live.update(make_layout(self.current_vu, self.full_transcript, self.status, self.output_path))
                    time.sleep(0.05)
        except (KeyboardInterrupt, Exception):
            self.running = False
        finally:
            t_worker.join(timeout=1)
            word_count = len(self.full_transcript.split())
            console.print("\n[bold green]✓ Session Ended[/bold green]")
            console.print(f"[bold]Output:[/bold] [cyan]{os.path.abspath(self.output_path)}[/cyan]")
            console.print(f"[bold]Words:[/bold] {word_count}\n")

# ============================================================
# CLI Configuration
# ============================================================

def main():
    description = """
🚀 Real-Time Whisper Transcriber for Linux & WSL2

This tool records audio in chunks and transcribes them using OpenAI's Whisper.
Files are automatically saved to the 'transcripts/' directory with unique names.
    """

    epilog = """
MODEL SELECTION:
  tiny   - (~75MB)  Fastest, lowest accuracy. Best for weak CPUs.
  base   - (~145MB) Fast, reliable. Good balance for most laptops.
  small  - (~480MB) Slower, significantly better punctuation/accuracy.
  medium - (~1.5GB) Slow, high quality. Best for clear, formal speech.
  large  - (~3.0GB) Very slow, highest accuracy. Recommended only with GPU.

CHUNK DURATION (-c):
  4-6s   - (Default) Good real-time feel. Fast updates.
  8-12s  - Better for long sentences or slow speakers. Improved context.
  15s+   - Highest accuracy, but long delay between speech and text.

EXAMPLES:
  python transcribe.py -m base                     # Balanced performance
  python transcribe.py -m small -c 10              # High accuracy, 10s chunks
  python transcribe.py -o meeting.txt              # Saves to transcripts/meeting.txt
    """

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-m", "--model", 
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to use. Larger models are more accurate but slower."
    )

    parser.add_argument(
        "-o", "--output", 
        default="transcription.txt",
        help="Output filename. Will be saved in 'transcripts/' folder with auto-incrementing numbers if it exists."
    )

    parser.add_argument(
        "-c", "--chunk", 
        type=int, 
        default=4,
        help="Duration of audio chunks in seconds before transcribing. Lower = faster updates; Higher = better context."
    )

    args = parser.parse_args()

    app = RealTimeWhisper(args.model, args.output, args.chunk)
    app.run()

if __name__ == "__main__":
    main()