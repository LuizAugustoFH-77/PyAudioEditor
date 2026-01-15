# PyAudioEditor 🎧

A modern, high-performance Digital Audio Workstation (DAW) built with Python, PyQt6, and Librosa.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green.svg)
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)

## ✨ Features

- **Multi-track Editing**: Import multiple audio files and arrange them on a synchronized timeline.
- **High-Performance Visualization**: Optimized waveform rendering with support for real-time spectrogram view.
- **Professional DSP Effects**:
  - Amplify, Fade In/Out, Normalize
  - Echo/Delay & Reverb
  - Low-pass, High-pass & Band-pass Filters
  - EQ (Peaking, Low-shelf, High-shelf)
  - Compressor, De-esser, Chorus
  - Soft Clipping (Saturation)
- **One-Click Presets**:
  - **Slowed + Reverb**: The classic aesthetic vibe.
  - **Nightcore**: Speed up with pitch shift.
  - **Lo-Fi Style**: Warm, filtered sound.
  - **Bass Boosted**: Robust low-end enhancement with soft saturation.
  - **🎤 Miku Ver.**: Transform vocals to sound like Hatsune Miku (Vocaloid style).
- **AI-Powered Vocal Separation**:
  - **Demucs AI**: State-of-the-art source separation (requires `torch` + `demucs`)
  - **Spleeter**: Alternative AI separation (requires `spleeter`)
  - **HPSS**: Fast Harmonic/Percussive separation via librosa
  - **DSP Karaoke**: Center-channel cancellation fallback
- **Robust File Support**: Load MP3, WAV, FLAC, OGG, and more via `librosa`.
- **Export**: Mix down your project to high-quality WAV, MP3, FLAC, or OGG.
- **Undo/Redo System**: 50 levels of history for all destructive edits.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- FFmpeg (required by `librosa` for MP3 support)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/LuizAugustoFH-77/PyAudioEditor.git
   cd PyAudioEditor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Optional: AI Vocal Separation

For best-quality AI vocal separation using **Demucs**:

```bash
# CPU only
pip install torch torchaudio demucs

# With NVIDIA GPU (CUDA)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install demucs

# With AMD GPU (DirectML on Windows)
pip install torch-directml demucs
```

Alternative with **Spleeter**:
```bash
pip install spleeter
```

### Running the App

```bash
python main.py
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=html
```

## 🛠️ Tech Stack

- **GUI**: [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)
- **Audio Engine**: [SoundDevice](https://python-sounddevice.readthedocs.io/)
- **DSP & I/O**: [Librosa](https://librosa.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
- **AI Separation**: [Demucs](https://github.com/facebookresearch/demucs), [Spleeter](https://github.com/deezer/spleeter)
- **Icons**: [QtAwesome](https://github.com/spyder-ide/qtawesome)
- **Theming**: [qdarktheme](https://github.com/5yutan5/PyQtDarkTheme)

## 📂 Project Structure

```
PyAudioEditor/
├── main.py                 # Application entry point
├── src/
│   ├── core/               # Audio engine, tracks, and DSP logic
│   │   ├── audio_engine.py # Main orchestrator
│   │   ├── project.py      # Project state management
│   │   ├── track.py        # Audio track representation
│   │   ├── playback.py     # Playback controller
│   │   ├── effects_basic.py    # Basic DSP effects
│   │   ├── effects_vocal.py    # Vocal effects + presets
│   │   ├── separation.py   # AI vocal separation
│   │   ├── undo_manager.py # Undo/redo system
│   │   ├── config.py       # Centralized configuration
│   │   └── types.py        # Type definitions
│   ├── ui/                 # PyQt widgets and main window
│   │   ├── main_window.py  # Main application window
│   │   ├── track_widget.py # Track control widget
│   │   ├── waveform_view.py    # Waveform visualization
│   │   └── time_ruler.py   # Time ruler widget
│   └── utils/              # Logging and helpers
├── tests/                  # Test suite
├── requirements.txt        # Project dependencies
├── pyproject.toml          # Modern Python project config
└── README.md               # You are here!
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
