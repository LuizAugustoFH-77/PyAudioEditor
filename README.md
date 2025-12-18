# PyAudioEditor 🎧

A modern, high-performance Digital Audio Workstation (DAW) built with Python, PyQt6, and Librosa.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green.svg)
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)

## ✨ Features

- **Multi-track Editing**: Import multiple audio files and arrange them on a synchronized timeline.
- **High-Performance Visualization**: Optimized waveform rendering with support for real-time spectrogram view.
- **Professional DSP Effects**:
  - Amplify, Fade In/Out
  - Echo/Delay & Reverb
  - Low-pass & Low-shelf Filters
  - Soft Clipping (Saturation)
- **One-Click Presets**:
  - **Slowed + Reverb**: The classic aesthetic vibe.
  - **Nightcore**: Speed up with pitch shift.
  - **Lo-Fi Style**: Warm, filtered sound.
  - **Bass Boosted**: Robust low-end enhancement with soft saturation.
- **AI-Powered Tools**:
  - Harmonic/Percussive Separation (HPSS).
  - Vocal Removal (Karaoke DSP).
- **Robust File Support**: Load MP3, WAV, FLAC, and more via `librosa`.
- **Export**: Mix down your project to high-quality WAV or MP3.
- **Undo/Redo System**: 20 levels of history for all destructive edits.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- FFmpeg (required by `librosa` for MP3 support)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PyAudioEditor.git
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

### Running the App

```bash
python main.py
```

## 🛠️ Tech Stack

- **GUI**: [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)
- **Audio Engine**: [SoundDevice](https://python-sounddevice.readthedocs.io/)
- **DSP & I/O**: [Librosa](https://librosa.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
- **Icons**: [QtAwesome](https://github.com/spyder-ide/qtawesome)
- **Theming**: [PyQt-Dark-Theme](https://github.com/5yutan5/PyQt-Dark-Theme)

## 📂 Project Structure

```
PyAudioEditor/
├── main.py              # Application entry point
├── src/
│   ├── core/            # Audio engine, tracks, and DSP logic
│   ├── ui/              # PyQt widgets and main window
│   └── utils/           # Logging and helpers
├── requirements.txt     # Project dependencies
└── README.md            # You are here!
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
