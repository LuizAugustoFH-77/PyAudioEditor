# PyAudioEditor - Agent Instructions

## Project Overview
PyAudioEditor is a professional audio editor built with PyQt6, featuring waveform visualization, multi-track editing, audio effects, and AI-powered vocal separation.

## Development Commands

### Run the Application
```bash
python main.py
```

### Run Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_effects_basic.py -v

# Run fast tests only (skip slow AI tests)
pytest -m "not slow"
```

### Type Checking
```bash
mypy src/
```

### Linting
```bash
ruff check src/
ruff format src/
```

### Install Dependencies
```bash
# Basic installation
pip install -r requirements.txt

# With AI vocal separation (Demucs) - CPU only
pip install -r requirements.txt torch torchaudio demucs

# Development installation
pip install -e ".[dev]"

# Full installation with all optional features
pip install -e ".[all]"
```

## Project Structure

```
PyAudioEditor/
├── main.py                 # Application entry point
├── src/
│   ├── core/               # Core audio processing (no Qt dependencies)
│   │   ├── audio_engine.py # Main orchestrator
│   │   ├── project.py      # Project state management
│   │   ├── track.py        # Audio track representation
│   │   ├── playback.py     # Playback controller
│   │   ├── undo_manager.py # Undo/redo system
│   │   ├── effects_basic.py    # Basic DSP effects
│   │   ├── effects_vocal.py    # Vocal processing effects
│   │   ├── separation.py   # AI vocal separation
│   │   ├── config.py       # Centralized configuration
│   │   └── types.py        # Type definitions
│   ├── ui/                 # Qt UI components
│   │   ├── main_window.py  # Main application window
│   │   ├── track_widget.py # Track control widget
│   │   ├── waveform_view.py    # Waveform visualization
│   │   └── time_ruler.py   # Time ruler widget
│   └── utils/
│       └── logger.py       # Logging configuration
├── tests/                  # Test suite
│   ├── conftest.py         # Pytest fixtures
│   ├── test_effects_basic.py
│   ├── test_track.py
│   ├── test_project.py
│   └── test_undo_manager.py
├── requirements.txt
├── pyproject.toml
└── AGENTS.md
```

## Code Style Guidelines

- **Python version**: 3.10+
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Use Google-style docstrings
- **Line length**: 100 characters max
- **Imports**: Use absolute imports, group with isort

## Architecture Patterns

### Core Module (src/core/)
- Pure Python, no Qt dependencies
- All functions should be testable without GUI
- Use dataclasses for data structures
- Effects are pure functions: `(AudioArray, sr, **params) -> AudioArray`

### UI Module (src/ui/)
- Qt-specific code only
- Widgets delegate logic to core modules
- Use signals/slots for communication

### Adding New Effects
1. Add the effect function to `effects_basic.py` or `effects_vocal.py`
2. Add mapping in `main_window.py::apply_effect_to_selection`
3. Add test in `tests/test_effects_basic.py`

### Adding New Presets
1. Create preset function in `effects_vocal.py`
2. Add mapping in `main_window.py::apply_preset_dialog`

## Performance Notes

- Waveform rendering uses adaptive downsampling
- Large files use min/max envelope visualization
- Spectrogram is cached and computed on demand
- Playback uses sounddevice with configurable block size
