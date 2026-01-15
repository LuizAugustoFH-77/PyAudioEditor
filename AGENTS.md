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

# With AI vocal separation + AMD GPU support (DirectML)
pip install -r requirements.txt torch-directml demucs

# With AI vocal separation + NVIDIA GPU support (CUDA)
pip install -r requirements.txt demucs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Development installation
pip install -e ".[dev]"

# Full installation with all optional features
pip install -e ".[all]"
```

## GPU Support

### AMD GPUs (RX 400/500/5000/6000/7000 series) on Windows
Use **DirectML** backend:
```bash
pip install torch-directml
```
The application will automatically detect your AMD GPU via DirectML.

### AMD GPUs (Recommended for GPU separation): ONNX Runtime + DirectML
Demucs via PyTorch is not reliably compatible with DirectML for AMD GPUs. For GPU-accelerated
separation on AMD, prefer the ONNX path:
```bash
pip install onnxruntime-directml
```
Then in the app use **Tools → AI & Vocal Separation → Separate Vocals (ONNX DirectML)** and
configure an MDX-style `.onnx` model via **Configure ONNX Model...**.

For **4-stem separation on GPU**, use **Tools → AI & Vocal Separation → Separate 4 Stems (ONNX DirectML)**.
This requires **multiple** MDX-style ONNX models (at minimum: Vocals, Drums, Bass). Configure them via
**Configure ONNX Models (4-Stem)...**. If you don't provide an "Other" model, the app computes:
$$\text{Other} = \text{Mix} - (\text{Vocals} + \text{Drums} + \text{Bass})$$

### NVIDIA GPUs
Use **CUDA** backend:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Device Selection
When running AI separation (Demucs, 4-stem), a dialog will appear allowing you to choose:
- **CPU**: Works everywhere, slower
- **DirectML**: For AMD/Intel GPUs on Windows
- **CUDA**: For NVIDIA GPUs

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
