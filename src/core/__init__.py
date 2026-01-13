"""
PyAudioEditor Core Module

This module contains the core audio processing logic:
- AudioEngine: Main orchestrator for audio operations
- Project: Project state management
- AudioTrack: Track representation
- PlaybackController: Audio playback handling
- Effects: Audio processing effects
- Separation: AI-powered vocal separation
"""
from .audio_engine import AudioEngine
from .project import Project
from .track import AudioTrack
from .playback import PlaybackController
from .undo_manager import UndoManager
from .config import (
    AUDIO_CONFIG,
    EFFECTS_CONFIG,
    SPECTROGRAM_CONFIG,
    WAVEFORM_CONFIG,
    UNDO_CONFIG,
    PlaybackState
)
from . import effects_basic
from . import effects_vocal
from . import separation

__all__ = [
    # Main classes
    'AudioEngine',
    'Project', 
    'AudioTrack',
    'PlaybackController',
    'UndoManager',
    # Config
    'AUDIO_CONFIG',
    'EFFECTS_CONFIG',
    'SPECTROGRAM_CONFIG',
    'WAVEFORM_CONFIG',
    'UNDO_CONFIG',
    'PlaybackState',
    # Submodules
    'effects_basic',
    'effects_vocal',
    'separation',
]
