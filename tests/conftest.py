"""
Pytest configuration and fixtures for PyAudioEditor tests.
"""
import pytest
import numpy as np
from typing import Generator

from src.core.track import AudioTrack
from src.core.project import Project
from src.core.undo_manager import UndoManager
from src.core.config import AUDIO_CONFIG


@pytest.fixture
def sample_mono_audio() -> np.ndarray:
    """Generate 1 second of mono sine wave audio."""
    sr = AUDIO_CONFIG.default_samplerate
    t = np.linspace(0, 1, sr, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def sample_stereo_audio() -> np.ndarray:
    """Generate 1 second of stereo sine wave audio."""
    sr = AUDIO_CONFIG.default_samplerate
    t = np.linspace(0, 1, sr, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    return np.column_stack((left, right))


@pytest.fixture
def sample_track(sample_stereo_audio) -> AudioTrack:
    """Create a sample audio track."""
    track = AudioTrack(name="Test Track")
    track.set_data(sample_stereo_audio, AUDIO_CONFIG.default_samplerate)
    return track


@pytest.fixture
def empty_project() -> Project:
    """Create an empty project."""
    return Project(name="Test Project")


@pytest.fixture
def project_with_track(sample_track) -> Project:
    """Create a project with one track."""
    project = Project(name="Test Project")
    project.add_track(sample_track)
    return project


@pytest.fixture
def undo_manager() -> UndoManager:
    """Create an undo manager."""
    return UndoManager(max_depth=10)
