"""
Type definitions for the PyAudioEditor core module.
Provides type aliases and protocols for type safety and better IDE support.
"""
from typing import Any, Callable, Optional, Protocol, TypeVar
import numpy as np
from numpy.typing import NDArray

# Audio data types
AudioArray = NDArray[np.float32]  # Shape: (samples,) or (samples, channels)
MonoArray = NDArray[np.float32]   # Shape: (samples,)
StereoArray = NDArray[np.float32] # Shape: (samples, 2)

# Callback types
UndoFunc = Callable[[], None]
RedoFunc = Callable[[], None]
ProgressCallback = Callable[[int, int, str], None]  # (current, total, status)

# Clipboard data structure
ClipboardData = dict[str, Any]

# Generic type for effect functions
T = TypeVar('T', bound=AudioArray)


class EffectFunc(Protocol):
    """Protocol for effect functions that process audio data."""
    def __call__(self, data: AudioArray, sr: int, **kwargs: Any) -> AudioArray: ...


class SeparationResult:
    """Result from vocal separation operations."""
    __slots__ = ('success', 'tracks', 'error')
    
    def __init__(
        self, 
        success: bool, 
        tracks: Optional[list[tuple[str, AudioArray]]] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.tracks = tracks or []
        self.error = error
