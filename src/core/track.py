"""
Audio track representation for PyAudioEditor.
Uses dataclass for cleaner initialization and slots for memory efficiency.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Self
import numpy as np

from .types import AudioArray
from .config import AUDIO_CONFIG


@dataclass(slots=True)
class AudioTrack:
    """
    Represents a single audio track with its data, metadata, and state.
    
    Attributes:
        name: Display name of the track
        data: Audio samples as numpy array (samples,) or (samples, channels)
        gain: Volume multiplier (0.0 to 2.0)
        pan: Stereo panning (-1.0 = left, 0.0 = center, 1.0 = right)
        muted: Whether track is muted
        soloed: Whether track is soloed
        samplerate: Sample rate in Hz
        splits: List of sample indices where track is split
    """
    name: str = "Track"
    data: Optional[AudioArray] = None
    gain: float = field(default_factory=lambda: AUDIO_CONFIG.default_gain)
    pan: float = 0.0
    muted: bool = False
    soloed: bool = False
    samplerate: int = field(default_factory=lambda: AUDIO_CONFIG.default_samplerate)
    splits: list[int] = field(default_factory=list)
    
    def set_data(self, data: AudioArray, samplerate: int) -> Self:
        """
        Set audio data and samplerate.
        Returns self for fluent API chaining.
        """
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.samplerate = samplerate
        return self
    
    @property
    def duration_samples(self) -> int:
        """Total number of samples in the track."""
        return len(self.data) if self.data is not None else 0
    
    @property
    def duration_seconds(self) -> float:
        """Duration of the track in seconds."""
        if self.samplerate <= 0:
            return 0.0
        return self.duration_samples / self.samplerate
    
    @property
    def channels(self) -> int:
        """Number of audio channels (1=mono, 2=stereo)."""
        if self.data is None:
            return 0
        return self.data.shape[1] if self.data.ndim > 1 else 1
    
    @property
    def is_stereo(self) -> bool:
        """Check if track is stereo."""
        return self.channels == 2
    
    @property
    def is_mono(self) -> bool:
        """Check if track is mono."""
        return self.channels == 1
    
    def get_mono(self) -> Optional[AudioArray]:
        """Get mono version of track data."""
        if self.data is None:
            return None
        if self.is_mono:
            return self.data
        # Average channels for stereo -> mono
        return np.mean(self.data, axis=1).astype(np.float32)
    
    def to_stereo(self) -> Optional[AudioArray]:
        """Convert track data to stereo."""
        if self.data is None:
            return None
        if self.is_stereo:
            return self.data
        # Duplicate mono channel
        return np.column_stack((self.data, self.data))
    
    def get_segment(self, start_sample: int, end_sample: int) -> Optional[AudioArray]:
        """Get a segment of audio data."""
        if self.data is None:
            return None
        start = max(0, start_sample)
        end = min(self.duration_samples, end_sample)
        if start >= end:
            return None
        return self.data[start:end].copy()
    
    def add_split(self, sample_index: int) -> bool:
        """Add a split point if valid."""
        if self.data is None:
            return False
        if not (0 < sample_index < self.duration_samples):
            return False
        if sample_index in self.splits:
            return False
        self.splits.append(sample_index)
        self.splits.sort()
        return True
    
    def remove_split(self, sample_index: int) -> bool:
        """Remove a split point."""
        if sample_index in self.splits:
            self.splits.remove(sample_index)
            return True
        return False
    
    def clear_splits(self) -> None:
        """Remove all split points."""
        self.splits.clear()
    
    def copy(self) -> AudioTrack:
        """Create a deep copy of the track."""
        return AudioTrack(
            name=self.name,
            data=self.data.copy() if self.data is not None else None,
            gain=self.gain,
            pan=self.pan,
            muted=self.muted,
            soloed=self.soloed,
            samplerate=self.samplerate,
            splits=self.splits.copy()
        )
    
    def __repr__(self) -> str:
        duration = f"{self.duration_seconds:.2f}s" if self.data is not None else "empty"
        return f"AudioTrack(name={self.name!r}, duration={duration}, sr={self.samplerate})"
