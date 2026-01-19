"""
Project abstraction for PyAudioEditor.
Encapsulates all project state (tracks, settings, undo history).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import numpy as np

from .config import AUDIO_CONFIG, UNDO_CONFIG
from .types import AudioArray, ClipboardData

if TYPE_CHECKING:
    from .track import AudioTrack
    from .undo_manager import UndoManager


@dataclass
class Project:
    """
    Represents an audio editing project.
    Contains all tracks, project settings, and undo history.
    """
    name: str = "Untitled Project"
    samplerate: int = field(default_factory=lambda: AUDIO_CONFIG.default_samplerate)
    tracks: list["AudioTrack"] = field(default_factory=list)
    custom_duration_samples: int = 0
    _undo_manager: Optional["UndoManager"] = field(default=None, repr=False)
    clipboard: Optional[ClipboardData] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self._undo_manager is None:
            from .undo_manager import UndoManager
            self._undo_manager = UndoManager(max_depth=UNDO_CONFIG.max_depth)
    
    @property
    def undo_manager(self) -> "UndoManager":
        """Lazy-loaded undo manager."""
        if self._undo_manager is None:
            from .undo_manager import UndoManager
            self._undo_manager = UndoManager(max_depth=UNDO_CONFIG.max_depth)
        return self._undo_manager
    
    @property
    def duration_samples(self) -> int:
        """Total project duration in samples (max of tracks and custom duration)."""
        max_track_dur = 0
        if self.tracks:
            max_track_dur = max(t.duration_samples for t in self.tracks)
        return max(max_track_dur, self.custom_duration_samples)
    
    @property
    def duration_seconds(self) -> float:
        """Total project duration in seconds."""
        if self.samplerate <= 0:
            return 0.0
        return self.duration_samples / self.samplerate
    
    @property
    def has_solo(self) -> bool:
        """Check if any track is soloed."""
        return any(t.soloed for t in self.tracks)
    
    def get_active_tracks(self) -> list["AudioTrack"]:
        """Get tracks that should be played (respecting solo/mute)."""
        if self.has_solo:
            return [t for t in self.tracks if t.soloed]
        return [t for t in self.tracks if not t.muted]
    
    def add_track(self, track: "AudioTrack") -> int:
        """Add a track and return its index."""
        self.tracks.append(track)
        return len(self.tracks) - 1
    
    def remove_track(self, index: int) -> Optional["AudioTrack"]:
        """Remove track at index and return it."""
        if 0 <= index < len(self.tracks):
            return self.tracks.pop(index)
        return None
    
    def get_track(self, index: int) -> Optional["AudioTrack"]:
        """Get track by index safely."""
        if 0 <= index < len(self.tracks):
            return self.tracks[index]
        return None
    
    def mix_down(self) -> Optional[AudioArray]:
        """
        Mix all active tracks to a single stereo array.
        Optimized with numpy vectorization.
        """
        if not self.tracks:
            return None
        
        max_len = self.duration_samples
        if max_len == 0:
            return None
        
        # Pre-allocate output buffer
        output = np.zeros((max_len, 2), dtype=np.float32)
        
        active_tracks = self.get_active_tracks()
        if not active_tracks:
            return output
        
        for track in active_tracks:
            if track.data is None:
                continue
            
            data = track.data
            track_len = len(data)
            offset = track.start_offset
            gain = track.gain
            
            # End of segment in output
            t_end = min(max_len, offset + track_len)
            d_end = t_end - offset # effective length to copy
            
            if d_end <= 0: continue
            
            # Convert mono to stereo if needed
            if data.ndim == 1:
                # Mono: apply to both channels
                output[offset:t_end, 0] += data[:d_end] * gain
                output[offset:t_end, 1] += data[:d_end] * gain
            else:
                # Stereo: direct addition with gain
                output[offset:t_end] += data[:d_end] * gain
        
        # Soft clip to prevent harsh digital distortion
        np.clip(output, -1.0, 1.0, out=output)
        
        return output
    
    def clear(self) -> None:
        """Reset project to empty state."""
        self.tracks.clear()
        self.undo_manager.clear()
        self.clipboard = None
        self.samplerate = AUDIO_CONFIG.default_samplerate
