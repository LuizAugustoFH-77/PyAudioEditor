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


from .clip import AudioClip

@dataclass(slots=True)
class AudioTrack:
    """
    Represents a single audio track containing multiple clips.
    
    Attributes:
        name: Display name of the track
        clips: List of AudioClip objects
        gain: Volume multiplier (0.0 to 2.0)
        pan: Stereo panning (-1.0 = left, 0.0 = center, 1.0 = right)
        muted: Whether track is muted
        soloed: Whether track is soloed
        samplerate: Sample rate in Hz
    """
    name: str = "Track"
    clips: list[AudioClip] = field(default_factory=list)
    gain: float = field(default_factory=lambda: AUDIO_CONFIG.default_gain)
    pan: float = 0.0
    muted: bool = False
    soloed: bool = False
    samplerate: int = field(default_factory=lambda: AUDIO_CONFIG.default_samplerate)
    
    # Legacy compatibility fields are handled via properties, no slots for them
    
    @property
    def start_offset(self) -> int:
        """Compatibility: Returns start offset of the first clip or 0."""
        return self.clips[0].start_offset if self.clips else 0
    
    @start_offset.setter
    def start_offset(self, value: int):
        """Compatibility: Moves all clips by relative amount."""
        if not self.clips:
            return
        old_start = self.clips[0].start_offset
        shift = value - old_start
        for clip in self.clips:
            new_pos = max(0, clip.start_offset + shift)
            clip.start_offset = new_pos
            
    @property
    def data(self) -> Optional[AudioArray]:
        """
        Compatibility: Returns a flattened mix of all clips.
        WARNING: This is expensive.
        """
        if not self.clips:
            return None
            
        max_end = 0
        min_start = float('inf')
        for clip in self.clips:
            max_end = max(max_end, clip.end_offset)
            min_start = min(min_start, clip.start_offset)
            
        if min_start == float('inf'):
            return None
            
        total_len = max_end - min_start
        if total_len <= 0:
            return None
            
        example_data = self.clips[0].data
        if example_data.ndim > 1:
            buffer = np.zeros((total_len, example_data.shape[1]), dtype=np.float32)
        else:
            buffer = np.zeros(total_len, dtype=np.float32)
            
        for clip in self.clips:
            rel_start = clip.start_offset - min_start
            length = len(clip.data)
            rel_end = rel_start + length
            if rel_start >= 0 and rel_end <= total_len:
                 buffer[rel_start:rel_end] = clip.data
                 
        return buffer

    @data.setter
    def data(self, value: AudioArray):
        """Compatibility: Sets data as a single clip starting at current offset."""
        if value is None:
            self.clips.clear()
            return
        # If there are existing clips, try to preserve the start offset of the first one
        # Otherwise default to 0
        current_offset = self.start_offset if self.clips else 0
        self.clips = [AudioClip(data=value, start_offset=current_offset, name="Main")]

    @property
    def splits(self) -> list[int]:
        """Compatibility: Returns empty list as splits are now implicit boundaries."""
        return []

    @splits.setter
    def splits(self, value):
        pass

    def set_data(self, data: AudioArray, samplerate: int) -> Self:
        """Set audio data and samplerate (Legacy wrapper)."""
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.samplerate = samplerate
        return self
    
    @property
    def duration_samples(self) -> int:
        """Total duration including all clips."""
        if not self.clips:
            return 0
        return max(c.end_offset for c in self.clips)
    
    @property
    def duration_seconds(self) -> float:
        if self.samplerate <= 0:
            return 0.0
        return self.duration_samples / self.samplerate
    
    @property
    def channels(self) -> int:
        if not self.clips:
            return 0
        example = self.clips[0].data
        return example.shape[1] if example.ndim > 1 else 1
    
    @property
    def is_stereo(self) -> bool:
        return self.channels == 2
    
    @property
    def is_mono(self) -> bool:
        return self.channels == 1
    
    def get_mono(self) -> Optional[AudioArray]:
        data = self.data
        if data is None: return None
        if self.is_mono: return data
        return np.mean(data, axis=1).astype(np.float32)
    
    def to_stereo(self) -> Optional[AudioArray]:
        data = self.data
        if data is None: return None
        if self.is_stereo: return data
        return np.column_stack((data, data))
    
    def get_segment(self, start_sample: int, end_sample: int) -> Optional[AudioArray]:
        # Expensive fallback
        data = self.data
        if data is None: return None
        
        # 'data' property returns array starting at min_start (virtual 0 for the content)
        # But get_segment usually expects global timeline samples if start_offset was involved? 
        # Actually AudioTrack.start_offset was "virtual start".
        # Let's assume get_segment wants data from the track's internal buffer 0 to N.
        # But wait, external callers pass GLOBAL timeline samples usually?
        # Re-reading `waveform_view.py` might clarify. 
        # Usually standard is: global timeline sample -> subtract start_offset -> index into data.
        
        # In multi-clip world:
        # We need to render the mixed clips.
        # For simplicity, let's just use the flattened data.
        # But we need to handle the offset correctly.
        
        # 'data' property synthesizes from min_start. 
        # So effective global start is min_start.
        
        min_start = self.start_offset
        rel_start = start_sample - min_start
        rel_end = end_sample - min_start
        
        if rel_start < 0: rel_start = 0
        if rel_end > len(data): rel_end = len(data)
        if rel_start >= rel_end: return None
        
        return data[rel_start:rel_end]
        
    def split_at(self, sample_index: int) -> bool:
        """Splits the clip under the cursor."""
        for i, clip in enumerate(self.clips):
            if clip.start_offset < sample_index < clip.end_offset:
                local_split = sample_index - clip.start_offset
                
                left_data = clip.data[:local_split].copy()
                right_data = clip.data[local_split:].copy()
                
                left_clip = AudioClip(left_data, clip.start_offset, f"{clip.name}_L")
                right_clip = AudioClip(right_data, sample_index, f"{clip.name}_R")
                
                self.clips.pop(i)
                self.clips.insert(i, left_clip)
                self.clips.insert(i+1, right_clip)
                return True
        return False
    
    def remove_split(self, sample_index: int) -> bool:
        # Merging clips is harder logic, skipping for now as it wasn't requested
        return False
    
    def clear_splits(self) -> None:
        pass
    
    def move_segment(self, old_start: int, old_end: int, new_start: int) -> bool:
        return True
    
    def copy(self) -> AudioTrack:
        new_track = AudioTrack(
            name=self.name,
            gain=self.gain,
            pan=self.pan,
            muted=self.muted,
            soloed=self.soloed,
            samplerate=self.samplerate,
            clips=[c.copy() for c in self.clips]
        )
        return new_track
    
    def __repr__(self) -> str:
        return f"AudioTrack(name={self.name!r}, clips={len(self.clips)})"
