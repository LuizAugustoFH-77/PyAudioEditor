from dataclasses import dataclass
import numpy as np

@dataclass
class AudioClip:
    """
    Represents a single segment of audio within a track.
    Can be moved independently in time.
    """
    data: np.ndarray
    start_offset: int  # Sample offset relative to the timeline start
    name: str = ""
    
    @property
    def end_offset(self) -> int:
        return self.start_offset + len(self.data)
    
    @property
    def length(self) -> int:
        return len(self.data)
    
    def copy(self) -> 'AudioClip':
        return AudioClip(
            data=self.data.copy(),
            start_offset=self.start_offset,
            name=self.name
        )
