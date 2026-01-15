"""
Centralized configuration for PyAudioEditor.
All magic numbers and default settings in one place.
"""
from dataclasses import dataclass
from enum import Enum, auto


class PlaybackState(Enum):
    """Playback state enumeration."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Audio engine configuration."""
    default_samplerate: int = 44100
    playback_blocksize: int = 4096
    playback_channels: int = 2
    playback_end_margin_samples: int = 22050  # ~0.5s buffer at end
    max_gain: float = 2.0
    default_gain: float = 1.0


@dataclass(frozen=True, slots=True)
class SpectrogramConfig:
    """Spectrogram visualization settings."""
    n_fft: int = 2048
    hop_length: int = 512
    max_duration_for_full_stft: float = 300.0  # 5 minutes


@dataclass(frozen=True, slots=True)
class WaveformConfig:
    """Waveform visualization settings."""
    min_visible_samples: int = 100
    default_color: tuple[int, int, int] = (44, 199, 201)  # Teal
    playhead_color: tuple[int, int, int] = (240, 79, 90)  # Warm red
    selection_alpha: int = 80
    downsample_threshold: int = 10000  # Use downsampling above this


@dataclass(frozen=True, slots=True)
class UndoConfig:
    """Undo/Redo configuration."""
    max_depth: int = 50
    

@dataclass(frozen=True, slots=True)
class EffectsConfig:
    """Default effect parameters."""
    # Delay/Echo
    delay_ms: float = 300.0
    delay_decay: float = 0.4
    
    # Reverb
    reverb_room_size: float = 0.5
    
    # Filters
    lowpass_cutoff: float = 1000.0
    highpass_cutoff: float = 100.0
    
    # Compression
    compressor_threshold_db: float = -20.0
    compressor_ratio: float = 4.0
    compressor_attack_ms: float = 5.0
    compressor_release_ms: float = 100.0
    


# Global config instances (immutable singletons)
AUDIO_CONFIG = AudioConfig()
SPECTROGRAM_CONFIG = SpectrogramConfig()
WAVEFORM_CONFIG = WaveformConfig()
UNDO_CONFIG = UndoConfig()
EFFECTS_CONFIG = EffectsConfig()
