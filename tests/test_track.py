"""
Tests for AudioTrack.
"""
import pytest
import numpy as np

from src.core.track import AudioTrack
from src.core.config import AUDIO_CONFIG


class TestAudioTrack:
    """Tests for AudioTrack functionality."""
    
    def test_default_initialization(self):
        track = AudioTrack()
        assert track.name == "Track"
        assert track.data is None
        assert track.gain == AUDIO_CONFIG.default_gain
        assert track.pan == 0.0
        assert not track.muted
        assert not track.soloed
        assert track.splits == []
    
    def test_custom_initialization(self):
        track = AudioTrack(name="My Track", gain=0.8, muted=True)
        assert track.name == "My Track"
        assert track.gain == 0.8
        assert track.muted
    
    def test_set_data(self, sample_stereo_audio):
        track = AudioTrack()
        result = track.set_data(sample_stereo_audio, 44100)
        
        assert track.data is not None
        assert track.samplerate == 44100
        assert result is track  # Fluent API
    
    def test_duration_samples(self, sample_track):
        expected = AUDIO_CONFIG.default_samplerate  # 1 second of audio
        assert sample_track.duration_samples == expected
    
    def test_duration_seconds(self, sample_track):
        assert np.isclose(sample_track.duration_seconds, 1.0, atol=0.01)
    
    def test_channels_stereo(self, sample_track):
        assert sample_track.channels == 2
        assert sample_track.is_stereo
        assert not sample_track.is_mono
    
    def test_channels_mono(self, sample_mono_audio):
        track = AudioTrack()
        track.set_data(sample_mono_audio, 44100)
        assert track.channels == 1
        assert track.is_mono
        assert not track.is_stereo
    
    def test_get_mono(self, sample_track):
        mono = sample_track.get_mono()
        assert mono is not None
        assert mono.ndim == 1
        assert len(mono) == sample_track.duration_samples
    
    def test_to_stereo_from_mono(self, sample_mono_audio):
        track = AudioTrack()
        track.set_data(sample_mono_audio, 44100)
        
        stereo = track.to_stereo()
        assert stereo is not None
        assert stereo.ndim == 2
        assert stereo.shape[1] == 2
    
    def test_get_segment(self, sample_track):
        segment = sample_track.get_segment(0, 1000)
        assert segment is not None
        assert len(segment) == 1000
    
    def test_get_segment_clamped(self, sample_track):
        # Request beyond bounds
        total = sample_track.duration_samples
        segment = sample_track.get_segment(total - 100, total + 1000)
        assert segment is not None
        assert len(segment) == 100
    
    def test_add_split(self, sample_track):
        assert sample_track.add_split(1000)
        assert 1000 in sample_track.splits
    
    def test_add_split_sorted(self, sample_track):
        sample_track.add_split(5000)
        sample_track.add_split(1000)
        sample_track.add_split(3000)
        assert sample_track.splits == [1000, 3000, 5000]
    
    def test_add_split_duplicate_rejected(self, sample_track):
        sample_track.add_split(1000)
        assert not sample_track.add_split(1000)
    
    def test_add_split_invalid_rejected(self, sample_track):
        assert not sample_track.add_split(0)  # At start
        assert not sample_track.add_split(sample_track.duration_samples)  # At end
        assert not sample_track.add_split(-100)  # Negative
    
    def test_remove_split(self, sample_track):
        sample_track.add_split(1000)
        assert sample_track.remove_split(1000)
        assert 1000 not in sample_track.splits
    
    def test_clear_splits(self, sample_track):
        sample_track.add_split(1000)
        sample_track.add_split(2000)
        sample_track.clear_splits()
        assert sample_track.splits == []
    
    def test_copy(self, sample_track):
        sample_track.add_split(1000)
        copy = sample_track.copy()
        
        assert copy.name == sample_track.name
        assert copy.gain == sample_track.gain
        assert copy.splits == sample_track.splits
        assert copy.data is not sample_track.data  # Deep copy
        assert np.allclose(copy.data, sample_track.data)
    
    def test_repr(self, sample_track):
        rep = repr(sample_track)
        assert "Test Track" in rep
        assert "1.00s" in rep
