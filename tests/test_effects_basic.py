"""
Tests for basic audio effects.
"""
import pytest
import numpy as np

from src.core import effects_basic as fx
from src.core.config import AUDIO_CONFIG


class TestGain:
    """Tests for apply_gain effect."""
    
    def test_gain_doubles_amplitude(self, sample_mono_audio):
        result = fx.apply_gain(sample_mono_audio, factor=2.0)
        assert np.allclose(result, sample_mono_audio * 2.0)
    
    def test_gain_halves_amplitude(self, sample_mono_audio):
        result = fx.apply_gain(sample_mono_audio, factor=0.5)
        assert np.allclose(result, sample_mono_audio * 0.5)
    
    def test_gain_preserves_shape(self, sample_stereo_audio):
        result = fx.apply_gain(sample_stereo_audio, factor=1.5)
        assert result.shape == sample_stereo_audio.shape
    
    def test_gain_preserves_dtype(self, sample_mono_audio):
        result = fx.apply_gain(sample_mono_audio, factor=2.0)
        assert result.dtype == np.float32


class TestFades:
    """Tests for fade in/out effects."""
    
    def test_fade_in_starts_at_zero(self, sample_mono_audio):
        result = fx.apply_fade_in(sample_mono_audio)
        assert result[0] == 0.0
    
    def test_fade_in_ends_at_original(self, sample_mono_audio):
        result = fx.apply_fade_in(sample_mono_audio)
        assert np.isclose(result[-1], sample_mono_audio[-1], atol=1e-5)
    
    def test_fade_out_starts_at_original(self, sample_mono_audio):
        result = fx.apply_fade_out(sample_mono_audio)
        assert np.isclose(result[0], sample_mono_audio[0], atol=1e-5)
    
    def test_fade_out_ends_at_zero(self, sample_mono_audio):
        result = fx.apply_fade_out(sample_mono_audio)
        assert result[-1] == 0.0
    
    def test_fade_preserves_stereo_shape(self, sample_stereo_audio):
        result_in = fx.apply_fade_in(sample_stereo_audio)
        result_out = fx.apply_fade_out(sample_stereo_audio)
        assert result_in.shape == sample_stereo_audio.shape
        assert result_out.shape == sample_stereo_audio.shape


class TestNormalize:
    """Tests for normalize effect."""
    
    def test_normalize_reaches_target_peak(self, sample_mono_audio):
        target = 0.95
        result = fx.apply_normalize(sample_mono_audio, target_peak=target)
        assert np.isclose(np.max(np.abs(result)), target, atol=1e-5)
    
    def test_normalize_handles_silence(self):
        silence = np.zeros(1000, dtype=np.float32)
        result = fx.apply_normalize(silence)
        assert np.allclose(result, 0.0)


class TestFilters:
    """Tests for filter effects."""
    
    def test_lowpass_preserves_shape(self, sample_stereo_audio):
        sr = AUDIO_CONFIG.default_samplerate
        result = fx.apply_lowpass(sample_stereo_audio, sr, cutoff=1000)
        assert result.shape == sample_stereo_audio.shape
    
    def test_highpass_preserves_shape(self, sample_stereo_audio):
        sr = AUDIO_CONFIG.default_samplerate
        result = fx.apply_highpass(sample_stereo_audio, sr, cutoff=200)
        assert result.shape == sample_stereo_audio.shape
    
    def test_lowpass_attenuates_high_frequencies(self, sample_mono_audio):
        sr = AUDIO_CONFIG.default_samplerate
        # Low cutoff should attenuate our 440Hz test signal
        result = fx.apply_lowpass(sample_mono_audio, sr, cutoff=100)
        # Output should have lower RMS than input
        assert np.sqrt(np.mean(result**2)) < np.sqrt(np.mean(sample_mono_audio**2))


class TestDelay:
    """Tests for delay effect."""
    
    def test_delay_preserves_length(self, sample_mono_audio):
        sr = AUDIO_CONFIG.default_samplerate
        result = fx.apply_delay(sample_mono_audio, sr, delay_ms=100)
        assert len(result) == len(sample_mono_audio)
    
    def test_delay_too_long_returns_original(self, sample_mono_audio):
        sr = AUDIO_CONFIG.default_samplerate
        # Delay longer than audio
        result = fx.apply_delay(sample_mono_audio, sr, delay_ms=5000)
        assert np.allclose(result, sample_mono_audio)


class TestResample:
    """Tests for resample effect."""
    
    def test_resample_no_change(self, sample_mono_audio):
        result = fx.apply_resample(sample_mono_audio, factor=1.0)
        assert np.allclose(result, sample_mono_audio)
    
    def test_resample_faster_shortens(self, sample_mono_audio):
        result = fx.apply_resample(sample_mono_audio, factor=2.0)
        assert len(result) < len(sample_mono_audio)
    
    def test_resample_slower_lengthens(self, sample_mono_audio):
        result = fx.apply_resample(sample_mono_audio, factor=0.5)
        assert len(result) > len(sample_mono_audio)


class TestReverse:
    """Tests for reverse effect."""
    
    def test_reverse_flips_data(self, sample_mono_audio):
        result = fx.apply_reverse(sample_mono_audio)
        assert np.allclose(result, np.flip(sample_mono_audio))
    
    def test_reverse_twice_returns_original(self, sample_mono_audio):
        result = fx.apply_reverse(fx.apply_reverse(sample_mono_audio))
        assert np.allclose(result, sample_mono_audio)


class TestSoftClip:
    """Tests for soft clip effect."""
    
    def test_soft_clip_limits_amplitude(self):
        # Create signal with clipping
        loud = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
        result = fx.apply_soft_clip(loud)
        assert np.max(np.abs(result)) <= 1.0
    
    def test_soft_clip_preserves_quiet_signal(self, sample_mono_audio):
        # Our test signal is already within range
        quiet = sample_mono_audio * 0.5
        result = fx.apply_soft_clip(quiet)
        # Should be mostly unchanged
        assert np.allclose(result, quiet, atol=0.1)
