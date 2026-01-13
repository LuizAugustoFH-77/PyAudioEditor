"""
Basic audio effects for PyAudioEditor.
All functions are pure (no side effects) and operate on numpy arrays.
Optimized with numpy vectorization for performance.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

from .types import AudioArray
from .config import EFFECTS_CONFIG


def apply_gain(data: AudioArray, factor: float = 2.0) -> AudioArray:
    """
    Multiply audio data by a gain factor.
    
    Args:
        data: Audio samples
        factor: Gain multiplier (1.0 = no change)
    
    Returns:
        Gained audio data
    """
    return (data * factor).astype(np.float32)


def apply_normalize(data: AudioArray, target_peak: float = 0.95) -> AudioArray:
    """
    Normalize audio to target peak level.
    
    Args:
        data: Audio samples
        target_peak: Target peak amplitude (0.0 to 1.0)
    
    Returns:
        Normalized audio data
    """
    peak = np.max(np.abs(data))
    if peak < 1e-6:  # Avoid division by zero
        return data
    return (data * (target_peak / peak)).astype(np.float32)


def apply_fade_in(data: AudioArray, duration_samples: int | None = None) -> AudioArray:
    """
    Apply linear fade-in to audio data.
    
    Args:
        data: Audio samples
        duration_samples: Fade duration in samples (None = full length)
    
    Returns:
        Faded audio data
    """
    length = len(data)
    if length == 0:
        return data
    
    fade_len = duration_samples if duration_samples else length
    fade_len = min(fade_len, length)
    
    # Create fade curve efficiently
    fade_curve = np.ones(length, dtype=np.float32)
    fade_curve[:fade_len] = np.linspace(0, 1, fade_len, dtype=np.float32)
    
    if data.ndim > 1:
        fade_curve = fade_curve[:, np.newaxis]
    
    return (data * fade_curve).astype(np.float32)


def apply_fade_out(data: AudioArray, duration_samples: int | None = None) -> AudioArray:
    """
    Apply linear fade-out to audio data.
    
    Args:
        data: Audio samples
        duration_samples: Fade duration in samples (None = full length)
    
    Returns:
        Faded audio data
    """
    length = len(data)
    if length == 0:
        return data
    
    fade_len = duration_samples if duration_samples else length
    fade_len = min(fade_len, length)
    
    # Create fade curve efficiently
    fade_curve = np.ones(length, dtype=np.float32)
    fade_curve[-fade_len:] = np.linspace(1, 0, fade_len, dtype=np.float32)
    
    if data.ndim > 1:
        fade_curve = fade_curve[:, np.newaxis]
    
    return (data * fade_curve).astype(np.float32)


def apply_delay(
    data: AudioArray, 
    sr: int, 
    delay_ms: float = EFFECTS_CONFIG.delay_ms, 
    decay: float = EFFECTS_CONFIG.delay_decay,
    feedback: bool = True
) -> AudioArray:
    """
    Apply delay/echo effect.
    
    Args:
        data: Audio samples
        sr: Sample rate
        delay_ms: Delay time in milliseconds
        decay: Volume decay per echo (0.0 to 1.0)
        feedback: Whether to use feedback (multiple echoes)
    
    Returns:
        Audio with delay effect
    """
    delay_samples = int(sr * (delay_ms / 1000.0))
    if delay_samples <= 0 or delay_samples >= len(data):
        return data
    
    output = data.copy()
    
    if feedback:
        # Feedback delay (each echo feeds into next)
        for i in range(delay_samples, len(data)):
            output[i] += output[i - delay_samples] * decay
    else:
        # Simple delay (only original echoes)
        output[delay_samples:] += data[:-delay_samples] * decay
    
    return output.astype(np.float32)


def apply_reverb(
    data: AudioArray, 
    sr: int, 
    room_size: float = EFFECTS_CONFIG.reverb_room_size,
    damping: float = 0.5
) -> AudioArray:
    """
    Apply simple reverb using parallel comb filters.
    
    Args:
        data: Audio samples
        sr: Sample rate
        room_size: Room size factor (0.0 to 1.0)
        damping: High frequency damping
    
    Returns:
        Audio with reverb effect
    """
    # Delay times scaled by room size (in ms)
    delays = [
        (25 * room_size, 0.30),
        (37 * room_size, 0.25),
        (53 * room_size, 0.20),
        (71 * room_size, 0.15),
        (97 * room_size, 0.10),
    ]
    
    combined = data.copy()
    total_gain = 1.0
    
    for delay_ms, gain in delays:
        delay_samples = int(sr * (delay_ms / 1000.0))
        if delay_samples < len(data) and delay_samples > 0:
            combined[delay_samples:] += data[:-delay_samples] * gain
            total_gain += gain
    
    # Normalize to prevent clipping
    return (combined / total_gain).astype(np.float32)


def apply_lowpass(
    data: AudioArray, 
    sr: int, 
    cutoff: float = EFFECTS_CONFIG.lowpass_cutoff,
    order: int = 2
) -> AudioArray:
    """
    Apply Butterworth low-pass filter.
    
    Args:
        data: Audio samples
        sr: Sample rate
        cutoff: Cutoff frequency in Hz
        order: Filter order
    
    Returns:
        Filtered audio data
    """
    nyquist = 0.5 * sr
    normal_cutoff = np.clip(cutoff / nyquist, 0.001, 0.999)
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data, axis=0).astype(np.float32)


def apply_highpass(
    data: AudioArray, 
    sr: int, 
    cutoff: float = EFFECTS_CONFIG.highpass_cutoff,
    order: int = 2
) -> AudioArray:
    """
    Apply Butterworth high-pass filter.
    
    Args:
        data: Audio samples
        sr: Sample rate
        cutoff: Cutoff frequency in Hz
        order: Filter order
    
    Returns:
        Filtered audio data
    """
    nyquist = 0.5 * sr
    normal_cutoff = np.clip(cutoff / nyquist, 0.001, 0.999)
    
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data, axis=0).astype(np.float32)


def apply_bandpass(
    data: AudioArray, 
    sr: int, 
    low_cutoff: float = 200.0,
    high_cutoff: float = 5000.0,
    order: int = 2
) -> AudioArray:
    """
    Apply Butterworth band-pass filter.
    
    Args:
        data: Audio samples
        sr: Sample rate
        low_cutoff: Low cutoff frequency in Hz
        high_cutoff: High cutoff frequency in Hz
        order: Filter order
    
    Returns:
        Filtered audio data
    """
    nyquist = 0.5 * sr
    low = np.clip(low_cutoff / nyquist, 0.001, 0.999)
    high = np.clip(high_cutoff / nyquist, low + 0.001, 0.999)
    
    b, a = butter(order, [low, high], btype='band', analog=False)
    return lfilter(b, a, data, axis=0).astype(np.float32)


def apply_low_shelf(
    data: AudioArray, 
    sr: int, 
    cutoff: float = 200.0, 
    gain_db: float = 6.0, 
    Q: float = 0.707
) -> AudioArray:
    """
    Apply low-shelf EQ filter.
    
    Args:
        data: Audio samples
        sr: Sample rate
        cutoff: Shelf frequency in Hz
        gain_db: Gain in dB (positive = boost, negative = cut)
        Q: Q factor for shelf shape
    
    Returns:
        Filtered audio data
    """
    import math
    
    A = 10 ** (gain_db / 40)
    omega = 2 * math.pi * cutoff / sr
    sn, cs = math.sin(omega), math.cos(omega)
    alpha = sn / (2 * Q)
    
    b0 = A * ((A + 1) - (A - 1) * cs + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cs)
    b2 = A * ((A + 1) - (A - 1) * cs - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cs + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cs)
    a2 = (A + 1) + (A - 1) * cs - 2 * math.sqrt(A) * alpha
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    return lfilter(b, a, data, axis=0).astype(np.float32)


def apply_high_shelf(
    data: AudioArray, 
    sr: int, 
    cutoff: float = 8000.0, 
    gain_db: float = 3.0, 
    Q: float = 0.707
) -> AudioArray:
    """
    Apply high-shelf EQ filter.
    
    Args:
        data: Audio samples
        sr: Sample rate
        cutoff: Shelf frequency in Hz
        gain_db: Gain in dB (positive = boost, negative = cut)
        Q: Q factor for shelf shape
    
    Returns:
        Filtered audio data
    """
    import math
    
    A = 10 ** (gain_db / 40)
    omega = 2 * math.pi * cutoff / sr
    sn, cs = math.sin(omega), math.cos(omega)
    alpha = sn / (2 * Q)
    
    b0 = A * ((A + 1) + (A - 1) * cs + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cs)
    b2 = A * ((A + 1) + (A - 1) * cs - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cs + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cs)
    a2 = (A + 1) - (A - 1) * cs - 2 * math.sqrt(A) * alpha
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    return lfilter(b, a, data, axis=0).astype(np.float32)


def apply_peaking_eq(
    data: AudioArray, 
    sr: int, 
    frequency: float = 1000.0, 
    gain_db: float = 0.0, 
    Q: float = 1.0
) -> AudioArray:
    """
    Apply parametric peaking EQ band.
    
    Args:
        data: Audio samples
        sr: Sample rate
        frequency: Center frequency in Hz
        gain_db: Gain in dB (positive = boost, negative = cut)
        Q: Q factor (bandwidth control)
    
    Returns:
        Filtered audio data
    """
    import math
    
    if abs(gain_db) < 0.01:  # Skip if negligible gain
        return data
    
    A = 10 ** (gain_db / 40.0)
    omega = 2 * math.pi * frequency / sr
    sn = math.sin(omega)
    cs = math.cos(omega)
    alpha = sn / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * cs
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cs
    a2 = 1 - alpha / A
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    return lfilter(b, a, data, axis=0).astype(np.float32)


def apply_soft_clip(data: AudioArray, threshold: float = 0.85) -> AudioArray:
    """
    Apply soft saturation to prevent harsh digital clipping.
    Uses cubic approximation for warm sound.
    
    Args:
        data: Audio samples
        threshold: Clipping threshold (0.0 to 1.0)
    
    Returns:
        Soft-clipped audio data
    """
    out = np.clip(data, -1.5, 1.5)
    out = out - (out ** 3) / 6.0
    
    # Normalize to ensure peak is at 1.0
    max_val = np.max(np.abs(out))
    if max_val > 1.0:
        out /= max_val
    
    return out.astype(np.float32)


def apply_resample(
    data: AudioArray, 
    factor: float = 1.0,
    kind: str = 'linear'
) -> AudioArray:
    """
    Change speed and pitch by resampling.
    
    Args:
        data: Audio samples
        factor: Speed factor (>1 = faster/higher, <1 = slower/lower)
        kind: Interpolation type ('linear', 'cubic', 'quadratic')
    
    Returns:
        Resampled audio data
    """
    if abs(factor - 1.0) < 0.001:
        return data
    
    length = len(data)
    new_length = int(length / factor)
    
    if new_length <= 0:
        return data
    
    x = np.linspace(0, length - 1, length)
    x_new = np.linspace(0, length - 1, new_length)
    
    if data.ndim > 1:
        # Stereo
        new_data = np.zeros((new_length, data.shape[1]), dtype=np.float32)
        for ch in range(data.shape[1]):
            f = interp1d(x, data[:, ch], kind=kind, fill_value="extrapolate")
            new_data[:, ch] = f(x_new)
        return new_data
    else:
        # Mono
        f = interp1d(x, data, kind=kind, fill_value="extrapolate")
        return f(x_new).astype(np.float32)


def apply_reverse(data: AudioArray) -> AudioArray:
    """
    Reverse audio data.
    
    Args:
        data: Audio samples
    
    Returns:
        Reversed audio data
    """
    return np.flip(data, axis=0).copy().astype(np.float32)


def apply_invert(data: AudioArray) -> AudioArray:
    """
    Invert audio polarity (phase inversion).
    
    Args:
        data: Audio samples
    
    Returns:
        Inverted audio data
    """
    return (-data).astype(np.float32)


def apply_silence(length_samples: int, channels: int = 2) -> AudioArray:
    """
    Generate silence.
    
    Args:
        length_samples: Duration in samples
        channels: Number of channels
    
    Returns:
        Silent audio array
    """
    if channels == 1:
        return np.zeros(length_samples, dtype=np.float32)
    return np.zeros((length_samples, channels), dtype=np.float32)


def apply_bitcrush(
    data: AudioArray, 
    bits: int = 12, 
    downsample: int = 1, 
    mix: float = 0.5
) -> AudioArray:
    """
    Apply bitcrusher effect for lo-fi/digital distortion.
    
    Args:
        data: Audio samples
        bits: Bit depth (1-16)
        downsample: Downsample factor
        mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
    
    Returns:
        Bitcrushed audio data
    """
    bits = np.clip(bits, 1, 16)
    levels = 2 ** bits
    
    # Quantize
    crushed = np.round(data * levels) / levels
    
    # Downsample (sample-and-hold)
    if downsample > 1:
        for i in range(0, len(crushed), downsample):
            crushed[i:i + downsample] = crushed[i]
    
    # Mix
    result = (1 - mix) * data + mix * crushed
    
    return result.astype(np.float32)
