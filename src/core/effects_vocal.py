"""
Advanced vocal processing effects for PyAudioEditor.
Includes pitch shifting, formant manipulation, compression, and vocal presets.
All functions are pure and operate on numpy arrays.
"""
from __future__ import annotations
import logging
import math
import shutil
import subprocess
import numpy as np
from scipy.signal import butter, lfilter

from .types import AudioArray
from .config import EFFECTS_CONFIG
from .effects_basic import (
    apply_highpass, apply_peaking_eq, apply_high_shelf, apply_soft_clip
)

logger = logging.getLogger("PyAudacity")


def apply_pitch_shift(
    data: AudioArray, 
    sr: int, 
    semitones: float = 0.0
) -> AudioArray:
    """
    Pitch shift without changing tempo (time-preserving).
    Uses librosa as fallback; pyrubberband if available for better quality.
    
    Args:
        data: Audio samples
        sr: Sample rate
        semitones: Pitch shift in semitones (+/- 12 range typical)
    
    Returns:
        Pitch-shifted audio data
    """
    if abs(semitones) < 0.01:
        return data
    
    # Try pyrubberband first (better quality)
    try:
        import pyrubberband as pyrb

        # pyrubberband requires the external rubberband CLI executable.
        # On Windows this is usually rubberband.exe and must be on PATH.
        if shutil.which("rubberband") is None and shutil.which("rubberband.exe") is None:
            raise FileNotFoundError(
                "rubberband-cli not found on PATH (rubberband/rubberband.exe)."
            )

        if data.ndim > 1:
            shifted = np.zeros_like(data)
            for ch in range(data.shape[1]):
                shifted[:, ch] = pyrb.pitch_shift(data[:, ch], sr, semitones)
            return shifted.astype(np.float32)
        else:
            return pyrb.pitch_shift(data, sr, semitones).astype(np.float32)
    except (ImportError, FileNotFoundError, OSError, subprocess.CalledProcessError, RuntimeError) as e:
        # ImportError: pyrubberband not installed
        # FileNotFoundError/OSError: rubberband executable missing / spawn failure
        # CalledProcessError/RuntimeError: rubberband execution failure
        logger.debug("pyrubberband pitch shift unavailable, falling back: %s", e)
    
    # Fallback to librosa
    try:
        import librosa
        if data.ndim > 1:
            shifted = np.zeros_like(data)
            for ch in range(data.shape[1]):
                shifted[:, ch] = librosa.effects.pitch_shift(
                    data[:, ch], sr=sr, n_steps=semitones
                )
            return shifted.astype(np.float32)
        else:
            return librosa.effects.pitch_shift(data, sr=sr, n_steps=semitones).astype(np.float32)
    except ImportError:
        logger.warning("Neither pyrubberband nor librosa available for pitch shift")
        return data
    except Exception as e:
        logger.error("Librosa pitch shift failed: %s", e, exc_info=True)
        return data


def apply_formant_shift(
    data: AudioArray, 
    sr: int, 
    shift_ratio: float = 1.0
) -> AudioArray:
    """
    Shift formants (vocal character) without changing pitch.
    Uses parselmouth (Praat) if available, otherwise approximates with EQ.
    
    Args:
        data: Audio samples
        sr: Sample rate
        shift_ratio: Formant shift ratio (>1.0 = higher/more feminine, <1.0 = lower/more masculine)
    
    Returns:
        Formant-shifted audio data
    """
    if abs(shift_ratio - 1.0) < 0.01:
        return data
    
    # Try parselmouth (Praat) - best quality
    try:
        import parselmouth
        from parselmouth.praat import call
        
        def process_channel(channel_data: np.ndarray) -> np.ndarray:
            snd = parselmouth.Sound(channel_data, sampling_frequency=sr)
            
            # LPC-based formant manipulation
            # Shift formants via Manipulation
            manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
            
            # Apply formant shift via duration/pitch manipulation
            # In Praat, scaling duration while keeping pitch constant effectively shifts formants
            # when combined with proper resampling
            duration_tier = call(manipulation, "Extract duration tier")
            call(duration_tier, "Add point", 0, 1.0/shift_ratio)
            call(manipulation, "Replace duration tier", duration_tier)
            
            resynthesis = call(manipulation, "Get resynthesis (overlap-add)")
            # Resample back to original length/pitch to isolate formant shift
            shifted_data = np.array(call(resynthesis, "Get all samples")[0])
            
            # Ensure same length as input
            if len(shifted_data) != len(channel_data):
                from scipy.signal import resample
                shifted_data = resample(shifted_data, len(channel_data))
                
            return shifted_data
        
        if data.ndim > 1:
            result = np.zeros_like(data)
            for ch in range(data.shape[1]):
                result[:, ch] = process_channel(data[:, ch])
            return result.astype(np.float32)
        else:
            return process_channel(data).astype(np.float32)
            
    except (ImportError, Exception) as e:
        logger.debug("Parselmouth not available, using EQ approximation: %s", e)
    
    # Fallback: EQ-based approximation
    result = data.copy()
    if shift_ratio > 1.0:
        result = apply_peaking_eq(result, sr, frequency=2500, gain_db=4.0, Q=0.8)
        result = apply_peaking_eq(result, sr, frequency=500, gain_db=-3.0, Q=1.0)
        result = apply_high_shelf(result, sr, cutoff=8000, gain_db=3.0)
    else:
        result = apply_peaking_eq(result, sr, frequency=500, gain_db=3.0, Q=1.0)
        result = apply_peaking_eq(result, sr, frequency=2500, gain_db=-3.0, Q=1.2)
    
    return result.astype(np.float32)


def apply_pitch_correction(
    data: AudioArray,
    sr: int,
    strength: float = 0.35,
    speed: float = 0.65,
    quantize: float = 0.25,
) -> AudioArray:
    """
    Apply gentle pitch correction without hard-robotic artifacts.

    Args:
        data: Audio samples
        sr: Sample rate
        strength: Amount of correction (0.0 to 1.0)
        speed: Smoothing (0.0 = slow/soft, 1.0 = fast/snappier)
        quantize: Pull towards nearest semitone (0.0 to 1.0)

    Returns:
        Pitch-corrected audio data
    """
    if strength <= 0.0:
        return data

    try:
        import librosa
    except ImportError:
        logger.warning("Librosa not available for pitch correction, skipping...")
        return data

    y_mono = np.mean(data, axis=1) if data.ndim > 1 else data

    hop_length = 512
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C6")

    try:
        f0 = librosa.yin(y_mono, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    except Exception as e:
        logger.error("Pitch detection failed: %s", e)
        return data

    valid = (f0 > 0) & (~np.isnan(f0))
    if not np.any(valid):
        return data

    midi = np.zeros_like(f0)
    midi[valid] = librosa.hz_to_midi(f0[valid])

    target = midi.copy()
    if quantize > 0.0:
        target[valid] = midi[valid] + (np.round(midi[valid]) - midi[valid]) * quantize

    try:
        from scipy.ndimage import gaussian_filter1d

        sigma = max(1, int((1.0 - speed) * 8))
        target = gaussian_filter1d(target, sigma=sigma)
    except Exception:
        pass

    shift_signal = (target - midi) * strength

    block_size = int(sr * 0.1)
    if block_size <= 0:
        return data
    if block_size > len(data):
        block_size = len(data)
    hop_size = max(1, block_size // 2)
    window = np.hanning(block_size)

    output = np.zeros_like(data)
    norm_weights = np.zeros(len(data))

    total_blocks = max(1, (len(data) - block_size) // hop_size)
    for b in range(total_blocks):
        i = b * hop_size
        if i + block_size > len(data):
            break

        msg_idx = int((i + block_size // 2) / hop_length)
        if msg_idx >= len(shift_signal):
            break

        n_steps = shift_signal[msg_idx]
        if abs(n_steps) < 0.05:
            shifted_block = data[i : i + block_size]
        else:
            block = data[i : i + block_size]
            if block.ndim > 1:
                shifted_block = np.zeros_like(block)
                for ch in range(block.shape[1]):
                    shifted_block[:, ch] = librosa.effects.pitch_shift(
                        block[:, ch], sr=sr, n_steps=n_steps
                    )
            else:
                shifted_block = librosa.effects.pitch_shift(block, sr=sr, n_steps=n_steps)

        if data.ndim > 1:
            win_2d = window[:, np.newaxis]
            output[i : i + block_size] += shifted_block * win_2d
        else:
            output[i : i + block_size] += shifted_block * window

        norm_weights[i : i + block_size] += window

    norm_weights[norm_weights < 1e-6] = 1.0
    if data.ndim > 1:
        output /= norm_weights[:, np.newaxis]
    else:
        output /= norm_weights

    return output.astype(np.float32)



def apply_compressor(
    data: AudioArray, 
    sr: int,
    threshold_db: float = EFFECTS_CONFIG.compressor_threshold_db,
    ratio: float = EFFECTS_CONFIG.compressor_ratio,
    attack_ms: float = EFFECTS_CONFIG.compressor_attack_ms,
    release_ms: float = EFFECTS_CONFIG.compressor_release_ms,
    makeup_db: float = 0.0
) -> AudioArray:
    """
    Apply dynamic range compression.
    
    Args:
        data: Audio samples
        sr: Sample rate
        threshold_db: Threshold level in dB
        ratio: Compression ratio (e.g., 4.0 = 4:1)
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        makeup_db: Makeup gain in dB
    
    Returns:
        Compressed audio data
    """
    if ratio <= 1.0:
        return data
    
    threshold = 10 ** (threshold_db / 20)
    
    # Time constants
    attack_samples = max(1, int(sr * attack_ms / 1000))
    release_samples = max(1, int(sr * release_ms / 1000))
    
    attack_coeff = 1 - math.exp(-1.0 / attack_samples)
    release_coeff = 1 - math.exp(-1.0 / release_samples)
    
    # Get mono envelope for detection
    if data.ndim > 1:
        env_input = np.max(np.abs(data), axis=1)
    else:
        env_input = np.abs(data)
    
    # Envelope follower
    envelope = np.zeros_like(env_input)
    env_prev = 0.0
    
    for i in range(len(env_input)):
        if env_input[i] > env_prev:
            env_prev += attack_coeff * (env_input[i] - env_prev)
        else:
            env_prev += release_coeff * (env_input[i] - env_prev)
        envelope[i] = env_prev
    
    # Calculate gain reduction
    gain_reduction = np.ones_like(envelope)
    above_threshold = envelope > threshold
    
    if np.any(above_threshold):
        # Apply compression curve
        gain_reduction[above_threshold] = (
            threshold + (envelope[above_threshold] - threshold) / ratio
        ) / envelope[above_threshold]
    
    # Apply makeup gain
    makeup_linear = 10 ** (makeup_db / 20)
    gain_reduction *= makeup_linear
    
    # Apply to audio
    if data.ndim > 1:
        gain_reduction = gain_reduction[:, np.newaxis]
    
    return (data * gain_reduction).astype(np.float32)


def apply_deesser(
    data: AudioArray, 
    sr: int,
    frequency: float = 7000.0,
    threshold_db: float = -20.0,
    reduction_db: float = 6.0,
    Q: float = 2.0
) -> AudioArray:
    """
    Apply de-esser to reduce sibilance.
    
    Args:
        data: Audio samples
        sr: Sample rate
        frequency: Center frequency for sibilance detection (Hz)
        threshold_db: Threshold for triggering reduction
        reduction_db: Amount of gain reduction
        Q: Filter Q for detection band
    
    Returns:
        De-essed audio data
    """
    from scipy.signal import butter, lfilter
    
    # Detection filter (bandpass around sibilance frequency)
    nyquist = sr / 2
    low = np.clip((frequency - 500) / nyquist, 0.001, 0.999)
    high = np.clip((frequency + 500) / nyquist, low + 0.001, 0.999)
    
    b, a = butter(2, [low, high], btype='band')
    
    # Get sibilance envelope
    if data.ndim > 1:
        sibilance = lfilter(b, a, np.mean(data, axis=1))
    else:
        sibilance = lfilter(b, a, data)
    
    envelope = np.abs(sibilance)
    
    # Smooth envelope
    smooth_samples = int(sr * 0.005)  # 5ms
    if smooth_samples > 1:
        envelope = np.convolve(
            envelope, 
            np.ones(smooth_samples) / smooth_samples, 
            mode='same'
        )
    
    # Calculate gain reduction
    threshold = 10 ** (threshold_db / 20)
    reduction = 10 ** (-reduction_db / 20)
    
    gain = np.ones_like(envelope)
    above_threshold = envelope > threshold
    
    if np.any(above_threshold):
        excess = envelope[above_threshold] / threshold
        gain[above_threshold] = 1.0 - (1.0 - reduction) * np.tanh(excess - 1)
    
    # Apply to original
    if data.ndim > 1:
        gain = gain[:, np.newaxis]
    
    return (data * gain).astype(np.float32)


def apply_chorus(
    data: AudioArray, 
    sr: int,
    rate: float = 0.5,
    depth_ms: float = 3.0,
    mix: float = 0.3,
    voices: int = 2
) -> AudioArray:
    """
    Apply chorus effect for width and thickness.
    
    Args:
        data: Audio samples
        sr: Sample rate
        rate: LFO rate in Hz
        depth_ms: Modulation depth in milliseconds
        mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        voices: Number of chorus voices
    
    Returns:
        Audio with chorus effect
    """
    depth_samples = int(sr * depth_ms / 1000.0)
    base_delay = int(sr * 0.020)  # 20ms base delay
    
    def process_channel(channel_data: np.ndarray) -> np.ndarray:
        length = len(channel_data)
        output = channel_data.copy()
        
        for voice in range(voices):
            phase_offset = voice * 2 * np.pi / voices
            
            # Generate LFO
            t = np.arange(length) / sr
            lfo = np.sin(2 * np.pi * rate * t + phase_offset)
            
            # Calculate delay in samples
            delay = base_delay + (depth_samples * lfo).astype(int)
            delay = np.clip(delay, 1, length - 1)
            
            # Apply variable delay
            indices = np.arange(length) - delay
            indices = np.clip(indices, 0, length - 1)
            
            delayed = channel_data[indices]
            output = output + delayed * (mix / voices)
        
        # Normalize
        output = output / (1 + mix)
        
        return output.astype(np.float32)
    
    if data.ndim > 1:
        result = np.zeros_like(data)
        for ch in range(data.shape[1]):
            result[:, ch] = process_channel(data[:, ch])
        return result
    else:
        return process_channel(data)


def apply_vocoder(
    data: AudioArray, 
    sr: int,
    carrier_type: str = 'saw',
    bands: int = 16,
    mix: float = 0.3
) -> AudioArray:
    """
    Apply vocoder effect for robotic/synthetic vocal sound.
    
    Args:
        data: Audio samples (modulator - the voice)
        sr: Sample rate
        carrier_type: 'saw', 'square', 'noise', or 'pulse'
        bands: Number of frequency bands
        mix: Wet/dry mix
    
    Returns:
        Vocoded audio data
    """
    length = len(data) if data.ndim == 1 else data.shape[0]
    
    # Generate carrier signal
    t = np.arange(length) / sr
    base_freq = 150  # Base frequency for carrier
    
    if carrier_type == 'saw':
        carrier = 2 * (t * base_freq % 1) - 1
    elif carrier_type == 'square':
        carrier = np.sign(np.sin(2 * np.pi * base_freq * t))
    elif carrier_type == 'pulse':
        carrier = (np.sin(2 * np.pi * base_freq * t) > 0.8).astype(float) * 2 - 1
    else:  # noise
        carrier = np.random.uniform(-1, 1, length).astype(np.float32)
    
    # Add harmonics for richer sound
    for harmonic in [2, 3, 4, 5]:
        if carrier_type in ['saw', 'square']:
            carrier += np.sin(2 * np.pi * base_freq * harmonic * t) / harmonic
    
    carrier = carrier / np.max(np.abs(carrier))  # Normalize
    
    def process_channel(channel_data: np.ndarray) -> np.ndarray:
        output = np.zeros_like(channel_data)
        
        # Create filter bank
        nyquist = sr / 2
        min_freq = 80
        max_freq = min(12000, nyquist * 0.9)
        
        # Logarithmic frequency spacing
        freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), bands + 1)
        
        for i in range(bands):
            low = freqs[i] / nyquist
            high = freqs[i + 1] / nyquist
            
            low = np.clip(low, 0.001, 0.999)
            high = np.clip(high, low + 0.001, 0.999)
            
            if high <= low:
                continue
            
            try:
                b, a = butter(2, [low, high], btype='band')
                
                # Filter modulator to get envelope
                mod_band = lfilter(b, a, channel_data)
                envelope = np.abs(mod_band)
                
                # Smooth envelope
                smooth_samples = int(sr * 0.010)  # 10ms
                if smooth_samples > 1:
                    envelope = np.convolve(
                        envelope, 
                        np.ones(smooth_samples) / smooth_samples, 
                        mode='same'
                    )
                
                # Filter carrier
                carrier_band = lfilter(b, a, carrier[:len(channel_data)])
                
                # Modulate carrier with envelope
                output += carrier_band * envelope
            except Exception:
                continue
        
        # Normalize output
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val
        
        # Mix with original
        result = (1 - mix) * channel_data + mix * output
        
        return result.astype(np.float32)
    
    if data.ndim > 1:
        result = np.zeros_like(data)
        for ch in range(data.shape[1]):
            result[:, ch] = process_channel(data[:, ch])
        return result
    else:
        return process_channel(data)


# =============================================================================
# VOCAL PRESETS
# =============================================================================

def apply_slowed_reverb_preset(data: AudioArray, sr: int) -> AudioArray:
    """
    Apply 'Slowed + Reverb' preset (popular on TikTok/YouTube).
    
    Args:
        data: Audio samples
        sr: Sample rate
    
    Returns:
        Processed audio with slowed + reverb effect
    """
    from .effects_basic import apply_resample, apply_reverb, apply_low_shelf
    
    # Slow down by ~15%
    processed = apply_resample(data, factor=0.85)
    
    # Add heavy reverb
    processed = apply_reverb(processed, sr, room_size=0.8, damping=0.4)
    
    # Boost bass
    processed = apply_low_shelf(processed, sr, cutoff=200, gain_db=4.0)
    
    # Soft saturation
    processed = apply_soft_clip(processed, threshold=0.9)
    
    return processed


def apply_nightcore_preset(data: AudioArray, sr: int) -> AudioArray:
    """
    Apply 'Nightcore' preset (speed up + pitch up).
    
    Args:
        data: Audio samples
        sr: Sample rate
    
    Returns:
        Processed audio with nightcore effect
    """
    from .effects_basic import apply_resample, apply_high_shelf
    
    # Speed up by ~25%
    processed = apply_resample(data, factor=1.25)
    
    # Boost highs for brightness
    processed = apply_high_shelf(processed, sr, cutoff=8000, gain_db=3.0)
    
    return processed


def apply_lofi_preset(data: AudioArray, sr: int) -> AudioArray:
    """
    Apply 'Lo-Fi' preset (vinyl warmth, slight saturation).
    
    Args:
        data: Audio samples
        sr: Sample rate
    
    Returns:
        Processed audio with lo-fi characteristics
    """
    from .effects_basic import (
        apply_lowpass, apply_low_shelf, apply_bitcrush
    )
    
    # Low-pass for vintage sound
    processed = apply_lowpass(data, sr, cutoff=12000)
    
    # Boost bass warmth
    processed = apply_low_shelf(processed, sr, cutoff=200, gain_db=3.0)
    
    # Slight bitcrush for texture
    processed = apply_bitcrush(processed, bits=14, downsample=1, mix=0.1)
    
    # Soft saturation
    processed = apply_soft_clip(processed, threshold=0.85)
    
    return processed


def apply_bass_boosted_preset(data: AudioArray, sr: int) -> AudioArray:
    """
    Apply 'Bass Boosted' preset.
    
    Args:
        data: Audio samples
        sr: Sample rate
    
    Returns:
        Processed audio with boosted bass
    """
    from .effects_basic import apply_low_shelf
    
    # Heavy bass boost
    processed = apply_low_shelf(data, sr, cutoff=100, gain_db=8.0)
    
    # Slight mid-bass boost
    processed = apply_peaking_eq(processed, sr, frequency=60, gain_db=6.0, Q=0.8)
    
    # Limiter to prevent clipping
    processed = apply_soft_clip(processed, threshold=0.9)
    
    return processed


def apply_podcast_clean_preset(data: AudioArray, sr: int) -> AudioArray:
    """
    Apply 'Podcast Clean' preset (clarity, level control, reduced sibilance).

    Args:
        data: Audio samples
        sr: Sample rate

    Returns:
        Processed audio with broadcast-style clarity
    """
    processed = apply_highpass(data, sr, cutoff=80)
    processed = apply_peaking_eq(processed, sr, frequency=3200, gain_db=3.0, Q=1.0)
    processed = apply_high_shelf(processed, sr, cutoff=9000, gain_db=2.0, Q=0.7)
    processed = apply_compressor(
        processed,
        sr,
        threshold_db=-22.0,
        ratio=3.0,
        attack_ms=5.0,
        release_ms=120.0,
        makeup_db=3.0,
    )
    processed = apply_deesser(processed, sr, frequency=6500, threshold_db=-24.0, reduction_db=5.0)
    processed = apply_soft_clip(processed, threshold=0.95)
    return processed


def apply_radio_voice_preset(data: AudioArray, sr: int) -> AudioArray:
    """
    Apply 'Radio Voice' preset (band-limited, gritty broadcast tone).

    Args:
        data: Audio samples
        sr: Sample rate

    Returns:
        Processed audio with classic radio/telephone character
    """
    from .effects_basic import apply_bandpass, apply_bitcrush

    processed = apply_bandpass(data, sr, low_cutoff=300.0, high_cutoff=3400.0, order=2)
    processed = apply_peaking_eq(processed, sr, frequency=1200, gain_db=2.5, Q=1.0)
    processed = apply_compressor(
        processed,
        sr,
        threshold_db=-26.0,
        ratio=4.0,
        attack_ms=4.0,
        release_ms=120.0,
        makeup_db=4.0,
    )
    processed = apply_bitcrush(processed, bits=12, downsample=2, mix=0.12)
    processed = apply_soft_clip(processed, threshold=0.9)
    return processed


def apply_dreamy_space_preset(data: AudioArray, sr: int) -> AudioArray:
    """
    Apply 'Dreamy Space' preset (lush, spacious, slightly washed).

    Args:
        data: Audio samples
        sr: Sample rate

    Returns:
        Processed audio with airy ambience
    """
    from .effects_basic import apply_delay, apply_lowpass, apply_low_shelf, apply_reverb

    processed = apply_lowpass(data, sr, cutoff=12000)
    processed = apply_chorus(processed, sr, rate=0.35, depth_ms=4.0, mix=0.25, voices=3)
    processed = apply_delay(processed, sr, delay_ms=180.0, decay=0.25, feedback=True)
    processed = apply_reverb(processed, sr, room_size=0.75, damping=0.45)
    processed = apply_low_shelf(processed, sr, cutoff=200, gain_db=2.0)
    processed = apply_high_shelf(processed, sr, cutoff=9000, gain_db=1.5)
    processed = apply_soft_clip(processed, threshold=0.92)
    return processed
