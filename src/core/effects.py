import numpy as np
from scipy.signal import butter, lfilter

def apply_gain(data, factor=2.0):
    """Multiplies audio data by a gain factor."""
    return data * factor

def apply_fade_in(data):
    """Applies a linear fade-in to the audio data."""
    length = len(data)
    if length == 0: return data
    fade_curve = np.linspace(0, 1, length)
    if data.ndim > 1:
        fade_curve = fade_curve[:, np.newaxis]
    return data * fade_curve

def apply_fade_out(data):
    """Applies a linear fade-out to the audio data."""
    length = len(data)
    if length == 0: return data
    fade_curve = np.linspace(1, 0, length)
    if data.ndim > 1:
        fade_curve = fade_curve[:, np.newaxis]
    return data * fade_curve

def apply_delay(data, sr, delay_ms=300, decay=0.4):
    """Applies a simple feedback delay (echo) effect."""
    delay_samples = int(sr * (delay_ms / 1000.0))
    output = data.copy()
    if delay_samples >= len(data) or delay_samples <= 0:
        return output
    
    # Feedback loop
    for i in range(delay_samples, len(data)):
        output[i] += output[i - delay_samples] * decay
    
    return output

def apply_reverb(data, sr, room_size=0.5):
    """Applies a very basic reverb effect using multiple parallel delays."""
    delays = [
        (int(25 * room_size), 0.3),
        (int(37 * room_size), 0.25),
        (int(53 * room_size), 0.2),
        (int(71 * room_size), 0.15)
    ]
    
    combined = data.copy()
    for ms, dcy in delays:
        delay_samples = int(sr * (ms / 1000.0))
        if delay_samples < len(data):
            # Simple non-feedback delay for reverb density
            combined[delay_samples:] += data[:-delay_samples] * dcy
    
    return combined / (1.0 + sum(d[1] for d in delays))

def apply_lowpass(data, sr, cutoff=1000):
    """Applies a first-order Butterworth low-pass filter."""
    nyquist = 0.5 * sr
    normal_cutoff = np.clip(cutoff / nyquist, 0.001, 0.999)
    
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data, axis=0).astype(np.float32)

def apply_low_shelf(data, sr, cutoff=200, gain_db=6.0, Q=0.707):
    """
    Applies a low-shelf filter to boost or cut low frequencies.
    Uses standard biquad cookbook formulae.
    """
    import math
    A = 10**(gain_db/40)
    omega = 2 * math.pi * cutoff / sr
    sn, cs = math.sin(omega), math.cos(omega)
    alpha = sn / (2 * Q)
    
    # Biquad coefficients
    b0 = A * ((A + 1) - (A - 1) * cs + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cs)
    b2 = A * ((A + 1) - (A - 1) * cs - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cs + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cs)
    a2 = (A + 1) + (A - 1) * cs - 2 * math.sqrt(A) * alpha
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    return lfilter(b, a, data, axis=0).astype(np.float32)

def apply_soft_clip(data, threshold=0.85):
    """
    Applies soft saturation to prevent harsh digital clipping.
    Uses a cubic approximation for a warm sound.
    """
    out = np.clip(data, -1.5, 1.5)
    out = out - (out**3) / 6.0 
    
    # Normalize to ensure peak is at 1.0
    max_val = np.max(np.abs(out))
    if max_val > 1.0: out /= max_val
        
    return out.astype(np.float32)

def apply_resample(data, factor=1.0):
    """Changes speed and pitch by resampling."""
    if factor == 1.0: return data
    
    from scipy.interpolate import interp1d
    length = len(data)
    new_length = int(length / factor)
    
    x = np.linspace(0, length - 1, length)
    x_new = np.linspace(0, length - 1, new_length)
    
    if data.ndim > 1:
        # Stereo
        new_data = np.zeros((new_length, data.shape[1]), dtype=np.float32)
        for ch in range(data.shape[1]):
            f = interp1d(x, data[:, ch], kind='linear', fill_value="extrapolate")
            new_data[:, ch] = f(x_new)
        return new_data
    else:
        # Mono
        f = interp1d(x, data, kind='linear', fill_value="extrapolate")
        return f(x_new).astype(np.float32)
