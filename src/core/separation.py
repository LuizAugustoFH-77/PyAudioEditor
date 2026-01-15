"""
AI-powered vocal separation backends for PyAudioEditor.
Supports Demucs, Spleeter, and DSP-based fallback (CPU only).
"""
from __future__ import annotations
import logging
import os
import tempfile
import threading
import numpy as np

from .types import AudioArray, SeparationResult
from .track import AudioTrack

logger = logging.getLogger("PyAudacity")


# =============================================================================
# MODEL CACHES (CPU ONLY)
# =============================================================================

_demucs_separator = None
_demucs_separator_lock = threading.Lock()
_demucs_model = None
_demucs_model_lock = threading.Lock()
_hdemucs_model = None
_hdemucs_lock = threading.Lock()
_spleeter_separators: dict[int, object] = {}
_spleeter_lock = threading.Lock()


# =============================================================================
# CAPABILITY CHECKS
# =============================================================================

def is_demucs_available() -> bool:
    """Check if Demucs is available."""
    try:
        import torch
        # Try the new API first (demucs >= 4.1)
        try:
            import demucs.api
            return True
        except ImportError:
            pass
        # Try the old API (demucs 4.0.x)
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        return True
    except ImportError:
        return False


def is_spleeter_available() -> bool:
    """Check if Spleeter is available."""
    try:
        import spleeter
        return True
    except ImportError:
        return False


def get_best_backend() -> str:
    """Get the best available separation backend."""
    if is_demucs_available():
        return "demucs"
    if is_spleeter_available():
        return "spleeter"
    return "dsp"


def _get_demucs_separator(model_name: str):
    """Cache Demucs separator to avoid repeated model loads."""
    global _demucs_separator
    if _demucs_separator is not None:
        return _demucs_separator
    with _demucs_separator_lock:
        if _demucs_separator is None:
            import torch
            import demucs.api
            _demucs_separator = demucs.api.Separator(
                model=model_name,
                device=torch.device("cpu"),
                progress=False,
            )
    return _demucs_separator


def _get_demucs_model(model_name: str):
    """Cache Demucs model for the legacy API."""
    global _demucs_model
    if _demucs_model is not None:
        return _demucs_model
    with _demucs_model_lock:
        if _demucs_model is None:
            import torch
            from demucs.pretrained import get_model
            model = get_model(model_name)
            model.to(torch.device("cpu"))
            model.eval()
            _demucs_model = model
    return _demucs_model


def _get_hdemucs_model():
    """Cache torchaudio HDemucs model."""
    global _hdemucs_model
    if _hdemucs_model is not None:
        return _hdemucs_model
    with _hdemucs_lock:
        if _hdemucs_model is None:
            import torch
            from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
            bundle = HDEMUCS_HIGH_MUSDB_PLUS
            model = bundle.get_model()
            model.to(torch.device("cpu"))
            model.eval()
            _hdemucs_model = model
    return _hdemucs_model


def _get_spleeter_separator(stems: int):
    """Cache Spleeter separator per stem count."""
    cached = _spleeter_separators.get(stems)
    if cached is not None:
        return cached
    with _spleeter_lock:
        cached = _spleeter_separators.get(stems)
        if cached is None:
            from spleeter.separator import Separator
            cached = Separator(f"spleeter:{stems}stems")
            _spleeter_separators[stems] = cached
    return cached


# =============================================================================
# SEPARATION IMPLEMENTATIONS
# =============================================================================

def separate_with_demucs(
    data: AudioArray,
    samplerate: int,
    two_stems: bool = True
) -> SeparationResult:
    """
    Separate audio using Demucs (Meta AI).
    Best quality but requires PyTorch.
    
    Args:
        data: Audio samples
        samplerate: Sample rate
        two_stems: If True, return vocals + instrumental. If False, return all stems.
    
    Returns:
        SeparationResult with success status and extracted tracks
    """
    try:
        if data is None or len(data) == 0:
            return SeparationResult(success=False, error="No audio data provided.")

        import torch

        model_name = "htdemucs"
        logger.info(
            "Starting Demucs separation (two_stems=%s, model=%s) on CPU...",
            two_stems,
            model_name,
        )
        
        # Ensure stereo
        if data.ndim == 1:
            audio_stereo = np.column_stack((data, data))
        else:
            audio_stereo = data
        audio_stereo = np.ascontiguousarray(audio_stereo, dtype=np.float32)
        
        # Try the new API first (demucs >= 4.1)
        try:
            import demucs.api
            
            # Demucs expects (channels, samples) - shape: (2, num_samples)
            audio_tensor = torch.from_numpy(np.ascontiguousarray(audio_stereo.T))

            # Cached separator avoids repeated model loading.
            separator = _get_demucs_separator(model_name)

            logger.info("Demucs model ready (new API), sample rate: %d", separator.samplerate)

            with torch.inference_mode():
                _, separated = separator.separate_tensor(audio_tensor, sr=samplerate)
            
            logger.info("Separation complete, stems: %s", list(separated.keys()))
            
        except ImportError:
            # Use the old API (demucs 4.0.x)
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            import torchaudio
            
            logger.info("Using Demucs old API (4.0.x) with model '%s'", model_name)
            
            # Load cached model
            model = _get_demucs_model(model_name)

            # Prepare audio tensor (batch, channels, samples)
            audio_tensor = torch.from_numpy(np.ascontiguousarray(audio_stereo.T)).unsqueeze(0)
            
            # Resample if needed
            model_sr = model.samplerate
            if samplerate != model_sr:
                resampler = torchaudio.transforms.Resample(samplerate, model_sr)
                audio_tensor = resampler(audio_tensor)
            
            # Run separation
            with torch.no_grad():
                sources = apply_model(model, audio_tensor, device=torch.device("cpu"), progress=True)
            
            # sources shape: (batch, stems, channels, samples)
            sources = sources.squeeze(0)  # (stems, channels, samples)
            
            # Resample back if needed
            if samplerate != model_sr:
                resampler_back = torchaudio.transforms.Resample(model_sr, samplerate)
                sources = torch.stack([resampler_back(s) for s in sources])
            
            # Build separated dict
            stem_names = model.sources  # ['drums', 'bass', 'other', 'vocals']
            separated = {}
            for i, name in enumerate(stem_names):
                separated[name] = sources[i]
            
            logger.info("Separation complete (old API), stems: %s", list(separated.keys()))
        
        # Build result tracks
        tracks: list[tuple[str, AudioArray]] = []
        
        if two_stems:
            # Get vocals
            if "vocals" in separated:
                vocals_tensor = separated["vocals"]
                vocals = vocals_tensor.cpu().numpy().T  # (samples, channels)
                tracks.append(("Vocals", vocals.astype(np.float32)))
            
            # Combine everything else as instrumental
            instrumental = None
            for stem_name, stem_tensor in separated.items():
                if stem_name != "vocals":
                    stem_data = stem_tensor.cpu().numpy()  # (channels, samples)
                    if instrumental is None:
                        instrumental = stem_data
                    else:
                        instrumental = instrumental + stem_data
            
            if instrumental is not None:
                tracks.append(("Instrumental", instrumental.T.astype(np.float32)))
        else:
            # Return all stems
            # htdemucs order: drums, bass, other, vocals
            stem_order = ['drums', 'bass', 'other', 'vocals']
            stem_display = ['Drums', 'Bass', 'Other', 'Vocals']
            
            for stem_key, stem_name in zip(stem_order, stem_display):
                if stem_key in separated:
                    stem_tensor = separated[stem_key]
                    stem_data = stem_tensor.cpu().numpy().T  # (samples, channels)
                    tracks.append((stem_name, stem_data.astype(np.float32)))
        
        logger.info("Demucs separation completed successfully with %d stems", len(tracks))
        return SeparationResult(success=True, tracks=tracks)
        
    except ImportError as e:
        logger.warning("Demucs not available: %s", e)
        return SeparationResult(
            success=False, 
            error="Demucs not installed. Install with: pip install demucs torch torchaudio"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error("Demucs separation failed: %s", error_msg, exc_info=True)
        return SeparationResult(success=False, error=error_msg)



def separate_with_torchaudio_hdemucs(
    data: AudioArray,
    samplerate: int,
    two_stems: bool = True
) -> SeparationResult:
    """
    Alternative Demucs separation using torchaudio's built-in HDemucs model.
    More stable API, included with torchaudio.
    
    Args:
        data: Audio samples
        samplerate: Sample rate
        two_stems: If True, return vocals + instrumental. If False, return all stems.
    
    Returns:
        SeparationResult with success status and extracted tracks
    """
    try:
        if data is None or len(data) == 0:
            return SeparationResult(success=False, error="No audio data provided.")

        import torch
        import torchaudio
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
        
        logger.info("Starting torchaudio HDemucs separation on CPU...")
        
        # Ensure stereo
        if data.ndim == 1:
            audio_stereo = np.column_stack((data, data))
        else:
            audio_stereo = data
        
        # (channels, samples)
        waveform = torch.from_numpy(audio_stereo.T.astype(np.float32))
        
        # Get cached model
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model = _get_hdemucs_model()
        
        # Resample if needed
        model_sample_rate = bundle.sample_rate
        if samplerate != model_sample_rate:
            resampler = torchaudio.transforms.Resample(samplerate, model_sample_rate)
            waveform = resampler(waveform)
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0)  # (1, channels, samples)
        
        # Run model
        with torch.inference_mode():
            sources = model(waveform)  # (1, 4, channels, samples)
        
        sources = sources.squeeze(0).cpu().numpy()  # (4, channels, samples)
        
        # Resample back if needed
        if samplerate != model_sample_rate:
            from scipy.signal import resample
            original_len = len(data)
            new_sources = []
            for src in sources:
                resampled = np.zeros((2, original_len), dtype=np.float32)
                for ch in range(2):
                    resampled[ch] = resample(src[ch], original_len)
                new_sources.append(resampled)
            sources = np.array(new_sources)
        
        # Order: drums, bass, other, vocals
        stem_names = ['Drums', 'Bass', 'Other', 'Vocals']
        tracks: list[tuple[str, AudioArray]] = []
        
        if two_stems:
            vocals = sources[3].T  # (samples, channels)
            instrumental = (sources[0] + sources[1] + sources[2]).T
            tracks = [
                ("Vocals", vocals.astype(np.float32)),
                ("Instrumental", instrumental.astype(np.float32))
            ]
        else:
            for i, name in enumerate(stem_names):
                stem_data = sources[i].T
                tracks.append((name, stem_data.astype(np.float32)))
        
        logger.info("torchaudio HDemucs separation completed successfully")
        return SeparationResult(success=True, tracks=tracks)
        
    except ImportError as e:
        logger.warning("torchaudio HDemucs not available: %s", e)
        return SeparationResult(
            success=False,
            error="torchaudio HDemucs not available. Install with: pip install torch torchaudio"
        )
    except Exception as e:
        logger.error("torchaudio HDemucs separation failed: %s", e, exc_info=True)
        return SeparationResult(success=False, error=str(e))


def separate_with_spleeter(
    data: AudioArray,
    samplerate: int,
    stems: int = 2
) -> SeparationResult:
    """
    Separate audio using Spleeter (Deezer).
    Good quality, CPU-friendly.
    
    Args:
        data: Audio samples
        samplerate: Sample rate
        stems: 2 (vocals/accompaniment), 4 (vocals/drums/bass/other), or 5 (+piano)
    
    Returns:
        SeparationResult with success status and extracted tracks
    """
    try:
        if data is None or len(data) == 0:
            return SeparationResult(success=False, error="No audio data provided.")

        logger.info("Starting Spleeter %d-stem separation...", stems)

        # Ensure stereo float32
        audio_data = data
        if audio_data.ndim == 1:
            audio_data = np.column_stack((audio_data, audio_data))
        audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)

        separator = _get_spleeter_separator(stems)

        stem_map = {}
        if stems == 2:
            stem_map = {"vocals": "Vocals", "accompaniment": "Instrumental"}
        elif stems == 4:
            stem_map = {"vocals": "Vocals", "drums": "Drums", "bass": "Bass", "other": "Other"}
        else:
            stem_map = {
                "vocals": "Vocals",
                "drums": "Drums",
                "bass": "Bass",
                "piano": "Piano",
                "other": "Other",
            }

        tracks: list[tuple[str, AudioArray]] = []

        try:
            separated = separator.separate(audio_data, samplerate)
            for stem_key, stem_name in stem_map.items():
                if stem_key in separated:
                    tracks.append((stem_name, separated[stem_key].astype(np.float32)))
            logger.info("Spleeter in-memory separation complete")
        except Exception as e:
            logger.warning("Spleeter in-memory separation failed, falling back: %s", e)
            import soundfile as sf
            from scipy.signal import resample

            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = os.path.join(tmpdir, "input.wav")
                output_dir = os.path.join(tmpdir, "output")

                sf.write(input_path, audio_data, samplerate)
                separator.separate_to_file(input_path, output_dir)

                stem_folder = os.path.join(output_dir, "input")
                stem_files = [f"{key}.wav" for key in stem_map.keys()]

                for stem_file, stem_key in zip(stem_files, stem_map.keys()):
                    stem_path = os.path.join(stem_folder, stem_file)
                    if os.path.exists(stem_path):
                        stem_data, stem_sr = sf.read(stem_path)

                        if stem_sr != samplerate:
                            target_len = int(len(stem_data) * samplerate / stem_sr)
                            if stem_data.ndim > 1:
                                resampled = np.zeros((target_len, stem_data.shape[1]))
                                for ch in range(stem_data.shape[1]):
                                    resampled[:, ch] = resample(stem_data[:, ch], target_len)
                                stem_data = resampled
                            else:
                                stem_data = resample(stem_data, target_len)

                        tracks.append((stem_map[stem_key], stem_data.astype(np.float32)))

        if not tracks:
            return SeparationResult(success=False, error="Spleeter produced no stems.")

        logger.info("Spleeter separation completed successfully")
        return SeparationResult(success=True, tracks=tracks)
        
    except ImportError as e:
        logger.warning("Spleeter not available: %s", e)
        return SeparationResult(
            success=False,
            error="Spleeter not installed. Install with: pip install spleeter"
        )
    except Exception as e:
        logger.error("Spleeter separation failed: %s", e, exc_info=True)
        return SeparationResult(success=False, error=str(e))


def separate_with_dsp(data: AudioArray, samplerate: int) -> SeparationResult:
    """
    Separate vocals using DSP-based center channel cancellation.
    Fallback method when AI backends are unavailable.
    Works best on professionally mixed music with centered vocals.
    
    Args:
        data: Audio samples
        samplerate: Sample rate
    
    Returns:
        SeparationResult with vocals and instrumental tracks
    """
    logger.info("Starting DSP-based vocal separation...")
    
    try:
        if data is None or len(data) == 0:
            return SeparationResult(success=False, error="No audio data provided.")

        # Ensure stereo
        if data.ndim == 1:
            logger.warning("DSP separation requires stereo audio, returning unchanged")
            return SeparationResult(
                success=False,
                error="DSP separation requires stereo audio"
            )
        
        if data.shape[1] < 2:
            return SeparationResult(
                success=False,
                error="DSP separation requires stereo audio"
            )
        
        left = data[:, 0]
        right = data[:, 1]
        
        # Center = (L + R) / 2 (typically contains vocals)
        # Sides = (L - R) / 2 (typically contains instruments)
        center = (left + right) / 2
        sides = (left - right) / 2
        
        # Vocals approximation: center channel
        vocals = np.column_stack((center, center)).astype(np.float32)
        
        # Instrumental: sides + attenuated center (remove vocals)
        instrumental_left = sides + center * 0.3
        instrumental_right = -sides + center * 0.3
        instrumental = np.column_stack((instrumental_left, instrumental_right)).astype(np.float32)
        
        logger.info("DSP separation completed")
        return SeparationResult(
            success=True,
            tracks=[
                ("Vocals (DSP)", vocals),
                ("Instrumental (DSP)", instrumental)
            ]
        )
        
    except Exception as e:
        logger.error("DSP separation failed: %s", e, exc_info=True)
        return SeparationResult(success=False, error=str(e))


def separate_vocals_auto(
    data: AudioArray,
    samplerate: int,
    two_stems: bool = True
) -> SeparationResult:
    """
    Automatically select the best available separation method.
    Tries Demucs first, then torchaudio HDemucs, then Spleeter, then falls back to DSP.
    
    Args:
        data: Audio samples
        samplerate: Sample rate
        two_stems: Whether to return only vocals + instrumental
    
    Returns:
        SeparationResult with the best available separation
    """
    # Try demucs first
    if is_demucs_available():
        logger.info("Trying Demucs API separation...")
        result = separate_with_demucs(data, samplerate, two_stems)
        if result.success:
            return result
        logger.warning("Demucs API failed, trying alternatives...")

    # Try torchaudio HDemucs
    try:
        import torchaudio
        logger.info("Trying torchaudio HDemucs separation...")
        result = separate_with_torchaudio_hdemucs(data, samplerate, two_stems)
        if result.success:
            return result
        logger.warning("torchaudio HDemucs failed, trying alternatives...")
    except ImportError:
        pass
    
    # Try Spleeter (handles device internally via TensorFlow)
    if is_spleeter_available():
        logger.info("Trying Spleeter separation...")
        stems = 2 if two_stems else 4
        result = separate_with_spleeter(data, samplerate, stems)
        if result.success:
            return result
        logger.warning("Spleeter failed, falling back to DSP...")
    
    # Fallback to DSP
    logger.warning("No AI separation available, using DSP fallback...")
    return separate_with_dsp(data, samplerate)


def create_tracks_from_separation(
    result: SeparationResult,
    original_name: str,
    samplerate: int
) -> list[AudioTrack]:
    """
    Create AudioTrack objects from separation result.
    
    Args:
        result: SeparationResult from separation function
        original_name: Name of the original track
        samplerate: Sample rate
    
    Returns:
        List of AudioTrack objects
    """
    tracks = []
    
    if not result.success:
        return tracks
    
    for stem_name, stem_data in result.tracks:
        track = AudioTrack(name=f"{original_name} ({stem_name})")
        track.set_data(stem_data, samplerate)
        tracks.append(track)
    
    return tracks
