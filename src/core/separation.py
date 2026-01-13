"""
AI-powered vocal separation backends for PyAudioEditor.
Supports Demucs, Spleeter, and DSP-based fallback.
"""
from __future__ import annotations
import logging
import tempfile
import os
from typing import Optional
import numpy as np

from .types import AudioArray, SeparationResult
from .track import AudioTrack

logger = logging.getLogger("PyAudacity")


# =============================================================================
# CAPABILITY CHECKS
# =============================================================================

def is_demucs_available() -> bool:
    """Check if Demucs is available."""
    try:
        import torch
        import demucs.api
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
    Best quality but requires PyTorch and GPU recommended.
    
    Args:
        data: Audio samples
        samplerate: Sample rate
        two_stems: If True, return vocals + instrumental. If False, return all stems.
    
    Returns:
        SeparationResult with success status and extracted tracks
    """
    try:
        import torch
        import demucs.api
        
        logger.info("Starting Demucs separation (two_stems=%s)...", two_stems)
        
        # Ensure stereo
        if data.ndim == 1:
            audio_stereo = np.column_stack((data, data))
        else:
            audio_stereo = data
        
        # Demucs expects (channels, samples) - shape: (2, num_samples)
        audio_tensor = torch.from_numpy(audio_stereo.T.astype(np.float32))
        
        # Initialize separator with htdemucs model
        separator = demucs.api.Separator(model="htdemucs", progress=True)
        
        logger.info("Demucs model loaded, sample rate: %d", separator.samplerate)
        
        # separate_tensor expects (channels, samples) and returns:
        # - origin: the resampled original waveform
        # - separated: dict with stem names as keys and tensors as values
        origin, separated = separator.separate_tensor(audio_tensor, sr=samplerate)
        
        logger.info("Separation complete, stems: %s", list(separated.keys()))
        
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
        logger.error("Demucs separation failed: %s", e, exc_info=True)
        return SeparationResult(success=False, error=str(e))


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
        import torch
        import torchaudio
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
        
        logger.info("Starting torchaudio HDemucs separation...")
        
        # Ensure stereo
        if data.ndim == 1:
            audio_stereo = np.column_stack((data, data))
        else:
            audio_stereo = data
        
        # (channels, samples)
        waveform = torch.from_numpy(audio_stereo.T.astype(np.float32))
        
        # Get the bundle
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model = bundle.get_model()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Resample if needed
        model_sample_rate = bundle.sample_rate
        if samplerate != model_sample_rate:
            resampler = torchaudio.transforms.Resample(samplerate, model_sample_rate)
            waveform = resampler(waveform)
        
        waveform = waveform.to(device)
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0)  # (1, channels, samples)
        
        # Run model
        with torch.no_grad():
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
        from spleeter.separator import Separator
        import soundfile as sf
        from scipy.signal import resample
        
        logger.info("Starting Spleeter %d-stem separation...", stems)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_dir = os.path.join(tmpdir, "output")
            
            # Ensure stereo
            audio_data = data
            if audio_data.ndim == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            sf.write(input_path, audio_data, samplerate)
            
            # Run separation
            separator = Separator(f'spleeter:{stems}stems')
            separator.separate_to_file(input_path, output_dir)
            
            # Load separated stems
            stem_folder = os.path.join(output_dir, "input")
            
            if stems == 2:
                stem_files = ['vocals.wav', 'accompaniment.wav']
                stem_names = ['Vocals', 'Instrumental']
            elif stems == 4:
                stem_files = ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']
                stem_names = ['Vocals', 'Drums', 'Bass', 'Other']
            else:  # 5 stems
                stem_files = ['vocals.wav', 'drums.wav', 'bass.wav', 'piano.wav', 'other.wav']
                stem_names = ['Vocals', 'Drums', 'Bass', 'Piano', 'Other']
            
            tracks: list[tuple[str, AudioArray]] = []
            
            for stem_file, stem_name in zip(stem_files, stem_names):
                stem_path = os.path.join(stem_folder, stem_file)
                if os.path.exists(stem_path):
                    stem_data, stem_sr = sf.read(stem_path)
                    
                    # Resample if needed
                    if stem_sr != samplerate:
                        target_len = int(len(stem_data) * samplerate / stem_sr)
                        if stem_data.ndim > 1:
                            resampled = np.zeros((target_len, stem_data.shape[1]))
                            for ch in range(stem_data.shape[1]):
                                resampled[:, ch] = resample(stem_data[:, ch], target_len)
                            stem_data = resampled
                        else:
                            stem_data = resample(stem_data, target_len)
                    
                    tracks.append((stem_name, stem_data.astype(np.float32)))
        
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
    # Try demucs.api first
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
    
    # Try Spleeter
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
