"""
Core audio engine for PyAudioEditor.
Orchestrates project management, editing operations, and playback.
Simplified by delegating responsibilities to specialized modules.
"""
from __future__ import annotations
import logging
import os
from typing import Optional, Callable
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from .config import AUDIO_CONFIG, PlaybackState
from .types import AudioArray, ClipboardData
from .project import Project
from .track import AudioTrack
from .playback import PlaybackController
from . import effects_basic as fx
from . import effects_vocal as fx_vocal
from . import separation

logger = logging.getLogger("PyAudacity")


class AudioEngine(QObject):
    """
    Core engine for audio processing, playback, and track management.
    Uses composition to delegate specialized tasks to focused modules.
    """
    
    # Qt Signals
    positionChanged = pyqtSignal(float)  # Emits current time in seconds
    stateChanged = pyqtSignal(str)       # Emits 'playing', 'paused', 'stopped'
    tracksChanged = pyqtSignal()         # Emits when tracks are added/removed/modified
    
    def __init__(self) -> None:
        super().__init__()
        
        # Core state
        self.project = Project()
        
        # Playback controller with signal bridges
        self._playback = PlaybackController(
            self.project,
            on_position_changed=self._on_position_changed,
            on_state_changed=self._on_state_changed
        )
        
        logger.info("AudioEngine initialized")
    
    # =========================================================================
    # PROPERTY ACCESSORS (for backward compatibility)
    # =========================================================================
    
    @property
    def tracks(self) -> list[AudioTrack]:
        """Get all tracks in the project."""
        return self.project.tracks
    
    @property
    def samplerate(self) -> int:
        """Get project sample rate."""
        return self.project.samplerate
    
    @samplerate.setter
    def samplerate(self, value: int) -> None:
        """Set project sample rate."""
        self.project.samplerate = value
    
    @property
    def current_frame(self) -> int:
        """Get current playback position in samples."""
        return self._playback.current_frame
    
    @current_frame.setter
    def current_frame(self, value: int) -> None:
        """Set current playback position."""
        self._playback.current_frame = value
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is playing."""
        return self._playback.is_playing
    
    @property
    def undo_manager(self):
        """Get undo manager from project."""
        return self.project.undo_manager
    
    @property
    def clipboard(self) -> Optional[ClipboardData]:
        """Get clipboard data."""
        return self.project.clipboard
    
    @clipboard.setter
    def clipboard(self, value: Optional[ClipboardData]) -> None:
        """Set clipboard data."""
        self.project.clipboard = value
    
    # =========================================================================
    # SIGNAL BRIDGES
    # =========================================================================
    
    def _on_position_changed(self, seconds: float) -> None:
        """Bridge position changes to Qt signal."""
        self.positionChanged.emit(seconds)
    
    def _on_state_changed(self, state: PlaybackState) -> None:
        """Bridge state changes to Qt signal."""
        state_map = {
            PlaybackState.PLAYING: 'playing',
            PlaybackState.PAUSED: 'paused',
            PlaybackState.STOPPED: 'stopped'
        }
        self.stateChanged.emit(state_map.get(state, 'stopped'))
    
    # =========================================================================
    # TRACK MANAGEMENT
    # =========================================================================
    
    def add_track(self, track: AudioTrack) -> int:
        """Add a track to the project."""
        index = self.project.add_track(track)
        self.tracksChanged.emit()
        return index
    
    def remove_track(self, index: int) -> bool:
        """Remove a track with undo support."""
        track = self.project.get_track(index)
        if track is None:
            return False
        
        def undo() -> None:
            self.project.tracks.insert(index, track)
            self.tracksChanged.emit()
        
        def redo() -> None:
            if index < len(self.project.tracks) and self.project.tracks[index] == track:
                del self.project.tracks[index]
                self.tracksChanged.emit()
        
        self.undo_manager.push_action(f"Remove track {track.name}", undo, redo)
        
        self.project.remove_track(index)
        self.tracksChanged.emit()
        return True
    
    def get_max_duration_samples(self) -> int:
        """Get maximum duration across all tracks in samples."""
        return self.project.duration_samples
    
    def get_duration(self) -> float:
        """Get maximum duration across all tracks in seconds."""
        return self.project.duration_seconds
    
    def clear_project(self) -> None:
        """Reset the engine to a clean state."""
        self.stop()
        self.project.clear()
        self.tracksChanged.emit()
        logger.info("Project cleared")
    
    # =========================================================================
    # FILE I/O
    # =========================================================================
    
    def load_file(self, file_path: str) -> bool:
        """Load an audio file and add it as a new track."""
        logger.info("Loading file: %s", file_path)
        
        try:
            import librosa
            
            # Match project samplerate if tracks exist
            target_sr = self.samplerate if self.tracks else None
            data, loaded_sr = librosa.load(file_path, sr=target_sr, mono=False)
            
            # Convert to (samples, channels)
            if data.ndim > 1:
                data = data.T
            
            data = data.astype(np.float32)
            
            # Update project samplerate if first track
            if not self.tracks:
                self.project.samplerate = loaded_sr
            
            track_name = os.path.basename(file_path)
            new_track = AudioTrack(name=track_name)
            new_track.set_data(data, self.samplerate)
            
            self.add_track(new_track)
            return True
            
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path, e, exc_info=True)
            return False
    
    def mix_down(self) -> Optional[AudioArray]:
        """Mix all tracks to a single stereo array."""
        return self.project.mix_down()
    
    # =========================================================================
    # PLAYBACK CONTROL
    # =========================================================================
    
    def play(self) -> bool:
        """Start audio playback."""
        return self._playback.play()
    
    def pause(self) -> None:
        """Pause playback."""
        self._playback.pause()
    
    def stop(self) -> None:
        """Stop playback and reset position."""
        self._playback.stop()
    
    def seek(self, sample_index: int) -> None:
        """Move the playhead to a specific sample."""
        self._playback.seek(sample_index)
        self.positionChanged.emit(self._playback.current_time)
    
    def toggle_play_pause(self) -> None:
        """Toggle between play and pause."""
        self._playback.toggle_play_pause()
    
    # =========================================================================
    # UNDO/REDO
    # =========================================================================
    
    def undo(self) -> bool:
        """Undo the last action."""
        return self.undo_manager.undo()
    
    def redo(self) -> bool:
        """Redo the last undone action."""
        return self.undo_manager.redo()
    
    # =========================================================================
    # EDITING OPERATIONS
    # =========================================================================
    
    def split_track(self, sample_index: int) -> bool:
        """Add a split point to all tracks at the given sample."""
        any_modified = False
        
        for track in self.tracks:
            if track.add_split(sample_index):
                any_modified = True
        
        if any_modified:
            def undo() -> None:
                for track in self.tracks:
                    track.remove_split(sample_index)
                self.tracksChanged.emit()
            
            def redo() -> None:
                self.split_track(sample_index)
            
            self.undo_manager.push_action("Split track", undo, redo)
            self.tracksChanged.emit()
        
        return any_modified
    
    def delete_range(self, start_sample: int, end_sample: int) -> bool:
        """Delete a range of audio from all tracks."""
        if not self.tracks:
            return False
        
        start = max(0, start_sample)
        
        # Store state for undo
        old_states = [
            (track, track.data[start:min(len(track.data), end_sample)].copy() 
             if track.data is not None and start < len(track.data) else None, 
             track.splits.copy())
            for track in self.tracks
        ]
        
        any_modified = False
        
        for track in self.tracks:
            if track.data is None:
                continue
            
            end = min(len(track.data), end_sample)
            if start >= end:
                continue
            
            # Update splits
            diff = end - start
            track.splits = [
                s if s < start else s - diff 
                for s in track.splits if s < start or s > end
            ]
            
            track.data = np.delete(track.data, slice(start, end), axis=0)
            any_modified = True
        
        if any_modified:
            def undo() -> None:
                for track, data, splits in old_states:
                    if data is not None and track.data is not None:
                        track.data = np.concatenate(
                            (track.data[:start], data, track.data[start:]), 
                            axis=0
                        )
                    track.splits = splits
                self.tracksChanged.emit()
            
            def redo() -> None:
                self.delete_range(start_sample, end_sample)
            
            self.undo_manager.push_action("Delete range", undo, redo)
            
            # Adjust playhead
            if start <= self.current_frame < end_sample:
                self.current_frame = start
            elif self.current_frame >= end_sample:
                self.current_frame -= (end_sample - start)
            
            self.tracksChanged.emit()
        
        return any_modified
    
    def cut_range(self, start_sample: int, end_sample: int) -> bool:
        """Cut a range of audio to clipboard."""
        self.copy_range(start_sample, end_sample)
        return self.delete_range(start_sample, end_sample)
    
    def copy_range(self, start_sample: int, end_sample: int) -> bool:
        """Copy a range of audio to clipboard."""
        if not self.tracks:
            return False
        
        start = max(0, start_sample)
        
        clipboard_data = []
        for track in self.tracks:
            if track.data is None:
                clipboard_data.append(None)
            else:
                track_end = min(end_sample, len(track.data))
                if start < track_end:
                    clipboard_data.append(track.data[start:track_end].copy())
                else:
                    clipboard_data.append(None)
        
        self.project.clipboard = {
            'data': clipboard_data,
            'samplerate': self.samplerate
        }
        
        logger.info("Copied range %d-%d to clipboard", start, end_sample)
        return True
    
    def paste_at(self, position: int) -> bool:
        """Paste clipboard content at position."""
        if not self.clipboard or not self.clipboard.get('data'):
            return False
        
        if not self.tracks:
            return False
        
        # Check samplerate compatibility
        if self.clipboard.get('samplerate') != self.samplerate:
            logger.warning(
                "Clipboard samplerate %s != project samplerate %s",
                self.clipboard.get('samplerate'), self.samplerate
            )
        
        clipboard_data = self.clipboard['data']
        
        # Store state for undo
        old_states = [
            (track, track.data.copy() if track.data is not None else None, track.splits.copy())
            for track in self.tracks
        ]
        
        any_pasted = False
        
        for i, track in enumerate(self.tracks):
            if i >= len(clipboard_data) or clipboard_data[i] is None:
                continue
            
            if track.data is None:
                continue
            
            paste_data = clipboard_data[i]
            pos = min(position, len(track.data))
            
            # Insert paste data at position
            track.data = np.concatenate([
                track.data[:pos],
                paste_data,
                track.data[pos:]
            ], axis=0)
            
            # Update splits
            paste_len = len(paste_data)
            track.splits = [s if s < pos else s + paste_len for s in track.splits]
            
            any_pasted = True
        
        if any_pasted:
            def undo() -> None:
                for track, old_data, old_splits in old_states:
                    if old_data is not None:
                        track.data = old_data
                    track.splits = old_splits
                self.tracksChanged.emit()
            
            def redo() -> None:
                self.paste_at(position)
            
            self.undo_manager.push_action("Paste", undo, redo)
            self.tracksChanged.emit()
        
        return any_pasted
    
    # =========================================================================
    # EFFECTS
    # =========================================================================
    
    def apply_effect(
        self,
        track_index: int,
        effect_func: Callable[[AudioArray, int], AudioArray],
        effect_name: str,
        start_sample: Optional[int] = None,
        end_sample: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Apply an effect to a track or selection.
        
        Args:
            track_index: Index of target track
            effect_func: Effect function from effects_basic or effects_vocal
            effect_name: Human-readable effect name for undo
            start_sample: Start of selection (None = entire track)
            end_sample: End of selection (None = entire track)
            **kwargs: Additional parameters for the effect
        
        Returns:
            True if effect was applied successfully
        """
        track = self.project.get_track(track_index)
        if track is None or track.data is None:
            return False
        
        # Determine range
        start = start_sample if start_sample is not None else 0
        end = end_sample if end_sample is not None else len(track.data)
        start = max(0, start)
        end = min(len(track.data), end)
        
        if start >= end:
            return False
        
        # Store for undo
        old_data = track.data[start:end].copy()
        
        try:
            # Apply effect
            segment = track.data[start:end]
            if kwargs:
                processed = effect_func(segment, self.samplerate, **kwargs)
            else:
                processed = effect_func(segment, self.samplerate)
            
            # Handle length changes
            if len(processed) != len(old_data):
                # Effect changed length - replace entire segment
                track.data = np.concatenate([
                    track.data[:start],
                    processed,
                    track.data[end:]
                ], axis=0)
            else:
                track.data[start:end] = processed
            
            # Create undo action
            def undo() -> None:
                if len(processed) != len(old_data):
                    # Restore original length
                    t = self.project.get_track(track_index)
                    if t is not None and t.data is not None:
                        t.data = np.concatenate([
                            t.data[:start],
                            old_data,
                            t.data[start + len(processed):]
                        ], axis=0)
                else:
                    t = self.project.get_track(track_index)
                    if t is not None and t.data is not None:
                        t.data[start:end] = old_data
                self.tracksChanged.emit()
            
            def redo() -> None:
                self.apply_effect(track_index, effect_func, effect_name, start_sample, end_sample, **kwargs)
            
            self.undo_manager.push_action(f"Apply {effect_name}", undo, redo)
            self.tracksChanged.emit()
            
            return True
            
        except Exception as e:
            logger.error("Failed to apply effect %s: %s", effect_name, e, exc_info=True)
            return False
    
    def apply_effect_all_tracks(
        self,
        effect_func: Callable[[AudioArray, int], AudioArray],
        effect_name: str,
        start_sample: Optional[int] = None,
        end_sample: Optional[int] = None,
        **kwargs
    ) -> bool:
        """Apply an effect to all tracks."""
        success = False
        for i in range(len(self.tracks)):
            if self.apply_effect(i, effect_func, effect_name, start_sample, end_sample, **kwargs):
                success = True
        return success
    
    # =========================================================================
    # VOCAL SEPARATION
    # =========================================================================
    
    def separate_vocals(self, track_index: int, two_stems: bool = True) -> bool | str:
        """
        Separate vocals from a track using the best available backend.
        
        Args:
            track_index: Index of track to separate
            two_stems: If True, return vocals + instrumental only
        
        Returns:
            True if separation was successful, or error message string
        """
        track = self.project.get_track(track_index)
        if track is None or track.data is None:
            return "Track not found or empty."
        
        result = separation.separate_vocals_auto(track.data, self.samplerate, two_stems)
        
        if not result.success:
            logger.error("Vocal separation failed: %s", result.error)
            return result.error or "Unknown separation error."
        
        # Create tracks from result
        new_tracks = separation.create_tracks_from_separation(
            result, 
            track.name, 
            self.samplerate
        )
        
        for new_track in new_tracks:
            self.add_track(new_track)
        
        return True
    
    def separate_with_demucs(self, track_index: int, two_stems: bool = True) -> bool | str:
        """Separate using Demucs specifically."""
        track = self.project.get_track(track_index)
        if track is None or track.data is None:
            return "Track not found or empty."
        
        result = separation.separate_with_demucs(track.data, self.samplerate, two_stems)
        
        if not result.success:
            logger.error("Demucs separation failed: %s", result.error)
            return result.error or "Unknown Demucs error."
        
        new_tracks = separation.create_tracks_from_separation(result, track.name, self.samplerate)
        for new_track in new_tracks:
            self.add_track(new_track)
        
        return True
    
    def separate_with_spleeter(self, track_index: int, stems: int = 2) -> bool | str:
        """Separate using Spleeter specifically."""
        track = self.project.get_track(track_index)
        if track is None or track.data is None:
            return "Track not found or empty."
        
        result = separation.separate_with_spleeter(track.data, self.samplerate, stems)
        
        if not result.success:
            logger.error("Spleeter separation failed: %s", result.error)
            return result.error or "Unknown Spleeter error."
        
        new_tracks = separation.create_tracks_from_separation(result, track.name, self.samplerate)
        for new_track in new_tracks:
            self.add_track(new_track)
        
        return True
    
    # Legacy method aliases for backward compatibility
    def separate_ai_demucs(self, track_index: int, two_stems: bool = True) -> bool | str:
        return self.separate_with_demucs(track_index, two_stems)
    
    def separate_vocals_spleeter(self, track_index: int, stems: int = 2) -> bool | str:
        return self.separate_with_spleeter(track_index, stems)
    
    def separate_vocals_auto(self, track_index: int, two_stems: bool = True) -> bool | str:
        return self.separate_vocals(track_index, two_stems)
    
    def separate_vocals_dsp(self, track_index: int) -> bool | str:
        """Separate using DSP fallback."""
        track = self.project.get_track(track_index)
        if track is None or track.data is None:
            return "Track not found or empty."
        
        result = separation.separate_with_dsp(track.data, self.samplerate)
        
        if not result.success:
            return result.error or "DSP separation failed."
        
        new_tracks = separation.create_tracks_from_separation(result, track.name, self.samplerate)
        for new_track in new_tracks:
            self.add_track(new_track)
        
        return True
    
    def separate_hpss(self, track_index: int) -> bool | str:
        """
        Separate track using Harmonic-Percussive Source Separation (HPSS).
        Fast DSP-based method using librosa.
        
        Args:
            track_index: Index of track to separate
        
        Returns:
            True if separation was successful, or error message string
        """
        track = self.project.get_track(track_index)
        if track is None or track.data is None:
            return "Track not found or empty."
        
        try:
            import librosa
            
            logger.info("Starting HPSS separation...")
            
            # Get mono for processing
            if track.data.ndim > 1:
                mono = np.mean(track.data, axis=1)
            else:
                mono = track.data
            
            # Apply HPSS
            harmonic, percussive = librosa.effects.hpss(mono)
            
            # Create stereo versions
            harmonic_stereo = np.column_stack((harmonic, harmonic)).astype(np.float32)
            percussive_stereo = np.column_stack((percussive, percussive)).astype(np.float32)
            
            # Create new tracks
            harmonic_track = AudioTrack(name=f"{track.name} (Harmonic)")
            harmonic_track.set_data(harmonic_stereo, self.samplerate)
            
            percussive_track = AudioTrack(name=f"{track.name} (Percussive)")
            percussive_track.set_data(percussive_stereo, self.samplerate)
            
            self.add_track(harmonic_track)
            self.add_track(percussive_track)
            
            logger.info("HPSS separation completed successfully")
            return True
            
        except ImportError:
            logger.error("librosa not available for HPSS")
            return "librosa not available. Please install it with 'pip install librosa'."
        except Exception as e:
            logger.error("HPSS separation failed: %s", e, exc_info=True)
            return str(e)
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._playback.cleanup()
        logger.info("AudioEngine cleaned up")
