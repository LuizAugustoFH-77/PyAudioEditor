"""
Playback controller for PyAudioEditor.
Handles audio output via sounddevice with low-latency streaming.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, Callable
import numpy as np
import sounddevice as sd

from .config import AUDIO_CONFIG, PlaybackState
from .types import AudioArray

if TYPE_CHECKING:
    from .project import Project

logger = logging.getLogger("PyAudacity")


class PlaybackController:
    """
    Controls audio playback using sounddevice.
    Optimized for low-latency real-time streaming.
    """
    __slots__ = (
        '_project', '_stream', '_current_frame', '_state',
        '_on_position_changed', '_on_state_changed', '_max_samples', '_disposed'
    )
    
    def __init__(
        self,
        project: "Project",
        on_position_changed: Optional[Callable[[float], None]] = None,
        on_state_changed: Optional[Callable[[PlaybackState], None]] = None
    ) -> None:
        """
        Initialize playback controller.
        
        Args:
            project: Project instance to play from
            on_position_changed: Callback for position updates (seconds)
            on_state_changed: Callback for state changes
        """
        self._project = project
        self._stream: Optional[sd.OutputStream] = None
        self._current_frame: int = 0
        self._state = PlaybackState.STOPPED
        self._on_position_changed = on_position_changed
        self._on_state_changed = on_state_changed
        self._max_samples: int = 0
        self._disposed: bool = False
    
    @property
    def current_frame(self) -> int:
        """Current playback position in samples."""
        return self._current_frame
    
    @current_frame.setter
    def current_frame(self, value: int) -> None:
        """Set current playback position."""
        self._current_frame = max(0, value)
    
    @property
    def current_time(self) -> float:
        """Current playback position in seconds."""
        sr = self._project.samplerate
        return self._current_frame / sr if sr > 0 else 0.0
    
    @property
    def state(self) -> PlaybackState:
        """Current playback state."""
        return self._state
    
    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._state == PlaybackState.PLAYING
    
    def _set_state(self, state: PlaybackState) -> None:
        """Update state and notify callback."""
        if self._disposed:
            self._state = state
            return
        if self._state != state:
            self._state = state
            if self._on_state_changed:
                self._on_state_changed(state)
    
    def _notify_position(self) -> None:
        """Notify position callback."""
        if self._disposed:
            return
        if self._on_position_changed:
            self._on_position_changed(self.current_time)
    
    def play(self) -> bool:
        """
        Start audio playback.
        
        Returns:
            True if playback started successfully
        """
        if self._disposed:
            return False

        if not self._project.tracks or self.is_playing:
            return False
        
        self._max_samples = self._project.duration_samples
        if self._max_samples == 0:
            return False
        
        self._set_state(PlaybackState.PLAYING)
        
        def playback_callback(
            outdata: np.ndarray,
            frames: int,
            time: object,
            status: sd.CallbackFlags
        ) -> None:
            """Real-time audio callback."""
            try:
                if status and status.output_underflow:
                    pass  # Handled by blocksize
                
                current_f = self._current_frame
                outdata.fill(0)
                
                if not self._project.tracks:
                    return
                
                # Get active tracks (respecting solo/mute)
                active_tracks = self._project.get_active_tracks()
                chunk_end = current_f + frames
                
                for track in active_tracks:
                    # Optimized: use clips directly to avoid expensive data property
                    # and correctly handle timing/offsets
                    if hasattr(track, 'clips') and track.clips:
                        for clip in track.clips:
                            c_start = clip.start_offset
                            c_data = clip.data
                            c_len = len(c_data)
                            c_end = c_start + c_len
                            
                            # Check intersection with current chunk
                            intersect_start = max(c_start, current_f)
                            intersect_end = min(c_end, chunk_end)
                            
                            if intersect_end > intersect_start:
                                # Map to buffer indices
                                clip_offset = intersect_start - c_start
                                clip_len = intersect_end - intersect_start
                                
                                out_offset = intersect_start - current_f
                                
                                # Mix
                                segment = c_data[clip_offset:clip_offset + clip_len]
                                gain = track.gain
                                
                                if segment.ndim == 1:
                                    # Mono to stereo
                                    mixed = segment * gain
                                    outdata[out_offset:out_offset + clip_len, 0] += mixed
                                    outdata[out_offset:out_offset + clip_len, 1] += mixed
                                else:
                                    # Stereo
                                    outdata[out_offset:out_offset + clip_len] += segment * gain
                    
                    elif track.data is not None:
                         # Legacy fallback: use data + start_offset
                         data = track.data
                         offset = track.start_offset
                         t_len = len(data)
                         
                         # Effective start/end of track in timeline
                         t_start = offset
                         t_end_abs = offset + t_len
                         
                         intersect_start = max(t_start, current_f)
                         intersect_end = min(t_end_abs, chunk_end)
                         
                         if intersect_end > intersect_start:
                             d_start = intersect_start - offset
                             d_len = intersect_end - intersect_start
                             out_offset = intersect_start - current_f
                             
                             segment = data[d_start:d_start + d_len]
                             gain = track.gain
                             
                             if segment.ndim == 1:
                                 mixed = segment * gain
                                 outdata[out_offset:out_offset + d_len, 0] += mixed
                                 outdata[out_offset:out_offset + d_len, 1] += mixed
                             else:
                                 outdata[out_offset:out_offset + d_len] += segment * gain
                
                # Prevent digital clipping
                np.clip(outdata, -1.0, 1.0, out=outdata)
                
                self._current_frame += frames
                
                # Stop at end with margin
                if self._current_frame > self._max_samples + AUDIO_CONFIG.playback_end_margin_samples:
                    raise sd.CallbackStop()
                    
            except Exception as e:
                logger.error("Playback callback error: %s", e, exc_info=True)
                raise sd.CallbackStop()
        
        def on_finished() -> None:
            """Called when stream finishes."""
            # finished_callback may fire during shutdown; never call user callbacks then.
            if self._disposed:
                return
            if self._state == PlaybackState.PLAYING:
                self._set_state(PlaybackState.STOPPED)
                self._current_frame = 0
                self._notify_position()
        
        try:
            self._stream = sd.OutputStream(
                samplerate=self._project.samplerate,
                channels=AUDIO_CONFIG.playback_channels,
                blocksize=AUDIO_CONFIG.playback_blocksize,
                callback=playback_callback,
                finished_callback=on_finished
            )
            self._stream.start()
            logger.info("Playback started at frame %d", self._current_frame)
            return True
            
        except Exception as e:
            logger.error("Failed to start playback: %s", e, exc_info=True)
            self._set_state(PlaybackState.STOPPED)
            return False
    
    def pause(self) -> None:
        """Pause playback (keep position)."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error stopping stream: %s", e)
            self._stream = None
        
        self._set_state(PlaybackState.PAUSED)
        logger.info("Playback paused at frame %d", self._current_frame)
    
    def stop(self) -> None:
        """Stop playback and reset position."""
        self.pause()
        self._current_frame = 0
        self._set_state(PlaybackState.STOPPED)
        self._notify_position()
        logger.info("Playback stopped")
    
    def seek(self, sample_index: int) -> None:
        """
        Move playhead to specific sample position.
        
        Args:
            sample_index: Target position in samples
        """
        total_len = self._project.duration_samples
        self._current_frame = max(0, min(sample_index, total_len))
        self._notify_position()
    
    def seek_seconds(self, seconds: float) -> None:
        """
        Move playhead to specific time position.
        
        Args:
            seconds: Target position in seconds
        """
        sample_index = int(seconds * self._project.samplerate)
        self.seek(sample_index)
    
    def toggle_play_pause(self) -> None:
        """Toggle between play and pause states."""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Mark disposed first so finished_callback can't touch Qt objects.
        self._disposed = True

        # Drop external callbacks to avoid calling into deleted Qt objects.
        self._on_position_changed = None
        self._on_state_changed = None

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._state = PlaybackState.STOPPED
