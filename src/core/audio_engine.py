import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from PyQt6.QtCore import QObject, pyqtSignal
from src.core.track import AudioTrack
from src.core.undo_manager import UndoManager
from src.core import effects
from src.utils.logger import logger

class AudioEngine(QObject):
    """
    Core engine for audio processing, playback, and track management.
    Uses sounddevice for low-latency output and librosa for robust file I/O.
    """
    positionChanged = pyqtSignal(float)
    stateChanged = pyqtSignal(str)
    tracksChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.tracks = [] 
        self.samplerate = 44100
        self.stream = None
        self.current_frame = 0
        self.is_playing = False
        self.clipboard = None 
        self.undo_manager = UndoManager()
        logger.info("AudioEngine initialized")

    # --- Track Management ---

    def add_track(self, track):
        """Adds a track and notifies listeners."""
        self.tracks.append(track)
        self.tracksChanged.emit()

    def remove_track(self, index):
        """Removes a track with undo support."""
        if 0 <= index < len(self.tracks):
            track = self.tracks[index]
            
            def undo():
                self.tracks.insert(index, track)
                self.tracksChanged.emit()
            
            def redo():
                if index < len(self.tracks) and self.tracks[index] == track:
                    del self.tracks[index]
                    self.tracksChanged.emit()

            self.undo_manager.push_action(f"Remove track {track.name}", undo, redo)
            del self.tracks[index]
            self.tracksChanged.emit()

    def clear_project(self):
        """Resets the engine to a clean state."""
        self.stop()
        self.tracks = []
        self.current_frame = 0
        self.undo_manager.clear()
        self.tracksChanged.emit()
        logger.info("Project cleared")

    # --- File I/O ---

    def load_file(self, file_path):
        """Loads an audio file using librosa and adds it as a new track."""
        logger.info(f"Loading file: {file_path}")
        try:
            import librosa
            
            # Match engine samplerate if tracks exist, else adopt file's rate
            target_sr = self.samplerate if self.tracks else None
            data, samplerate = librosa.load(file_path, sr=target_sr, mono=False)
            
            # Convert to (samples, channels)
            if data.ndim > 1:
                data = data.T
            
            data = data.astype(np.float32)
            
            if not self.tracks:
                self.samplerate = samplerate
            
            track_name = os.path.basename(file_path)
            new_track = AudioTrack(track_name)
            new_track.set_data(data, self.samplerate)
            
            self.add_track(new_track)
            return True
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}", exc_info=True)
            return False

    # --- Playback Control ---

    def play(self):
        """Starts audio playback."""
        if not self.tracks or self.is_playing:
            return

        self.is_playing = True
        self.stateChanged.emit('playing')
        
        max_samples = self.get_max_duration_samples()
        
        def playback_callback(outdata, frames, time, status):
            try:
                if status and status.output_underflow:
                    pass # Underflow handled by blocksize
                
                current_f = self.current_frame
                outdata.fill(0)
                
                if not self.tracks: return

                # Real-time Solo/Mute logic
                has_solo = any(t.soloed for t in self.tracks)
                active_tracks = [t for t in self.tracks if (t.soloed if has_solo else not t.muted)]

                chunk_end = current_f + frames
                
                for track in active_tracks:
                    data = track.data
                    if data is None: continue
                    
                    t_len = data.shape[0]
                    if current_f >= t_len: continue
                    
                    # Calculate slice
                    t_end = min(chunk_end, t_len)
                    out_end = t_end - current_f
                    
                    # Mix (optimized broadcasting)
                    if data.ndim == 1:
                        slice_data = data[current_f:t_end] * track.gain
                        outdata[:out_end, 0] += slice_data
                        outdata[:out_end, 1] += slice_data
                    else:
                        outdata[:out_end] += data[current_f:t_end] * track.gain
                
                # Prevent digital clipping
                np.clip(outdata, -1.0, 1.0, out=outdata)
                
                self.current_frame += frames
                if self.current_frame > max_samples + 22050:
                    raise sd.CallbackStop()
            except Exception:
                raise sd.CallbackStop()

        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=2,
            blocksize=4096,
            callback=playback_callback,
            finished_callback=self._on_stream_finished
        )
        self.stream.start()

    def pause(self):
        """Pauses playback."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.stateChanged.emit('paused')

    def stop(self):
        """Stops playback and resets position."""
        self.pause()
        self.current_frame = 0
        self.positionChanged.emit(0)
        self.stateChanged.emit('stopped')

    def seek(self, sample_index):
        """Moves the playhead to a specific sample."""
        total_len = self.get_max_duration_samples()
        self.current_frame = max(0, min(sample_index, total_len))
        self.positionChanged.emit(self.current_frame / self.samplerate)

    # --- Editing Operations ---

    def undo(self): return self.undo_manager.undo()
    def redo(self): return self.undo_manager.redo()

    def split_track(self, sample_index):
        """Adds a split point to all tracks."""
        any_mod = False
        for track in self.tracks:
            if track.data is not None and 0 < sample_index < len(track.data):
                if sample_index not in track.splits:
                    track.splits.append(sample_index)
                    track.splits.sort()
                    any_mod = True
        
        if any_mod:
            def undo():
                for track in self.tracks:
                    if sample_index in track.splits: track.splits.remove(sample_index)
                self.tracksChanged.emit()
            
            def redo():
                self.split_track(sample_index)
            
            self.undo_manager.push_action("Split track", undo, redo)
            self.tracksChanged.emit()
            return True
        return False

    def delete_range(self, start_sample, end_sample):
        """Deletes a range of audio from all tracks."""
        if not self.tracks: return False
        
        start = max(0, start_sample)
        old_states = [(t, t.data[start:min(len(t.data), end_sample)].copy() if t.data is not None and start < len(t.data) else None, t.splits.copy()) for t in self.tracks]

        any_mod = False
        for track in self.tracks:
            if track.data is None: continue
            end = min(len(track.data), end_sample)
            if start >= end: continue
            
            # Update splits
            diff = end - start
            track.splits = [s if s < start else s - diff for s in track.splits if s < start or s > end]
            track.data = np.delete(track.data, slice(start, end), axis=0)
            any_mod = True
            
        if any_mod:
            def undo():
                for i, (track, data, splits) in enumerate(old_states):
                    if data is not None:
                        track.data = np.concatenate((track.data[:start], data, track.data[start:]), axis=0)
                    track.splits = splits
                self.tracksChanged.emit()

            def redo():
                self.delete_range(start_sample, end_sample)

            self.undo_manager.push_action("Delete range", undo, redo)
            
            # Adjust playhead
            if start <= self.current_frame < end_sample:
                self.current_frame = start
            elif self.current_frame >= end_sample:
                self.current_frame -= (end_sample - start)
            
            self.tracksChanged.emit()
            return True
        return False

    # --- Effects & Presets ---

    def apply_effect(self, effect_name, track_index, start_sample, end_sample, **kwargs):
        """Applies a DSP effect to a specific track range."""
        if not (0 <= track_index < len(self.tracks)): return False
        track = self.tracks[track_index]
        if track.data is None: return False
        
        start, end = max(0, start_sample), min(len(track.data), end_sample)
        if start >= end: return False
        
        original_slice = track.data[start:end].copy()
        effect_func = getattr(effects, f"apply_{effect_name}", None)
        
        try:
            # Inject samplerate for time-based effects
            if effect_name in ['delay', 'reverb', 'lowpass', 'low_shelf']:
                kwargs['sr'] = self.samplerate
                
            modified_slice = effect_func(original_slice, **kwargs)
            track.data[start:end] = modified_slice
            
            self.undo_manager.push_action(f"Apply {effect_name}", 
                lambda: (setattr(track, 'data', np.concatenate((track.data[:start], original_slice, track.data[end:]), axis=0)), self.tracksChanged.emit()),
                lambda: self.apply_effect(effect_name, track_index, start_sample, end_sample, **kwargs))
            
            self.tracksChanged.emit()
            return True
        except Exception as e:
            logger.error(f"Effect error: {e}")
            return False

    def apply_preset(self, preset_name, track_index):
        """Applies a complex preset to a whole track."""
        if not (0 <= track_index < len(self.tracks)): return False
        track = self.tracks[track_index]
        
        original_data, original_splits = track.data.copy(), track.splits.copy()
        
        try:
            if preset_name == "slowed_reverb":
                track.data = effects.apply_resample(track.data, 0.8)
                track.data = effects.apply_reverb(track.data, self.samplerate, room_size=0.8)
                track.data = effects.apply_lowpass(track.data, self.samplerate, cutoff=2500)
            elif preset_name == "nightcore":
                track.data = effects.apply_resample(track.data, 1.25)
            elif preset_name == "lofi":
                track.data = effects.apply_lowpass(track.data, self.samplerate, cutoff=1500)
                track.data = effects.apply_gain(track.data, 0.8)
            elif preset_name == "bass_boosted":
                track.data = effects.apply_low_shelf(track.data, self.samplerate, cutoff=150, gain_db=12.0)
                track.data = effects.apply_soft_clip(track.data, threshold=0.85)
            
            track.splits = [] # Timeline changed
            self.undo_manager.push_action(f"Preset: {preset_name}", 
                lambda: (setattr(track, 'data', original_data), setattr(track, 'splits', original_splits), self.tracksChanged.emit()),
                lambda: self.apply_preset(preset_name, track_index))
            
            self.tracksChanged.emit()
            return True
        except Exception as e:
            logger.error(f"Preset error: {e}")
            return False

    # --- Utilities ---

    def get_max_duration_samples(self):
        return max([len(t.data) for t in self.tracks if t.data is not None] or [0])
        
    def get_duration(self):
        return self.get_max_duration_samples() / self.samplerate

    def mix_down(self):
        """Mixes all tracks into a single stereo buffer."""
        max_len = self.get_max_duration_samples()
        if max_len == 0: return None
        
        master = np.zeros((max_len, 2), dtype=np.float32)
        has_solo = any(t.soloed for t in self.tracks)
        active_tracks = [t for t in self.tracks if (t.soloed if has_solo else not t.muted)]
        
        for track in active_tracks:
            if track.data is None: continue
            t_len = len(track.data)
            if track.data.ndim == 1:
                master[:t_len, 0] += track.data * track.gain
                master[:t_len, 1] += track.data * track.gain
            else:
                master[:t_len] += track.data * track.gain
                
        return np.clip(master, -1.0, 1.0)

    def _on_stream_finished(self):
        self.is_playing = False
        self.stateChanged.emit('stopped')

    # --- AI Tools ---

    def separate_hpss(self, track_index):
        """Harmonic/Percussive separation."""
        if not (0 <= track_index < len(self.tracks)): return False
        track = self.tracks[track_index]
        import librosa
        
        logger.info("Running HPSS...")
        y = track.data
        if y.ndim > 1:
            h0, p0 = librosa.effects.hpss(y[:, 0])
            h1, p1 = librosa.effects.hpss(y[:, 1])
            harm, perc = np.column_stack((h0, h1)), np.column_stack((p0, p1))
        else:
            harm, perc = librosa.effects.hpss(y)
            
        self.add_track(AudioTrack(f"{track.name} (Harmonic)").set_data(harm, self.samplerate))
        self.add_track(AudioTrack(f"{track.name} (Percussive)").set_data(perc, self.samplerate))
        return True

    def separate_vocals_dsp(self, track_index):
        """Center channel cancellation."""
        if not (0 <= track_index < len(self.tracks)): return False
        track = self.tracks[track_index]
        if track.data.ndim == 1: return False
        
        L, R = track.data[:, 0], track.data[:, 1]
        inst_mono = (L - R) / 2.0
        vocal_mono = (L + R) / 2.0
        
        self.add_track(AudioTrack(f"{track.name} (Instrumental)").set_data(np.column_stack((inst_mono, inst_mono)), self.samplerate))
        self.add_track(AudioTrack(f"{track.name} (Vocals/Center)").set_data(np.column_stack((vocal_mono, vocal_mono)), self.samplerate))
        return True

    def separate_ai_demucs(self, track_index):
        logger.warning("Demucs requires external installation.")
        return False

