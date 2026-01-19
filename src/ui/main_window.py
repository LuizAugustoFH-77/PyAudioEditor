"""
Main window for PyAudioEditor.
Provides the primary user interface for audio editing.
"""
from __future__ import annotations
import logging
import os
import time
from typing import Optional, Callable, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollBar, QScrollArea,
    QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox,
    QInputDialog, QMessageBox, QProgressBar, QFrame, QListWidget, QSplitter, QTextBrowser
)
from PyQt6.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QCloseEvent, QGuiApplication, QResizeEvent
import qtawesome as qta

from src.core.audio_engine import AudioEngine
from src.core.track import AudioTrack
from src.core.types import SeparationResult
from src.core import separation as separation_core
from src.core import effects_basic as fx
from src.core import effects_vocal as fx_vocal
from .time_ruler import TimeRulerWidget
from .track_widget import TrackWidget
from .waveform_view import WaveformWidget

logger = logging.getLogger("PyAudacity")

# --- Worker Thread for Background Processing ---
class AudioProcessingWorker(QThread):
    progress = pyqtSignal(int, str, float)  # percent, status, eta_seconds
    finished = pyqtSignal(bool, object)     # success, result/error_msg

    def __init__(self, target_func, *args, **kwargs):
        super().__init__()
        self.target_func = target_func
        self.args = args
        self.kwargs = kwargs
        self.start_time = 0

    def run(self):
        try:
            self.start_time = time.time()
            # Handle progress callback injection if supported mainly for my implementation
            # For now running directly
            result = self.target_func(*self.args, **self.kwargs)
            self.finished.emit(True, result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))

    def update_progress(self, current, total, status="Processing..."):
        if total <= 0: return
        percent = int((current / total) * 100)
        elapsed = time.time() - self.start_time
        # Simple ETA
        eta = (elapsed / current) * (total - current) if current > 0 else 0
        self.progress.emit(percent, status, eta)


# --- Progress Dialog ---
class ProgressDialog(QDialog):
    def __init__(self, title="Processing Audio", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(420, 170)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        
        layout = QVBoxLayout(self)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-weight: 600; margin-bottom: 6px;")
        
        self.bar = QProgressBar()
        self.bar.setRange(0, 0) # Indeterminate by default
        self.bar.setTextVisible(True)
        
        self.eta_label = QLabel("Estimating time...")
        self.eta_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.eta_label.setStyleSheet("color: #9aa3ad;")
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.bar)
        layout.addWidget(self.eta_label)
        layout.addStretch()

    def update_status(self, percent, status, eta_secs):
        if self.bar.maximum() == 0:
            self.bar.setRange(0, 100)
            
        self.bar.setValue(percent)
        self.status_label.setText(status)
        
        if eta_secs > 0:
            mins, secs = divmod(int(eta_secs), 60)
            self.eta_label.setText(f"Time remaining: ~{mins:02d}:{secs:02d}")
        else:
            self.eta_label.setText("Finishing up...")

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PyAudacity - Audio Editor")
        self._apply_initial_window_geometry()
        
        # Core Components
        self.audio_engine = AudioEngine()
        self.audio_engine.positionChanged.connect(self.on_position_changed)
        self.audio_engine.stateChanged.connect(self.on_state_changed)
        self.audio_engine.tracksChanged.connect(self.refresh_track_list)
        
        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(12, 10, 12, 10)
        self.main_layout.setSpacing(6)

        self.create_toolbar()
        self.create_menus()
        self.create_track_view()
        self.create_transport_controls()
        
        # Global View State
        self.global_visible_start = 0
        self.global_visible_len = 0
        self.track_widgets = []

        # Async task context (used to apply results safely on UI thread)
        self._async_context: dict[str, Any] = {}
        self._is_loading = False

        
        # Update timer (GUI update 30fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.periodic_update)
        self.timer.start(30) # 30 ms interval

    def _apply_initial_window_geometry(self) -> None:
        """Set a 16:9 startup size that fits the current screen."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 720)
            return

        available = screen.availableGeometry()
        max_w = int(available.width() * 0.92)
        max_h = int(available.height() * 0.92)
        target_w = 1280
        target_h = 720

        if max_w >= target_w and max_h >= target_h:
            width = target_w
            height = target_h
        else:
            width = max_w
            height = int(width * 9 / 16)
            if height > max_h:
                height = max_h
                width = int(height * 16 / 9)

        self.resize(width, height)
        self.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )

    def create_about_menu(self) -> None:
        menubar = self.menuBar()
        about_menu = menubar.addMenu("&About")

        quick_topics = [
            ("Getting Started", "getting_started"),
            ("Editing Basics", "editing"),
            ("Effects and Presets", "effects"),
            ("AI and Separation", "ai"),
            ("Shortcuts", "shortcuts"),
        ]

        for label, key in quick_topics:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, k=key: self._show_about_dialog(k))
            about_menu.addAction(action)

        about_menu.addSeparator()

        all_action = QAction("All Topics...", self)
        all_action.triggered.connect(lambda: self._show_about_dialog(None))
        about_menu.addAction(all_action)

    def _show_about_dialog(self, initial_topic: Optional[str]) -> None:
        topics = {
            "getting_started": {
                "title": "Getting Started",
                "html": (
                    "<h2>Getting Started</h2>"
                    "<ul>"
                    "<li><b>File &gt; Open Project/File</b>: replace the current project.</li>"
                    "<li><b>File &gt; Import Audio</b>: add a new track to the project.</li>"
                    "<li><b>File &gt; Export As</b>: mix down and export to WAV/MP3/FLAC/OGG.</li>"
                    "</ul>"
                ),
            },
            "workspace": {
                "title": "Workspace",
                "html": (
                    "<h2>Workspace</h2>"
                    "<ul>"
                    "<li><b>Tracks panel</b>: mute/solo, volume, and close per track.</li>"
                    "<li><b>Waveform view</b>: scroll to zoom, drag to select.</li>"
                    "<li><b>Time ruler</b>: click or drag to seek.</li>"
                    "<li><b>Transport</b>: play/pause, stop, and time display.</li>"
                    "</ul>"
                ),
            },
            "editing": {
                "title": "Editing Basics",
                "html": (
                    "<h2>Editing Basics</h2>"
                    "<ul>"
                    "<li>Cut, copy, paste, and delete operate on the selection.</li>"
                    "<li>Split trims at the playhead to create segment markers.</li>"
                    "<li>Undo/redo is available across edits and effects.</li>"
                    "</ul>"
                ),
            },
            "effects": {
                "title": "Effects and Presets",
                "html": (
                    "<h2>Effects and Presets</h2>"
                    "<ul>"
                    "<li>Effects menu includes gain, fades, delay, reverb, and filters.</li>"
                    "<li>Vocal processing adds pitch shift, compressor, de-esser, and chorus.</li>"
                    "<li>Presets chain multiple effects for common styles.</li>"
                    "</ul>"
                ),
            },
            "ai": {
                "title": "AI and Separation",
                "html": (
                    "<h2>AI and Separation</h2>"
                    "<ul>"
                    "<li>Tools menu provides DSP options plus Demucs AI separation.</li>"
                    "<li>Demucs runs on CPU and may take several minutes on large files.</li>"
                    "</ul>"
                ),
            },
            "shortcuts": {
                "title": "Shortcuts",
                "html": (
                    "<h2>Shortcuts</h2>"
                    "<ul>"
                    "<li><code>Ctrl+O</code> Open, <code>Ctrl+S</code> Export</li>"
                    "<li><code>Ctrl+Z</code> Undo, <code>Ctrl+Y</code> Redo</li>"
                    "<li><code>Ctrl+C</code> Copy, <code>Ctrl+X</code> Cut</li>"
                    "<li><code>Ctrl+V</code> Paste, <code>Del</code> Delete</li>"
                    "<li><code>Ctrl+A</code> Select all</li>"
                    "</ul>"
                ),
            },
            "tips": {
                "title": "Tips",
                "html": (
                    "<h2>Tips</h2>"
                    "<ul>"
                    "<li>Use the spectrogram toggle for a frequency view.</li>"
                    "<li>Large files display downsampled waveforms for smooth scrolling.</li>"
                    "</ul>"
                ),
            },
        }

        dialog = QDialog(self)
        dialog.setWindowTitle("About PyAudioEditor")
        dialog.setMinimumSize(720, 420)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        topic_list = QListWidget()
        topic_list.setMinimumWidth(200)

        content = QTextBrowser()
        content.setOpenExternalLinks(False)
        content.setStyleSheet(
            "QTextBrowser { background-color: #11151c; border: 1px solid #252c3a; "
            "border-radius: 6px; padding: 12px; }"
        )

        splitter.addWidget(topic_list)
        splitter.addWidget(content)
        splitter.setStretchFactor(1, 1)

        for key, data in topics.items():
            topic_list.addItem(data["title"])

        keys = list(topics.keys())

        def set_topic(index: int) -> None:
            if index < 0 or index >= len(keys):
                return
            content.setHtml(topics[keys[index]]["html"])

        def on_topic_changed() -> None:
            row = topic_list.currentRow()
            set_topic(row)

        topic_list.currentRowChanged.connect(lambda _: on_topic_changed())

        start_index = 0
        if initial_topic and initial_topic in topics:
            start_index = keys.index(initial_topic)
        topic_list.setCurrentRow(start_index)
        set_topic(start_index)

        dialog.exec()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ensure audio stream and worker threads stop before Qt teardown."""
        try:
            if hasattr(self, 'timer') and self.timer is not None:
                self.timer.stop()

            if hasattr(self, 'fake_progress_timer') and self.fake_progress_timer is not None:
                self.fake_progress_timer.stop()

            # Stop any running worker thread cleanly.
            if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
                self.worker.requestInterruption()
                self.worker.quit()
                self.worker.wait(2000)

            # Stop playback / detach callbacks before QObject destruction.
            if hasattr(self, 'audio_engine') and self.audio_engine is not None:
                self.audio_engine.cleanup()
        finally:
            super().closeEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_time_ruler_insets()

    def create_toolbar(self):
        # File & History Toolbar
        file_toolbar = self.addToolBar("File")
        file_toolbar.setMovable(False)
        file_toolbar.setIconSize(QSize(18, 18))
        
        undo_action = QAction(qta.icon("fa5s.undo", color="white"), "Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo)
        file_toolbar.addAction(undo_action)

        redo_action = QAction(qta.icon("fa5s.redo", color="white"), "Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.redo)
        file_toolbar.addAction(redo_action)

        # Edit Toolbar
        edit_toolbar = self.addToolBar("Edit")
        edit_toolbar.setMovable(False)
        edit_toolbar.setIconSize(QSize(18, 18))
        
        cut_action = QAction(qta.icon("fa5s.cut", color="white"), "Cut", self)
        cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        cut_action.triggered.connect(self.cut_selection)
        edit_toolbar.addAction(cut_action)

        copy_action = QAction(qta.icon("fa5s.copy", color="white"), "Copy", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.copy_selection)
        edit_toolbar.addAction(copy_action)

        paste_action = QAction(qta.icon("fa5s.paste", color="white"), "Paste", self)
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        paste_action.triggered.connect(self.paste_at_cursor)
        edit_toolbar.addAction(paste_action)

        delete_action = QAction(qta.icon("fa5s.trash-alt", color="white"), "Delete", self)
        delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        delete_action.triggered.connect(self.delete_selection)
        edit_toolbar.addAction(delete_action)

        edit_toolbar.addSeparator()

        split_action = QAction(qta.icon("fa5s.cut", color="#ffaa00"), "Split (Trim)", self)
        split_action.setShortcut("Ctrl+S")
        split_action.triggered.connect(self.split_at_playhead)
        edit_toolbar.addAction(split_action)

        select_all_action = QAction("Select All", self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self.select_all)
        self.addAction(select_all_action)

    def create_effects_menu(self):
        menubar = self.menuBar()
        effects_menu = menubar.addMenu("&Effects")
        
        # Standard Effects
        effects_list = [
            ("Amplify", "gain"),
            ("Fade In", "fade_in"),
            ("Fade Out", "fade_out"),
            ("Echo/Delay", "delay"),
            ("Reverb", "reverb"),
            ("Low-pass Filter", "lowpass"),
        ]
        
        for label, name in effects_list:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, n=name: self.show_effect_dialog(n))
            effects_menu.addAction(action)
            
        effects_menu.addSeparator()
        
        # Advanced Vocal Effects Submenu
        vocal_menu = effects_menu.addMenu("Vocal Processing")
        
        vocal_effects = [
            ("Pitch Shift (Time-Preserving)", "pitch_shift"),
            ("Compressor", "compressor"),
            ("De-Esser", "deesser"),
            ("Chorus", "chorus"),
        ]
        
        for label, name in vocal_effects:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, n=name: self.show_effect_dialog(n))
            vocal_menu.addAction(action)
        
        effects_menu.addSeparator()
        
        # Presets Submenu
        presets_menu = effects_menu.addMenu("Presets (Popular Styles)")
        
        presets_list = [
            ("Slowed + Reverb", "slowed_reverb"),
            ("Nightcore", "nightcore"),
            ("Lo-Fi Style", "lofi"),
            ("Bass Boosted", "bass_boosted"),
            ("Podcast Clean", "podcast_clean"),
            ("Radio Voice", "radio_voice"),
            ("Dreamy Space", "dreamy_space"),
        ]
        
        for label, name in presets_list:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, n=name: self.apply_preset_dialog(n))
            presets_menu.addAction(action)

    def apply_preset_dialog(self, preset_name: str) -> None:
        """Apply a preset effect to a selected track."""
        if not self.audio_engine.tracks:
            self.statusBar().showMessage("No tracks to apply preset", 3000)
            return

        idx = self.select_track_dialog("Apply Preset", f"Apply {preset_name} to:")
        if idx is None:
            return

        # Map preset names to functions
        preset_map = {
            "slowed_reverb": (fx_vocal.apply_slowed_reverb_preset, "Slowed + Reverb"),
            "nightcore": (fx_vocal.apply_nightcore_preset, "Nightcore"),
            "lofi": (fx_vocal.apply_lofi_preset, "Lo-Fi"),
            "bass_boosted": (fx_vocal.apply_bass_boosted_preset, "Bass Boosted"),
            "podcast_clean": (fx_vocal.apply_podcast_clean_preset, "Podcast Clean"),
            "radio_voice": (fx_vocal.apply_radio_voice_preset, "Radio Voice"),
            "dreamy_space": (fx_vocal.apply_dreamy_space_preset, "Dreamy Space"),
        }
        
        if preset_name not in preset_map:
            self.statusBar().showMessage(f"Unknown preset: {preset_name}", 3000)
            return
        
        effect_func, display_name = preset_map[preset_name]

        if self.audio_engine.apply_effect(idx, effect_func, display_name):
            self.statusBar().showMessage(f"Applied {display_name} to track {idx+1}", 3000)
        else:
            self.statusBar().showMessage(f"Failed to apply {display_name}", 3000)

    def select_track_dialog(self, title, label):
        """Unified track selection dialog."""
        if not self.audio_engine.tracks:
            return None
            
        track_names = [f"{i+1}: {t.name}" for i, t in enumerate(self.audio_engine.tracks)]
        
        # Default to selected track if possible (logic to be added to track widgets later)
        current_idx = 0
        
        item, ok = QInputDialog.getItem(self, title, label, track_names, current_idx, False)
        
        if ok and item:
            return int(item.split(":")[0]) - 1
        return None

    def run_async_task(self, task_func, title="Processing..."):
        """Runs a task in background with progress dialog."""
        # Create dialog
        self.progress_dialog = ProgressDialog(title, self)
        
        # Add a note for AI tasks
        if "Demucs" in title:
            ai_note = QLabel("AI processing is intensive and may take several minutes.")
            ai_note.setStyleSheet("color: #aaa; font-style: italic; font-size: 11px;")
            self.progress_dialog.layout().insertWidget(1, ai_note)
            
        self.progress_dialog.show()
        
        # Setup worker
        self.worker = AudioProcessingWorker(task_func)
        self.worker.progress.connect(self.progress_dialog.update_status)
        self.worker.finished.connect(self.on_async_task_finished)
        
        # Start
        self.worker.start()
        
        # Fake progress for indeterminate tasks
        self.fake_progress_timer = QTimer()
        self.fake_progress_timer.timeout.connect(self.update_fake_progress)
        self.fake_progress_val = 0
        self.fake_progress_timer.start(500)

    def update_fake_progress(self):
        if hasattr(self, 'progress_dialog') and self.progress_dialog.isVisible():
            # Slowly increment to 90%
            if self.fake_progress_val < 90:
                # Use a slower increment for AI tasks
                title = self.progress_dialog.windowTitle()
                is_ai = "Demucs" in title
                
                step = 0.5 if is_ai else 1.0
                self.fake_progress_val += (step + (90 - self.fake_progress_val) * 0.02)
                
                status = "AI Model Processing... (may take time)" if is_ai else "Processing..."
                self.progress_dialog.update_status(int(self.fake_progress_val), status, 0)

    def on_async_task_finished(self, success, result):
        # Stop fake timer
        if hasattr(self, 'fake_progress_timer'):
            self.fake_progress_timer.stop()
            
        # Close dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            
        if success:
            if isinstance(result, dict) and result.get("kind") == "load_file":
                clear_project = bool(self._async_context.get("clear_project"))
                self._is_loading = False
                self._async_context = {}
                if not result.get("success"):
                    msg = result.get("error") or "Failed to load audio file."
                    self.statusBar().showMessage(msg, 6000)
                    QMessageBox.warning(self, "Load Failed", msg)
                    return

                if clear_project:
                    self.audio_engine.clear_project()

                if not self.audio_engine.tracks:
                    self.audio_engine.samplerate = int(
                        result.get("sr", self.audio_engine.samplerate)
                    )

                new_track = AudioTrack(name=result.get("track_name", "Track"))
                new_track.set_data(result["data"], self.audio_engine.samplerate)
                self.audio_engine.add_track(new_track)

                label = "Opened" if clear_project else "Imported"
                file_path = result.get("file_path", "")
                self.statusBar().showMessage(f"{label}: {file_path}", 5000)
                return

            # If this task returned a SeparationResult, apply it on the UI thread here.
            if isinstance(result, SeparationResult):
                if not result.success:
                    msg = result.error or "Unknown separation error."
                    self.statusBar().showMessage(msg, 6000)
                    QMessageBox.warning(self, "Separation Failed", msg)
                    return

                base_name = self._async_context.get("base_track_name", "Track")
                sr = int(self._async_context.get("samplerate", self.audio_engine.samplerate))

                new_tracks = separation_core.create_tracks_from_separation(result, base_name, sr)
                if not new_tracks:
                    self.statusBar().showMessage("Separation returned no stems.", 6000)
                    QMessageBox.warning(self, "Separation Failed", "Separation returned no stems.")
                    return

                for t in new_tracks:
                    self.audio_engine.add_track(t)

                self.statusBar().showMessage(f"Added {len(new_tracks)} separated stems", 4000)
                self.refresh_track_list()
                return

            if result is True:
                self.statusBar().showMessage("Task completed successfully", 3000)
            elif isinstance(result, str):
                # Check for AI dependency errors
                lower_res = result.lower()
                if "pip install" in lower_res or "not installed" in lower_res or "not available" in lower_res:
                    if "demucs" in lower_res:
                        self.show_ai_install_help("demucs")
                    else:
                        QMessageBox.warning(self, "AI Tool Missing", result)
                else:
                    self.statusBar().showMessage(result, 5000)
                    if "failed" in lower_res or "error" in lower_res:
                        QMessageBox.warning(self, "Task Failed", result)
            elif result is False:
                QMessageBox.warning(self, "Task Failed", "The operation returned a failure status.")

            # Default refresh
            self.refresh_track_list()
        else:
            if self._async_context.get("kind") == "load_file":
                self._is_loading = False
                self._async_context = {}
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred during background processing:\n{result}",
            )

    def show_effect_dialog(self, effect_name):

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Apply {effect_name}")
        layout = QFormLayout(dialog)
        
        params = {}
        
        if effect_name == "gain":
            sb = QDoubleSpinBox()
            sb.setRange(0, 10)
            sb.setValue(2.0)
            layout.addRow("Factor:", sb)
            params["factor"] = sb
        elif effect_name == "delay":
            sb_ms = QDoubleSpinBox()
            sb_ms.setRange(1, 5000)
            sb_ms.setValue(300)
            layout.addRow("Delay (ms):", sb_ms)
            params["delay_ms"] = sb_ms
            
            sb_decay = QDoubleSpinBox()
            sb_decay.setRange(0, 1)
            sb_decay.setValue(0.4)
            layout.addRow("Decay:", sb_decay)
            params["decay"] = sb_decay
        elif effect_name == "reverb":
            sb = QDoubleSpinBox()
            sb.setRange(0, 1)
            sb.setValue(0.5)
            layout.addRow("Room Size:", sb)
            params["room_size"] = sb
        elif effect_name == "lowpass":
            sb = QDoubleSpinBox()
            sb.setRange(20, 20000)
            sb.setValue(1000)
            layout.addRow("Cutoff (Hz):", sb)
            params["cutoff"] = sb
        elif effect_name == "pitch_shift":
            sb = QDoubleSpinBox()
            sb.setRange(-12, 12)
            sb.setValue(0)
            sb.setSingleStep(0.5)
            layout.addRow("Semitones:", sb)
            params["semitones"] = sb
            
            # Add info label
            info = QLabel("Positive = higher pitch, Negative = lower pitch")
            info.setStyleSheet("color: #888; font-size: 10px;")
            layout.addRow(info)
        elif effect_name == "compressor":
            sb_thresh = QDoubleSpinBox()
            sb_thresh.setRange(-60, 0)
            sb_thresh.setValue(-20)
            layout.addRow("Threshold (dB):", sb_thresh)
            params["threshold_db"] = sb_thresh
            
            sb_ratio = QDoubleSpinBox()
            sb_ratio.setRange(1, 20)
            sb_ratio.setValue(4.0)
            layout.addRow("Ratio:", sb_ratio)
            params["ratio"] = sb_ratio
            
            sb_attack = QDoubleSpinBox()
            sb_attack.setRange(0.1, 100)
            sb_attack.setValue(5.0)
            layout.addRow("Attack (ms):", sb_attack)
            params["attack_ms"] = sb_attack
            
            sb_release = QDoubleSpinBox()
            sb_release.setRange(10, 1000)
            sb_release.setValue(100)
            layout.addRow("Release (ms):", sb_release)
            params["release_ms"] = sb_release
            
            sb_makeup = QDoubleSpinBox()
            sb_makeup.setRange(-12, 24)
            sb_makeup.setValue(0)
            layout.addRow("Makeup Gain (dB):", sb_makeup)
            params["makeup_db"] = sb_makeup
        elif effect_name == "deesser":
            sb_freq = QDoubleSpinBox()
            sb_freq.setRange(2000, 12000)
            sb_freq.setValue(6000)
            layout.addRow("Frequency (Hz):", sb_freq)
            params["frequency"] = sb_freq
            
            sb_thresh = QDoubleSpinBox()
            sb_thresh.setRange(-40, 0)
            sb_thresh.setValue(-20)
            layout.addRow("Threshold (dB):", sb_thresh)
            params["threshold_db"] = sb_thresh
            
            sb_red = QDoubleSpinBox()
            sb_red.setRange(0, 20)
            sb_red.setValue(6)
            layout.addRow("Reduction (dB):", sb_red)
            params["reduction_db"] = sb_red
        elif effect_name == "chorus":
            sb_rate = QDoubleSpinBox()
            sb_rate.setRange(0.1, 5)
            sb_rate.setValue(0.5)
            layout.addRow("Rate (Hz):", sb_rate)
            params["rate"] = sb_rate
            
            sb_depth = QDoubleSpinBox()
            sb_depth.setRange(0.5, 20)
            sb_depth.setValue(3.0)
            layout.addRow("Depth (ms):", sb_depth)
            params["depth_ms"] = sb_depth
            
            sb_mix = QDoubleSpinBox()
            sb_mix.setRange(0, 1)
            sb_mix.setValue(0.3)
            layout.addRow("Mix:", sb_mix)
            params["mix"] = sb_mix
            
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            final_params = {k: v.value() for k, v in params.items()}
            self.apply_effect_to_selection(effect_name, final_params)

    def apply_effect_to_selection(self, effect_name: str, params: dict) -> None:
        """Apply an effect to the current selection or entire track."""
        target_track_idx = -1
        selection = None
        
        for i, tw in enumerate(self.track_widgets):
            s_range = tw.waveform.get_selection_range_samples()
            if s_range:
                target_track_idx = i
                selection = s_range
                break
                
        if target_track_idx == -1 and self.audio_engine.tracks:
            target_track_idx = 0
            track = self.audio_engine.tracks[0]
            selection = (0, len(track.data)) if track.data is not None else None

        if target_track_idx == -1 or selection is None:
            self.statusBar().showMessage("No track or selection for effect", 3000)
            return

        start, end = selection
        
        # Map effect names to functions
        effect_map = {
            "gain": (fx.apply_gain, "Amplify"),
            "fade_in": (lambda d, sr: fx.apply_fade_in(d), "Fade In"),
            "fade_out": (lambda d, sr: fx.apply_fade_out(d), "Fade Out"),
            "delay": (fx.apply_delay, "Delay"),
            "reverb": (fx.apply_reverb, "Reverb"),
            "lowpass": (fx.apply_lowpass, "Low-pass"),
            "highpass": (fx.apply_highpass, "High-pass"),
            "normalize": (lambda d, sr: fx.apply_normalize(d), "Normalize"),
            "reverse": (lambda d, sr: fx.apply_reverse(d), "Reverse"),
            "pitch_shift": (fx_vocal.apply_pitch_shift, "Pitch Shift"),
            "compressor": (fx_vocal.apply_compressor, "Compressor"),
            "deesser": (fx_vocal.apply_deesser, "De-esser"),
            "chorus": (fx_vocal.apply_chorus, "Chorus"),
        }
        
        if effect_name not in effect_map:
            self.statusBar().showMessage(f"Unknown effect: {effect_name}", 3000)
            return
        
        effect_func, display_name = effect_map[effect_name]
        
        success = self.audio_engine.apply_effect(
            target_track_idx,
            effect_func,
            display_name,
            start,
            end,
            **params
        )
        
        if success:
            self.statusBar().showMessage(f"Applied {display_name}", 3000)
        else:
            self.statusBar().showMessage(f"Failed to apply {display_name}", 3000)

    def split_at_playhead(self):
        pos = self.audio_engine.current_frame
        if self.audio_engine.split_track(pos):
            self.refresh_track_list()
            self.statusBar().showMessage(f"Split at {pos}", 3000)

    def select_all(self):
        if not self.track_widgets: return
        
        max_samples = self.audio_engine.get_max_duration_samples()
        width = self.track_widgets[0].waveform.width()
        
        for tw in self.track_widgets:
            tw.waveform.selection_start = 0
            tw.waveform.selection_end = width
            tw.waveform.update()
        
        self.statusBar().showMessage("Selected all", 2000)

    def create_menus(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction(qta.icon("fa5s.folder-open", color="white"), "&Open Project/File", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        
        import_action = QAction(qta.icon("fa5s.file-import", color="white"), "&Import Audio...", self)
        import_action.triggered.connect(self.import_file_dialog)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        export_action = QAction(qta.icon("fa5s.file-export", color="white"), "&Export AS...", self)
        export_action.setShortcut(QKeySequence.StandardKey.Save)
        export_action.triggered.connect(self.export_file_dialog)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.create_effects_menu()
        self.create_tools_menu()
        self.create_about_menu()

    def create_tools_menu(self):
        menubar = self.menuBar()
        tools_menu = menubar.addMenu("&Tools")
        
        # Spectrogram Toggle
        spec_action = QAction(qta.icon("fa5s.wave-square", color="aqua"), "Toggle Spectrogram (All)", self)
        spec_action.setCheckable(True)
        spec_action.triggered.connect(self.toggle_all_spectrograms)
        tools_menu.addAction(spec_action)
        
        tools_menu.addSeparator()
        
        # Vocal Separation Submenu
        ai_menu = tools_menu.addMenu("🎵 Vocal Separation")
        
        # Quick DSP methods
        hpss_action = QAction("Separate Harmonic/Percussive (Fast)", self)
        hpss_action.setToolTip("Uses HPSS algorithm - fast but basic separation")
        hpss_action.triggered.connect(lambda: self.apply_tool_to_track("hpss"))
        ai_menu.addAction(hpss_action)
        
        karaoke_action = QAction("Separate Vocals (Karaoke DSP)", self)
        karaoke_action.setToolTip("Center-channel cancellation - works on stereo tracks")
        karaoke_action.triggered.connect(lambda: self.apply_tool_to_track("karaoke"))
        ai_menu.addAction(karaoke_action)
        
        ai_menu.addSeparator()
        ai_menu.addAction(QAction("─── Demucs (AI, CPU) ───", self))
        
        # Demucs (AI, CPU)
        demucs_action = QAction("🤖 Separate Vocals (Demucs AI)", self)
        demucs_action.setToolTip("State-of-the-art AI separation (CPU only) - requires torch & demucs")
        demucs_action.triggered.connect(lambda: self.apply_tool_to_track("demucs"))
        ai_menu.addAction(demucs_action)
        
        demucs_4stem_action = QAction("🤖 Separate 4 Stems (Demucs AI)", self)
        demucs_4stem_action.setToolTip("Separates into Vocals, Drums, Bass, Other (CPU only)")
        demucs_4stem_action.triggered.connect(lambda: self.apply_tool_to_track("demucs_4stem"))
        ai_menu.addAction(demucs_4stem_action)


    def toggle_all_spectrograms(self, checked):
        for tw in self.track_widgets:
            tw.waveform.toggle_spectrogram(checked)
        self.statusBar().showMessage(f"Spectrogram {'enabled' if checked else 'disabled'}", 2000)

    def apply_tool_to_track(self, tool_name):
        if not self.audio_engine.tracks:
            self.statusBar().showMessage("No tracks available", 3000)
            return
            
        idx = self.select_track_dialog("Select Track", f"Apply {tool_name} to:")
        if idx is None:
            return

        # Define the task based on tool name
        task_func = None
        task_name = "Processing"

        if tool_name == "hpss":
            task_func = lambda: self.audio_engine.separate_hpss(idx)
            task_name = "HPSS Separation"
        elif tool_name == "karaoke":
            task_func = lambda: self.audio_engine.separate_vocals_dsp(idx)
            task_name = "Karaoke DSP"
        elif tool_name == "demucs":
            # IMPORTANT: compute in worker, but apply (add tracks) on UI thread.
            base = self.audio_engine.tracks[idx]
            self._async_context = {"kind": "separation", "base_track_name": base.name, "samplerate": self.audio_engine.samplerate}
            task_func = lambda: separation_core.separate_with_demucs(base.data, self.audio_engine.samplerate, True)
            task_name = "Demucs AI Separation"
        elif tool_name == "demucs_4stem":
            base = self.audio_engine.tracks[idx]
            self._async_context = {"kind": "separation", "base_track_name": base.name, "samplerate": self.audio_engine.samplerate}
            task_func = lambda: separation_core.separate_with_demucs(base.data, self.audio_engine.samplerate, False)
            task_name = "Demucs 4-Stem Separation"
            
        if task_func:
            self.run_async_task(task_func, f"Running {task_name}...")

    def show_ai_install_help(self, tool_name):
        """Show help dialog for installing AI dependencies."""
        if tool_name in ["demucs", "demucs_4stem"]:
            QMessageBox.warning(self, "Demucs Not Available",
                "Demucs AI separation requires additional packages.\n\n"
                "To install, run in your terminal:\n"
                "pip install torch torchaudio demucs\n\n"
                "Note: This requires ~2GB download.")
        else:
            QMessageBox.warning(self, "Tool Failed",
                f"The tool '{tool_name}' failed to execute.\n"
                "Check the console for error details.")

    def undo(self):
        if self.audio_engine.undo():
            self.refresh_track_list()
            self.statusBar().showMessage("Undo successful", 3000)

    def redo(self):
        if self.audio_engine.redo():
            self.refresh_track_list()
            self.statusBar().showMessage("Redo successful", 3000)

    def create_transport_controls(self):
        transport_widget = QWidget()
        transport_widget.setStyleSheet(
            "background-color: #151922; border-top: 1px solid #252c3a;"
        )
        transport_layout = QHBoxLayout(transport_widget)
        transport_layout.setContentsMargins(18, 10, 18, 10)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet(
            "font-family: 'Cascadia Mono', 'Consolas'; "
            "font-size: 18px; font-weight: 600; color: #9fe8e8; min-width: 180px;"
        )
        
        # Style helper for transport buttons
        btn_style = """
            QPushButton {
                background-color: #1f2533; 
                border: 1px solid #2c3444;
                border-radius: 18px; 
                padding: 6px;
            }
            QPushButton:hover { background-color: #283145; }
            QPushButton:pressed { background-color: #1a202d; }
        """
        
        self.btn_stop = QPushButton()
        self.btn_stop.setIcon(qta.icon("fa5s.stop", color="#ff5555"))
        self.btn_stop.setIconSize(QSize(24, 24))
        self.btn_stop.setStyleSheet(btn_style)
        self.btn_stop.clicked.connect(self.audio_engine.stop)
        
        self.btn_play_pause = QPushButton()
        self.btn_play_pause.setIcon(qta.icon("fa5s.play", color="#55ff55"))
        self.btn_play_pause.setIconSize(QSize(32, 32))
        self.btn_play_pause.setStyleSheet(btn_style)
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        
        transport_layout.addWidget(self.time_label)
        transport_layout.addStretch()
        transport_layout.addWidget(self.btn_stop)
        transport_layout.addWidget(self.btn_play_pause)
        transport_layout.addStretch()
        
        # Add to main layout at the bottom
        self.main_layout.addWidget(transport_widget)
        self.statusBar().showMessage("Ready")

    def toggle_play_pause(self):
        if self.audio_engine.is_playing:
            self.audio_engine.pause()
        else:
            self.audio_engine.play()
        
    def create_track_view(self):
        # Time Ruler
        self.time_ruler = TimeRulerWidget()
        self.time_ruler.seekRequested.connect(self.audio_engine.seek)
        self.time_ruler.set_left_offset(TrackWidget.CONTROLS_WIDTH)
        self.main_layout.addWidget(self.time_ruler)

        # Scroll Area for Tracks
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.verticalScrollBar().rangeChanged.connect(
            self._update_time_ruler_insets
        )
        
        self.tracks_container = QWidget()
        self.tracks_layout = QVBoxLayout()
        self.tracks_layout.setSpacing(6)
        self.tracks_layout.setContentsMargins(0, 6, 0, 6)
        self.tracks_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.tracks_container.setLayout(self.tracks_layout)
        
        self.scroll_area.setWidget(self.tracks_container)
        
        # Timeline Scrollbar (Global)
        self.timeline_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.timeline_scrollbar.valueChanged.connect(self.on_global_scroll)
        
        self.main_layout.addWidget(self.scroll_area, stretch=1)
        self.main_layout.addWidget(self.timeline_scrollbar)
        
        self.track_widgets = [] # Keep references
        self._update_time_ruler_insets()

    def refresh_track_list(self):
        from src.utils.logger import logger
        logger.info("Refreshing track list UI...")
        
        # Clear existing
        for w in self.track_widgets:
            self.tracks_layout.removeWidget(w)
            w.deleteLater()
        self.track_widgets = []
        
        # Add new
        logger.info(f"Adding {len(self.audio_engine.tracks)} tracks to UI")
        max_samples = self.audio_engine.get_max_duration_samples()
        
        # Update global view state if project length changed or uninitialized
        if max_samples > 0:
            # If uninitialized OR if we have a single track that grew (common for presets like Slowed)
            if self.global_visible_len <= 0 or (len(self.audio_engine.tracks) == 1 and self.global_visible_len < max_samples):
                self.global_visible_start = 0
                self.global_visible_len = max_samples
            elif self.global_visible_len > max_samples:
                # Clamp when effects shorten audio (e.g., nightcore)
                self.global_visible_len = max_samples
                self.global_visible_start = min(self.global_visible_start, max(0, max_samples - self.global_visible_len))

        for track in self.audio_engine.tracks:
            tw = TrackWidget(track, self.audio_engine)
            
            # Sync view with global state
            tw.waveform.set_view(self.global_visible_start, self.global_visible_len)
            
            tw.waveform.viewChanged.connect(self.on_track_view_changed)
            tw.waveform.seekRequested.connect(self.audio_engine.seek)
            
            self.tracks_layout.addWidget(tw)
            self.track_widgets.append(tw)
        
        # Update Time Ruler if tracks exist
        if self.track_widgets:
            self.time_ruler.set_view(self.global_visible_start, self.global_visible_len, max_samples, self.audio_engine.samplerate)
            self.update_scrollbar_range()
        else:
            # Reset view if no tracks
            self.global_visible_start = 0
            self.global_visible_len = 0
            self.time_ruler.set_view(0, 0, 0, 44100)

        # Force layout update
        self.tracks_container.update()
        self._update_time_ruler_insets()
        logger.info("Track list refresh complete")

    def _update_time_ruler_insets(self, *args) -> None:
        """Align time ruler with scroll area insets."""
        if not hasattr(self, "scroll_area"):
            return

        scrollbar = self.scroll_area.verticalScrollBar()
        right_inset = 0
        if scrollbar.isVisible():
            right_inset = scrollbar.width() or scrollbar.sizeHint().width()
        self.time_ruler.set_right_inset(right_inset)

    def periodic_update(self):
        """Updates UI elements (playhead, time label) at 30fps."""
        if not self.audio_engine.is_playing:
            return

        current_sample = self.audio_engine.current_frame
        
        # Update playheads
        self.time_ruler.set_playhead(current_sample)
        for tw in self.track_widgets:
            tw.waveform.set_playhead(current_sample)
                
        # Update Time Label
        self._update_time_label(current_sample)

    def on_track_view_changed(self):
        """Synchronizes all tracks when one is zoomed or scrolled."""
        sender = self.sender()
        if not isinstance(sender, WaveformWidget): return
            
        max_samples = self.audio_engine.get_max_duration_samples()
        
        # Update global state from sender
        self.global_visible_len = max(100, min(sender.visible_len, max_samples))
        self.global_visible_start = max(
            0,
            min(sender.visible_start, max_samples - self.global_visible_len),
        )
        
        # Sync all tracks
        for tw in self.track_widgets:
            if tw.waveform == sender: continue
            tw.waveform.blockSignals(True)
            tw.waveform.set_view(self.global_visible_start, self.global_visible_len)
            tw.waveform.blockSignals(False)

        # Clamp sender too to avoid ruler/playhead drift near edges
        if (
            sender.visible_start != self.global_visible_start
            or sender.visible_len != self.global_visible_len
        ):
            sender.blockSignals(True)
            sender.set_view(self.global_visible_start, self.global_visible_len)
            sender.blockSignals(False)
        
        self.update_scrollbar_range()

    def update_scrollbar_range(self):
        """Updates the global scrollbar based on the current view."""
        if not self.track_widgets: return
        
        total = self.audio_engine.get_max_duration_samples()
        visible = self.global_visible_len
        start = self.global_visible_start
        
        self.time_ruler.set_view(start, visible, total, self.audio_engine.samplerate)
        
        self.timeline_scrollbar.blockSignals(True)
        self.timeline_scrollbar.setRange(0, max(0, total - visible))
        self.timeline_scrollbar.setPageStep(visible)
        self.timeline_scrollbar.setValue(start)
        self.timeline_scrollbar.blockSignals(False)

    def on_global_scroll(self, value):
        """Handles global scrollbar movement."""
        if not self.track_widgets: return
        
        max_samples = self.audio_engine.get_max_duration_samples()
        self.global_visible_start = max(0, min(value, max_samples - self.global_visible_len))
        
        self.time_ruler.set_view(self.global_visible_start, self.global_visible_len, max_samples, self.audio_engine.samplerate)
        
        for tw in self.track_widgets:
            tw.waveform.blockSignals(True)
            tw.waveform.set_view(self.global_visible_start, self.global_visible_len)
            tw.waveform.blockSignals(False)

    # Edit operations now delegate to engine (Global)
    # Edit operations
    def delete_selection(self):
        selection = None
        for tw in self.track_widgets:
            s_range = tw.waveform.get_selection_range_samples()
            if s_range:
                selection = s_range
                break

        if selection:
            start, end = selection
            if self.audio_engine.delete_range(start, end):
                self.statusBar().showMessage("Deleted selection", 3000)
        else:
            self.statusBar().showMessage("No selection to delete", 3000)

    def undo(self):
        if self.audio_engine.undo():
            self.refresh_track_list()
            self.statusBar().showMessage("Undo successful", 2000)

    def redo(self):
        if self.audio_engine.redo():
            self.refresh_track_list()
            self.statusBar().showMessage("Redo successful", 2000)

    def cut_selection(self):
        selection = None
        for tw in self.track_widgets:
            s_range = tw.waveform.get_selection_range_samples()
            if s_range:
                selection = s_range
                break

        if selection:
            start, end = selection
            if self.audio_engine.cut_range(start, end):
                self.statusBar().showMessage("Cut selection", 3000)
        else:
            self.statusBar().showMessage("No selection to cut", 3000)

    def copy_selection(self):
        selection = None
        for tw in self.track_widgets:
            s_range = tw.waveform.get_selection_range_samples()
            if s_range:
                selection = s_range
                break

        if selection:
            start, end = selection
            self.audio_engine.copy_range(start, end)
            self.statusBar().showMessage("Copied to clipboard", 3000)
        else:
            self.statusBar().showMessage("No selection to copy", 3000)

    def paste_at_cursor(self):
        # Paste at selection start or current playhead
        paste_pos = self.audio_engine.current_frame
        
        # Check if there is a selection to paste AT
        for tw in self.track_widgets:
            s_range = tw.waveform.get_selection_range_samples()
            if s_range:
                paste_pos = s_range[0]
                break

        if self.audio_engine.paste_at(paste_pos):
            self.statusBar().showMessage("Pasted from clipboard", 3000)
        else:
            self.statusBar().showMessage("Paste failed (Empty clipboard?)", 3000)

    def open_file_dialog(self):
        if self._is_loading:
            self.statusBar().showMessage("Already loading audio...", 2000)
            return

        if self.audio_engine.tracks:
            reply = QMessageBox.question(
                self,
                "Open File",
                "Opening a new file will clear the current project. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self._select_and_load_audio(clear_project=True, dialog_title="Open Audio File")

    def import_file_dialog(self):
        if self._is_loading:
            self.statusBar().showMessage("Already loading audio...", 2000)
            return

        self._select_and_load_audio(clear_project=False, dialog_title="Import Audio File")

    def _select_and_load_audio(self, clear_project: bool, dialog_title: str) -> None:
        from src.utils.logger import logger
        logger.info("Opening audio file dialog: %s", dialog_title)

        if self.audio_engine.is_playing:
            self.audio_engine.pause()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            dialog_title,
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg)",
        )
        if not file_path:
            return

        logger.info("User selected: %s", file_path)
        self._start_audio_load(file_path, clear_project)

    def _start_audio_load(self, file_path: str, clear_project: bool) -> None:
        if self._is_loading:
            return

        if self.audio_engine.is_playing:
            self.audio_engine.pause()

        target_sr = None
        if not clear_project and self.audio_engine.tracks:
            target_sr = self.audio_engine.samplerate

        self._async_context = {
            "kind": "load_file",
            "clear_project": clear_project,
        }
        self._is_loading = True
        task_name = f"Loading {os.path.basename(file_path)}..."
        task_func = lambda: self._load_audio_file_worker(file_path, target_sr)
        self.run_async_task(task_func, task_name)

    def _load_audio_file_worker(
        self,
        file_path: str,
        target_sr: Optional[int],
    ) -> dict[str, Any]:
        try:
            import librosa
            import numpy as np

            data, loaded_sr = librosa.load(file_path, sr=target_sr, mono=False)
            if data.ndim > 1:
                data = data.T

            data = data.astype(np.float32)

            return {
                "kind": "load_file",
                "success": True,
                "file_path": file_path,
                "track_name": os.path.basename(file_path),
                "data": data,
                "sr": int(loaded_sr),
            }
        except Exception as exc:
            return {
                "kind": "load_file",
                "success": False,
                "file_path": file_path,
                "error": str(exc),
            }

    def export_file_dialog(self):
        if not self.audio_engine.tracks:
            self.statusBar().showMessage("Nothing to export", 3000)
            return
            
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Audio", "output.wav", 
            "WAV (*.wav);;MP3 (*.mp3);;FLAC (*.flac);;OGG (*.ogg)"
        )
        
        if file_path:
            import soundfile as sf
            try:
                master_data = self.audio_engine.mix_down()
                if master_data is not None:
                    sf.write(file_path, master_data, self.audio_engine.samplerate)
                    self.statusBar().showMessage(f"Exported to: {file_path}", 5000)
                    
                    # Feedback visual com caixa de mensagem
                    QMessageBox.information(
                        self, 
                        "Export Successful", 
                        f"Your audio has been exported successfully!\n\nLocation: {file_path}"
                    )
                else:
                    self.statusBar().showMessage("Export failed: No audio data", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not export file: {e}")

    def _update_time_label(self, current_sample: int) -> None:
        samplerate = self.audio_engine.samplerate
        cur_sec = current_sample / samplerate if samplerate else 0
        total_sec = self.audio_engine.get_duration()

        fmt = lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}"
        self.time_label.setText(f"{fmt(cur_sec)} / {fmt(total_sec)}")

    def on_position_changed(self, seconds):
        current_sample = self.audio_engine.current_frame
        self.time_ruler.set_playhead(current_sample)
        for tw in self.track_widgets:
            tw.waveform.set_playhead(current_sample)
        self._update_time_label(current_sample)

    def on_state_changed(self, state):
        self.statusBar().showMessage(f"State: {state}", 2000)
        if state == 'playing':
            self.btn_play_pause.setIcon(qta.icon("fa5s.pause", color="#ffff55"))
        else:
            self.btn_play_pause.setIcon(qta.icon("fa5s.play", color="#55ff55"))
