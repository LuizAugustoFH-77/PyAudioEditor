from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QScrollBar, QScrollArea,
                             QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox,
                             QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QAction, QIcon, QKeySequence
import os
import qdarktheme
import qtawesome as qta

from src.core.audio_engine import AudioEngine
from src.ui.waveform_view import WaveformWidget
from src.ui.track_widget import TrackWidget
from src.ui.time_ruler import TimeRulerWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PyAudacity - Audio Editor")
        self.resize(1000, 700)
        
        # Core Components
        self.audio_engine = AudioEngine()
        self.audio_engine.positionChanged.connect(self.on_position_changed)
        self.audio_engine.stateChanged.connect(self.on_state_changed)
        self.audio_engine.tracksChanged.connect(self.refresh_track_list)
        
        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.create_toolbar()
        self.create_menus()
        self.create_track_view()
        self.create_transport_controls()
        
        # Global View State
        self.global_visible_start = 0
        self.global_visible_len = 0
        self.track_widgets = []
        
        # Update timer (GUI update 30fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.periodic_update)
        self.timer.start(30) # 30 ms interval

    def create_toolbar(self):
        # File & History Toolbar
        file_toolbar = self.addToolBar("File")
        file_toolbar.setMovable(False)
        
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
        
        # Presets Submenu
        presets_menu = effects_menu.addMenu("Presets (Popular Styles)")
        
        presets_list = [
            ("Slowed + Reverb", "slowed_reverb"),
            ("Nightcore", "nightcore"),
            ("Lo-Fi Style", "lofi"),
            ("Bass Boosted", "bass_boosted"),
        ]
        
        for label, name in presets_list:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, n=name: self.apply_preset_dialog(n))
            presets_menu.addAction(action)

    def apply_preset_dialog(self, preset_name):
        if not self.audio_engine.tracks:
            self.statusBar().showMessage("No tracks to apply preset", 3000)
            return
            
        # If only one track, apply directly
        if len(self.audio_engine.tracks) == 1:
            self.audio_engine.apply_preset(preset_name, 0)
            self.statusBar().showMessage(f"Applied {preset_name} to track 1", 3000)
            return
            
        # Multiple tracks: Show selection dialog
        track_names = [f"{i+1}: {t.name}" for i, t in enumerate(self.audio_engine.tracks)]
        item, ok = QInputDialog.getItem(self, "Select Track", "Apply preset to:", track_names, 0, False)
        
        if ok and item:
            idx = int(item.split(":")[0]) - 1
            self.audio_engine.apply_preset(preset_name, idx)
            self.statusBar().showMessage(f"Applied {preset_name} to {item}", 3000)

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
            
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            final_params = {k: v.value() for k, v in params.items()}
            self.apply_effect_to_selection(effect_name, final_params)

    def apply_effect_to_selection(self, effect_name, params):
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

        if target_track_idx != -1 and selection:
            start, end = selection
            if self.audio_engine.apply_effect(effect_name, target_track_idx, start, end, **params):
                self.statusBar().showMessage(f"Applied {effect_name}", 3000)
            else:
                self.statusBar().showMessage(f"Failed to apply {effect_name}", 3000)
        else:
            self.statusBar().showMessage("No track or selection for effect", 3000)

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

    def create_tools_menu(self):
        menubar = self.menuBar()
        tools_menu = menubar.addMenu("&Tools")
        
        # Spectrogram Toggle
        spec_action = QAction(qta.icon("fa5s.wave-square", color="aqua"), "Toggle Spectrogram (All)", self)
        spec_action.setCheckable(True)
        spec_action.triggered.connect(self.toggle_all_spectrograms)
        tools_menu.addAction(spec_action)
        
        tools_menu.addSeparator()
        
        # AI Separation Submenu
        ai_menu = tools_menu.addMenu("AI & Separation")
        
        hpss_action = QAction("Separate Harmonic/Percussive (Fast)", self)
        hpss_action.triggered.connect(lambda: self.apply_tool_to_track("hpss"))
        ai_menu.addAction(hpss_action)
        
        karaoke_action = QAction("Separate Vocals (Karaoke DSP)", self)
        karaoke_action.triggered.connect(lambda: self.apply_tool_to_track("karaoke"))
        ai_menu.addAction(karaoke_action)
        
        demucs_action = QAction("Separate Vocals (AI - Demucs)", self)
        demucs_action.triggered.connect(lambda: self.apply_tool_to_track("demucs"))
        ai_menu.addAction(demucs_action)

    def toggle_all_spectrograms(self, checked):
        for tw in self.track_widgets:
            tw.waveform.toggle_spectrogram(checked)
        self.statusBar().showMessage(f"Spectrogram {'enabled' if checked else 'disabled'}", 2000)

    def apply_tool_to_track(self, tool_name):
        if not self.audio_engine.tracks:
            self.statusBar().showMessage("No tracks available", 3000)
            return
            
        # Select track
        track_names = [f"{i+1}: {t.name}" for i, t in enumerate(self.audio_engine.tracks)]
        item, ok = QInputDialog.getItem(self, "Select Track", f"Apply {tool_name} to:", track_names, 0, False)
        
        if ok and item:
            idx = int(item.split(":")[0]) - 1
            self.statusBar().showMessage(f"Running {tool_name} on {item}...", 5000)
            
            success = False
            if tool_name == "hpss":
                success = self.audio_engine.separate_hpss(idx)
            elif tool_name == "karaoke":
                success = self.audio_engine.separate_vocals_dsp(idx)
            elif tool_name == "demucs":
                # This might be slow, ideally background thread
                success = self.audio_engine.separate_ai_demucs(idx)
                
            if success:
                self.statusBar().showMessage(f"Tool {tool_name} completed", 3000)
            else:
                self.statusBar().showMessage(f"Tool {tool_name} failed", 3000)

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
        transport_widget.setStyleSheet("background-color: #222; border-top: 1px solid #444;")
        transport_layout = QHBoxLayout(transport_widget)
        transport_layout.setContentsMargins(20, 10, 20, 10)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: 'Consolas'; font-size: 20px; font-weight: bold; color: #00ffff; min-width: 180px;")
        
        # Style helper for transport buttons
        btn_style = """
            QPushButton {
                background-color: transparent; 
                border-radius: 20px; 
                padding: 5px;
            }
            QPushButton:hover { background-color: #444; }
            QPushButton:pressed { background-color: #555; }
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
        self.main_layout.addWidget(self.time_ruler)

        # Scroll Area for Tracks
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.tracks_container = QWidget()
        self.tracks_layout = QVBoxLayout()
        self.tracks_layout.setSpacing(0)
        self.tracks_layout.setContentsMargins(0,0,0,0)
        self.tracks_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.tracks_container.setLayout(self.tracks_layout)
        
        self.scroll_area.setWidget(self.tracks_container)
        
        # Timeline Scrollbar (Global)
        self.timeline_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.timeline_scrollbar.valueChanged.connect(self.on_global_scroll)
        
        self.main_layout.addWidget(self.scroll_area, stretch=1)
        self.main_layout.addWidget(self.timeline_scrollbar)
        
        self.track_widgets = [] # Keep references

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
        logger.info("Track list refresh complete")

    def periodic_update(self):
        """Updates UI elements (playhead, time label) at 30fps."""
        current_sample = self.audio_engine.current_frame
        samplerate = self.audio_engine.samplerate
        
        # Update playheads
        self.time_ruler.set_playhead(current_sample)
        for tw in self.track_widgets:
            tw.waveform.set_playhead(current_sample)
                
        # Update Time Label
        cur_sec = current_sample / samplerate if samplerate else 0
        total_sec = self.audio_engine.get_duration()
        
        fmt = lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}"
        self.time_label.setText(f"{fmt(cur_sec)} / {fmt(total_sec)}")

    def on_track_view_changed(self):
        """Synchronizes all tracks when one is zoomed or scrolled."""
        sender = self.sender()
        if not isinstance(sender, WaveformWidget): return
            
        max_samples = self.audio_engine.get_max_duration_samples()
        
        # Update global state from sender
        self.global_visible_len = max(100, min(sender.visible_len, max_samples))
        self.global_visible_start = max(0, min(sender.visible_start, max_samples - self.global_visible_len))
        
        # Sync all tracks
        for tw in self.track_widgets:
            if tw.waveform == sender: continue
            tw.waveform.blockSignals(True)
            tw.waveform.set_view(self.global_visible_start, self.global_visible_len)
            tw.waveform.blockSignals(False)
        
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
        if self.audio_engine.tracks:
            reply = QMessageBox.question(self, "Open File", 
                                       "Opening a new file will clear the current project. Continue?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        
        self.audio_engine.clear_project()
        self.import_file_dialog()

    def import_file_dialog(self):
        from src.utils.logger import logger
        logger.info("Opening import file dialog")
        
        # Pause if playing
        if self.audio_engine.is_playing:
            self.audio_engine.pause()
            
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)")
        if file_path:
            logger.info(f"User selected: {file_path}")
            success = self.audio_engine.load_file(file_path)
            if success:
                logger.debug(f"File {file_path} loaded successfully")
                self.statusBar().showMessage(f"Imported: {file_path}")
            else:
                logger.error(f"Failed to load file: {file_path}")
                self.statusBar().showMessage(f"Failed to load: {file_path}", 5000)

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

    def on_position_changed(self, seconds):
        pass

    def on_state_changed(self, state):
        self.statusBar().showMessage(f"State: {state}", 2000)
        if state == 'playing':
            self.btn_play_pause.setIcon(qta.icon("fa5s.pause", color="#ffff55"))
        else:
            self.btn_play_pause.setIcon(qta.icon("fa5s.play", color="#55ff55"))
