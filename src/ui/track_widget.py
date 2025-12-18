from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider
from PyQt6.QtCore import Qt, QSize
import qtawesome as qta
from src.ui.waveform_view import WaveformWidget
from src.utils.logger import logger

class TrackWidget(QWidget):
    def __init__(self, track, audio_engine, parent=None):
        super().__init__(parent)
        self.track = track
        self.audio_engine = audio_engine
        logger.debug(f"TrackWidget created for: {track.name}")
        logger.debug(f"Track data shape: {track.data.shape if track.data is not None else 'None'}")
        self.init_ui()
        
    def init_ui(self):
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setMinimumHeight(100)
        self.setMaximumHeight(120)
        
        # Track Control Panel (Left)
        self.controls_widget = QWidget()
        self.controls_widget.setFixedWidth(140)
        self.controls_widget.setStyleSheet("background-color: #2a2a2a; border-right: 1px solid #444;")
        self.controls_layout = QVBoxLayout()
        self.controls_layout.setContentsMargins(8, 8, 8, 8)
        self.controls_layout.setSpacing(5)
        self.controls_widget.setLayout(self.controls_layout)
        
        # Style for track buttons
        track_btn_style = """
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 2px;
                min-width: 28px;
                min-height: 24px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #222; }
        """
        
        # Name
        self.name_row = QHBoxLayout()
        self.name_label = QLabel(self.track.name)
        self.name_label.setStyleSheet("font-weight: bold; color: #ddd; font-size: 11px;")
        self.name_label.setWordWrap(True)
        
        self.btn_close = QPushButton()
        self.btn_close.setIcon(qta.icon("fa5s.times", color="#888"))
        self.btn_close.setIconSize(QSize(10, 10))
        self.btn_close.setFixedSize(16, 16)
        self.btn_close.setStyleSheet("QPushButton { border: none; background: transparent; } QPushButton:hover { color: white; }")
        self.btn_close.clicked.connect(lambda: self.audio_engine.remove_track(self.track_index()))
        
        self.name_row.addWidget(self.name_label)
        self.name_row.addStretch()
        self.name_row.addWidget(self.btn_close)
        
        # Mute/Solo
        self.btn_row = QHBoxLayout()
        self.btn_row.setSpacing(4)
        self.btn_mute = QPushButton()
        self.btn_mute.setIcon(qta.icon("fa5s.volume-mute", color="white"))
        self.btn_mute.setIconSize(QSize(14, 14))
        self.btn_mute.setCheckable(True)
        self.btn_mute.toggled.connect(self.toggle_mute)
        self.btn_mute.setStyleSheet(track_btn_style + "QPushButton:checked { background-color: #a82a2a; border-color: #ff5555; }")
        
        self.btn_solo = QPushButton()
        self.btn_solo.setIcon(qta.icon("fa5s.headphones", color="white"))
        self.btn_solo.setIconSize(QSize(14, 14))
        self.btn_solo.setCheckable(True)
        self.btn_solo.toggled.connect(self.toggle_solo)
        self.btn_solo.setStyleSheet(track_btn_style + "QPushButton:checked { background-color: #2a8a2a; border-color: #55ff55; }")
        
        self.btn_row.addWidget(self.btn_mute)
        self.btn_row.addWidget(self.btn_solo)
        
        # Volume
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 150) 
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedHeight(15)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 4px;
                background: #111;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #888;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover { background: #aaa; }
        """)
        self.volume_slider.valueChanged.connect(self.set_volume)
        
        self.controls_layout.addLayout(self.name_row)
        self.controls_layout.addLayout(self.btn_row)
        
        vol_layout = QHBoxLayout()
        vol_layout.setSpacing(5)
        vol_icon = QLabel()
        vol_icon.setPixmap(qta.icon("fa5s.volume-up", color="gray").pixmap(12, 12))
        vol_layout.addWidget(vol_icon)
        vol_layout.addWidget(self.volume_slider)
        
        self.controls_layout.addLayout(vol_layout)
        self.controls_layout.addStretch()
        
        # Waveform View (Right)
        self.waveform = WaveformWidget()
        
        # Pass data to waveform
        if self.track.data is not None:
            logger.info(f"Setting waveform data: {len(self.track.data)} samples")
            self.waveform.blockSignals(True)
            self.waveform.set_data(self.track.data, splits=self.track.splits)
            self.waveform.blockSignals(False)
        else:
            logger.warning("Track data is None when creating TrackWidget!")
        
        self.layout.addWidget(self.controls_widget)
        self.layout.addWidget(self.waveform, stretch=1)
        
        # Style
        self.setStyleSheet("background-color: #222; border-bottom: 1px solid #333;")

    def toggle_mute(self, checked):
        self.track.muted = checked
        
    def toggle_solo(self, checked):
        self.track.soloed = checked
        
    def set_volume(self, value):
        self.track.gain = value / 100.0

    def toggle_spectrogram(self, enabled):
        self.waveform.toggle_spectrogram(enabled)

    def track_index(self):
        try:
            return self.audio_engine.tracks.index(self.track)
        except:
            return -1
