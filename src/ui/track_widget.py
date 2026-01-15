"""
Track widget for PyAudioEditor.
Displays track controls and waveform visualization.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider
)
from PyQt6.QtCore import Qt, QSize
import qtawesome as qta

from .waveform_view import WaveformWidget

if TYPE_CHECKING:
    from src.core.track import AudioTrack
    from src.core.audio_engine import AudioEngine

logger = logging.getLogger("PyAudacity")


class TrackWidget(QWidget):
    """
    Widget representing a single audio track with controls and waveform display.
    """
    
    def __init__(
        self, 
        track: "AudioTrack", 
        audio_engine: "AudioEngine", 
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        
        self._track = track
        self._audio_engine = audio_engine
        
        logger.debug("TrackWidget created for: %s", track.name)
        
        self._init_ui()
    
    @property
    def track(self) -> "AudioTrack":
        return self._track
    
    @property
    def waveform(self) -> WaveformWidget:
        return self._waveform
    
    def _init_ui(self) -> None:
        """Initialize UI components."""
        self._layout = QHBoxLayout()
        self.setLayout(self._layout)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setMinimumHeight(100)
        self.setMaximumHeight(120)
        
        # Track Control Panel (Left)
        self._create_controls_panel()
        
        # Waveform View (Right)
        self._waveform = WaveformWidget()
        
        if self._track.data is not None:
            logger.info("Setting waveform data: %d samples", len(self._track.data))
            self._waveform.blockSignals(True)
            self._waveform.set_data(self._track.data, splits=self._track.splits)
            self._waveform.blockSignals(False)
        
        self._layout.addWidget(self._controls_widget)
        self._layout.addWidget(self._waveform, stretch=1)
        
        self.setStyleSheet(
            "background-color: #171c25; border: 1px solid #252c3a; border-radius: 6px;"
        )
    
    def _create_controls_panel(self) -> None:
        """Create the left control panel."""
        self._controls_widget = QWidget()
        self._controls_widget.setFixedWidth(140)
        self._controls_widget.setStyleSheet(
            "background-color: #1c212c; border-right: 1px solid #2c3444; "
            "border-top-left-radius: 6px; border-bottom-left-radius: 6px;"
        )
        
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(5)
        self._controls_widget.setLayout(controls_layout)
        
        # Button style
        btn_style = """
            QPushButton {
                background-color: #1f2533;
                border: 1px solid #2c3444;
                border-radius: 6px;
                padding: 3px;
                min-width: 28px;
                min-height: 24px;
            }
            QPushButton:hover { background-color: #283145; }
            QPushButton:pressed { background-color: #1a202d; }
        """
        
        # Name row
        name_row = QHBoxLayout()
        
        self._name_label = QLabel(self._track.name)
        self._name_label.setStyleSheet(
            "font-weight: 600; color: #e6e8eb; font-size: 11px;"
        )
        self._name_label.setWordWrap(True)
        
        self._btn_close = QPushButton()
        self._btn_close.setIcon(qta.icon("fa5s.times", color="#9aa3ad"))
        self._btn_close.setIconSize(QSize(10, 10))
        self._btn_close.setFixedSize(16, 16)
        self._btn_close.setStyleSheet(
            "QPushButton { border: none; background: transparent; } "
            "QPushButton:hover { color: white; }"
        )
        self._btn_close.clicked.connect(self._on_close_clicked)
        
        name_row.addWidget(self._name_label)
        name_row.addStretch()
        name_row.addWidget(self._btn_close)
        
        # Mute/Solo buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        
        self._btn_mute = QPushButton()
        self._btn_mute.setIcon(qta.icon("fa5s.volume-mute", color="white"))
        self._btn_mute.setIconSize(QSize(14, 14))
        self._btn_mute.setCheckable(True)
        self._btn_mute.toggled.connect(self._toggle_mute)
        self._btn_mute.setStyleSheet(
            btn_style + 
            "QPushButton:checked { background-color: #7a2a2f; border-color: #f04f5a; }"
        )
        
        self._btn_solo = QPushButton()
        self._btn_solo.setIcon(qta.icon("fa5s.headphones", color="white"))
        self._btn_solo.setIconSize(QSize(14, 14))
        self._btn_solo.setCheckable(True)
        self._btn_solo.toggled.connect(self._toggle_solo)
        self._btn_solo.setStyleSheet(
            btn_style + 
            "QPushButton:checked { background-color: #1f6b4d; border-color: #5ad37a; }"
        )
        
        btn_row.addWidget(self._btn_mute)
        btn_row.addWidget(self._btn_solo)
        
        # Volume slider
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 150)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedHeight(15)
        self._volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #2c3444;
                height: 4px;
                background: #10131a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #2cc7c9;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover { background: #4ad2d4; }
        """)
        self._volume_slider.valueChanged.connect(self._set_volume)
        
        # Volume row with icon
        vol_layout = QHBoxLayout()
        vol_layout.setSpacing(5)
        
        vol_icon = QLabel()
        vol_icon.setPixmap(qta.icon("fa5s.volume-up", color="#9aa3ad").pixmap(12, 12))
        vol_layout.addWidget(vol_icon)
        vol_layout.addWidget(self._volume_slider)
        
        # Add all to controls layout
        controls_layout.addLayout(name_row)
        controls_layout.addLayout(btn_row)
        controls_layout.addLayout(vol_layout)
        controls_layout.addStretch()
    
    def _on_close_clicked(self) -> None:
        """Handle close button click."""
        index = self._get_track_index()
        if index >= 0:
            self._audio_engine.remove_track(index)
    
    def _toggle_mute(self, checked: bool) -> None:
        """Toggle track mute state."""
        self._track.muted = checked
    
    def _toggle_solo(self, checked: bool) -> None:
        """Toggle track solo state."""
        self._track.soloed = checked
    
    def _set_volume(self, value: int) -> None:
        """Set track volume from slider."""
        self._track.gain = value / 100.0
    
    def toggle_spectrogram(self, enabled: bool) -> None:
        """Toggle spectrogram display."""
        self._waveform.toggle_spectrogram(enabled)
    
    def _get_track_index(self) -> int:
        """Get index of this track in the engine."""
        try:
            return self._audio_engine.tracks.index(self._track)
        except ValueError:
            return -1
    
    def track_index(self) -> int:
        """Get track index (legacy compatibility)."""
        return self._get_track_index()
