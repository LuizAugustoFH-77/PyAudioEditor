"""
Track widget for PyAudioEditor.
Displays track controls and waveform visualization.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QSize, QPropertyAnimation, QSequentialAnimationGroup, QPauseAnimation,
    QEasingCurve, pyqtProperty
)
from PyQt6.QtGui import QPainter, QPalette
import qtawesome as qta

from .waveform_view import WaveformWidget

if TYPE_CHECKING:
    from src.core.track import AudioTrack
    from src.core.audio_engine import AudioEngine

logger = logging.getLogger("PyAudacity")


class SlidingTextLabel(QWidget):
    """Widget that animates long text back and forth for readability."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._text = text
        self._offset = 0.0
        self._max_offset = 0.0
        self._padding = 2

        self._pause_start = QPauseAnimation(700, self)
        self._pause_end = QPauseAnimation(700, self)
        self._anim_forward = QPropertyAnimation(self, b"offset", self)
        self._anim_backward = QPropertyAnimation(self, b"offset", self)
        self._anim_forward.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._anim_backward.setEasingCurve(QEasingCurve.Type.InOutSine)

        self._anim_group = QSequentialAnimationGroup(self)
        self._anim_group.addAnimation(self._pause_start)
        self._anim_group.addAnimation(self._anim_forward)
        self._anim_group.addAnimation(self._pause_end)
        self._anim_group.addAnimation(self._anim_backward)
        self._anim_group.setLoopCount(-1)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(self.fontMetrics().height())
        self.setToolTip(text)

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        if text == self._text:
            return
        self._text = text
        self.setToolTip(text)
        self._refresh_animation()
        self.update()

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, value: float) -> None:
        self._offset = value
        self.update()

    offset = pyqtProperty(float, _get_offset, _set_offset)

    def _refresh_animation(self) -> None:
        if not self._text:
            self._anim_group.stop()
            self._offset = 0.0
            return

        available = max(0, self.width() - self._padding * 2)
        if available <= 0:
            self._anim_group.stop()
            self._offset = 0.0
            return

        text_width = self.fontMetrics().horizontalAdvance(self._text)
        max_offset = max(0.0, float(text_width - available))

        if max_offset <= 0:
            self._anim_group.stop()
            self._offset = 0.0
            self.update()
            return

        if abs(max_offset - self._max_offset) < 1.0 and self._anim_group.state() != 0:
            return

        self._max_offset = max_offset
        duration_ms = max(2500, int((max_offset / 40.0) * 1000))
        duration_ms = min(duration_ms, 14000)

        self._anim_group.stop()
        self._anim_forward.setStartValue(0.0)
        self._anim_forward.setEndValue(max_offset)
        self._anim_forward.setDuration(duration_ms)
        self._anim_backward.setStartValue(max_offset)
        self._anim_backward.setEndValue(0.0)
        self._anim_backward.setDuration(duration_ms)
        self._anim_group.start()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setClipRect(self.rect())
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))

        metrics = self.fontMetrics()
        y = (self.height() + metrics.ascent() - metrics.descent()) // 2
        x = int(self._padding - self._offset)
        painter.drawText(x, y, self._text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_animation()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_animation()


class TrackWidget(QWidget):
    """
    Widget representing a single audio track with controls and waveform display.
    """

    CONTROLS_WIDTH = 140

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
        
        # Connect segment moved signal
        self._waveform.segmentMoved.connect(self._on_segment_moved)
        
        if self._track.clips or self._track.data is not None:
            logger.info("Setting waveform data: %d clips", len(self._track.clips))
            self._waveform.blockSignals(True)
            self._waveform.set_data(
                None, # Don't pass flattened data if using clips
                splits=self._track.splits, 
                start_offset=self._track.start_offset,
                clips=self._track.clips
            )
            self._waveform.blockSignals(False)
        
        self._layout.addWidget(self._controls_widget)
        self._layout.addWidget(self._waveform, stretch=1)
        
        self.setStyleSheet(
            "background-color: #171c25; border: 1px solid #252c3a; border-radius: 6px;"
        )
    
    def _create_controls_panel(self) -> None:
        """Create the left control panel."""
        self._controls_widget = QWidget()
        self._controls_widget.setFixedWidth(self.CONTROLS_WIDTH)
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
        
        self._name_label = SlidingTextLabel(self._track.name)
        self._name_label.setStyleSheet(
            "font-weight: 600; color: #e6e8eb; font-size: 11px;"
        )
        self._name_label.setMinimumWidth(0)
        
        self._btn_close = QPushButton()
        self._btn_close.setIcon(qta.icon("fa5s.times", color="#9aa3ad"))
        self._btn_close.setIconSize(QSize(10, 10))
        self._btn_close.setFixedSize(16, 16)
        self._btn_close.setStyleSheet(
            "QPushButton { border: none; background: transparent; } "
            "QPushButton:hover { color: white; }"
        )
        self._btn_close.clicked.connect(self._on_close_clicked)
        
        name_row.addWidget(self._name_label, stretch=1)
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
    
    def _on_segment_moved(self, old_start: int, old_end: int, new_start: int, new_end: int) -> None:
        """Handle segment moved signal from waveform widget."""
        track_idx = self._get_track_index()
        if track_idx >= 0:
            if self._audio_engine.move_segment(track_idx, old_start, old_end, new_start):
                 # Update view
                 self._waveform.set_data(
                     None, 
                     splits=self._track.splits, 
                     start_offset=self._track.start_offset,
                     clips=self._track.clips
                 )
                 logger.info("Segment moved from %d-%d to %d-%d", old_start, old_end, new_start, new_end)
