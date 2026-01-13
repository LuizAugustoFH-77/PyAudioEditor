"""
Time ruler widget for PyAudioEditor.
Displays time scale and playhead with seek functionality.
"""
from __future__ import annotations
import math
from typing import Optional

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF

from src.core.config import WAVEFORM_CONFIG


class TimeRulerWidget(QWidget):
    """
    Time ruler showing time scale, tick marks, and playhead indicator.
    Supports click-to-seek and playhead dragging.
    """
    
    seekRequested = pyqtSignal(int)  # Emits sample index
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self.setFixedHeight(30)
        self.setMinimumWidth(100)
        
        # View state
        self._visible_start: int = 0
        self._visible_len: int = 0
        self._total_samples: int = 0
        self._samplerate: int = 44100
        
        # Colors
        self._bg_color = QColor(30, 30, 30)
        self._text_color = QColor(200, 200, 200)
        self._line_color = QColor(100, 100, 100)
        self._playhead_color = QColor(*WAVEFORM_CONFIG.playhead_color)
        
        # Layout
        self._left_offset: int = 140  # Match TrackWidget controls width
        
        # Playhead state
        self._playhead_sample: int = 0
        self._dragging_playhead: bool = False
    
    @property
    def visible_start(self) -> int:
        return self._visible_start
    
    @property
    def visible_len(self) -> int:
        return self._visible_len
    
    def set_view(
        self, 
        start: int, 
        length: int, 
        total: int, 
        sr: int = 44100
    ) -> None:
        """Update the visible time range."""
        self._visible_start = start
        self._visible_len = length
        self._total_samples = total
        self._samplerate = sr
        self.update()
    
    def set_playhead(self, sample: int) -> None:
        """Update playhead position."""
        self._playhead_sample = sample
        self.update()
    
    def paintEvent(self, event) -> None:
        """Render the time ruler."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self._bg_color)
        painter.fillRect(0, 0, self._left_offset, self.height(), QColor(40, 40, 40))
        
        if self._visible_len <= 0 or self._samplerate <= 0:
            return
        
        width = self.width() - self._left_offset
        if width <= 0:
            return
        
        painter.translate(self._left_offset, 0)
        
        # Calculate time range
        start_time = self._visible_start / self._samplerate
        end_time = (self._visible_start + self._visible_len) / self._samplerate
        duration = end_time - start_time
        
        if duration <= 0:
            return
        
        # Determine tick interval
        pixels_per_second = width / duration
        
        intervals = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600]
        interval = intervals[-1]
        
        for i in intervals:
            if i * pixels_per_second >= 60:
                interval = i
                break
        
        # Limit ticks to prevent freezing
        if duration / interval > 100:
            interval = duration / 10
        
        # Draw ticks
        first_tick = math.floor(start_time / interval) * interval
        
        painter.setPen(QPen(self._line_color, 1))
        painter.setFont(QFont("Arial", 8))
        
        curr_tick = first_tick
        while curr_tick <= end_time:
            if curr_tick >= start_time:
                x = (curr_tick - start_time) * pixels_per_second
                
                # Major tick
                painter.drawLine(int(x), self.height() - 10, int(x), self.height())
                
                # Label
                minutes = int(curr_tick // 60)
                seconds = curr_tick % 60
                
                if interval < 1:
                    label = f"{minutes}:{seconds:04.1f}"
                else:
                    label = f"{minutes}:{int(seconds):02d}"
                
                painter.setPen(self._text_color)
                painter.drawText(int(x) + 2, self.height() - 12, label)
                painter.setPen(self._line_color)
                
                # Minor ticks
                minor_interval = interval / 5
                if minor_interval * pixels_per_second > 5:
                    for j in range(1, 5):
                        minor_tick = curr_tick + j * minor_interval
                        if minor_tick <= end_time:
                            mx = (minor_tick - start_time) * pixels_per_second
                            painter.drawLine(int(mx), self.height() - 5, int(mx), self.height())
            
            curr_tick += interval
            if interval <= 0:
                break
        
        # Draw playhead
        self._draw_playhead(painter, width)
    
    def _draw_playhead(self, painter: QPainter, width: int) -> None:
        """Draw playhead indicator."""
        playhead_offset = self._playhead_sample - self._visible_start
        
        if 0 <= playhead_offset <= self._visible_len:
            ph_x = (playhead_offset / self._visible_len) * width
            
            # Vertical line
            painter.setPen(QPen(self._playhead_color, 1))
            painter.drawLine(int(ph_x), 0, int(ph_x), self.height())
            
            # Triangle handle
            handle_w = 12
            handle_h = 10
            triangle = QPolygonF([
                QPointF(ph_x - handle_w / 2, self.height() - handle_h),
                QPointF(ph_x + handle_w / 2, self.height() - handle_h),
                QPointF(ph_x, self.height())
            ])
            
            painter.setBrush(self._playhead_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(triangle)
            
            # Top indicator
            painter.setPen(QPen(self._playhead_color, 2))
            painter.drawLine(int(ph_x), 0, int(ph_x), 5)
    
    def mousePressEvent(self, event) -> None:
        """Handle click to seek or start playhead drag."""
        x = event.position().x()
        
        if x < self._left_offset:
            return
        
        width = self.width() - self._left_offset
        if width <= 0:
            return
        
        # Check for playhead hit
        playhead_offset = self._playhead_sample - self._visible_start
        ph_x = (playhead_offset / self._visible_len) * width + self._left_offset
        
        if abs(x - ph_x) < 15:
            self._dragging_playhead = True
        else:
            # Jump to position
            ratio = (x - self._left_offset) / width
            sample = int(self._visible_start + ratio * self._visible_len)
            sample = max(0, min(sample, self._total_samples))
            self.seekRequested.emit(sample)
    
    def mouseMoveEvent(self, event) -> None:
        """Handle playhead drag."""
        if not self._dragging_playhead:
            return
        
        x = event.position().x()
        width = self.width() - self._left_offset
        
        if width <= 0:
            return
        
        ratio = (x - self._left_offset) / width
        ratio = max(0, min(1, ratio))
        
        sample = int(self._visible_start + ratio * self._visible_len)
        sample = max(0, min(sample, self._total_samples))
        
        self.seekRequested.emit(sample)
    
    def mouseReleaseEvent(self, event) -> None:
        """End playhead drag."""
        self._dragging_playhead = False
