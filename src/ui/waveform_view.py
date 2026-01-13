"""
Waveform visualization widget for PyAudioEditor.
Optimized for performance with large audio files.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QPalette, QPolygonF, QImage, QPixmap

from src.core.config import WAVEFORM_CONFIG, SPECTROGRAM_CONFIG
from src.core.types import AudioArray

logger = logging.getLogger("PyAudacity")


class WaveformWidget(QWidget):
    """
    Widget for displaying audio waveform with zoom, selection, and playhead.
    Optimized with downsampling and caching for large files.
    """
    
    viewChanged = pyqtSignal()
    seekRequested = pyqtSignal(int)  # Emits sample index
    splitRequested = pyqtSignal(int)
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        # Audio data
        self._audio_data: Optional[AudioArray] = None
        self._splits: list[int] = []
        self._sr: int = 44100
        
        # Display settings
        self.setMinimumHeight(100)
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.Base)
        self.setSizePolicy(
            self.sizePolicy().Policy.Expanding,
            self.sizePolicy().Policy.Expanding
        )
        
        # Colors from config
        self._color = QColor(*WAVEFORM_CONFIG.default_color)
        self._playhead_color = QColor(*WAVEFORM_CONFIG.playhead_color)
        self._split_color = QColor(255, 255, 255, 150)
        self._selection_color = QColor(255, 255, 255, WAVEFORM_CONFIG.selection_alpha)
        
        # Spectrogram state
        self._show_spectrogram = False
        self._spectrogram_pixmap: Optional[QPixmap] = None
        self._spectrogram_dirty = False
        
        # Selection state (in pixels)
        self._selection_start: Optional[float] = None
        self._selection_end: Optional[float] = None
        
        # Playhead state (in samples)
        self._playhead_sample: int = 0
        
        # View state
        self._visible_start: int = 0
        self._visible_len: int = 0
        
        # Performance: cached downsampled data
        self._cached_waveform: Optional[np.ndarray] = None
        self._cached_visible_range: tuple[int, int] = (0, 0)
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def audio_data(self) -> Optional[AudioArray]:
        return self._audio_data
    
    @property
    def sr(self) -> int:
        return self._sr
    
    @property
    def splits(self) -> list[int]:
        return self._splits
    
    @property
    def playhead_sample(self) -> int:
        return self._playhead_sample
    
    @property
    def visible_start(self) -> int:
        return self._visible_start
    
    @property
    def visible_len(self) -> int:
        return self._visible_len
    
    @property
    def show_spectrogram(self) -> bool:
        return self._show_spectrogram
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    def set_playhead(self, sample_index: int) -> None:
        """Update playhead position."""
        if self._playhead_sample != sample_index:
            self._playhead_sample = sample_index
            self.update()
    
    def set_data(
        self, 
        data: AudioArray, 
        sr: int = 44100, 
        splits: Optional[list[int]] = None
    ) -> None:
        """Set audio data for visualization."""
        self._audio_data = data
        self._sr = sr
        self._splits = splits if splits is not None else []
        
        # Invalidate caches
        self._spectrogram_pixmap = None
        self._spectrogram_dirty = True
        self._cached_waveform = None
        
        if data is not None and self._visible_len == 0:
            self._visible_start = 0
            self._visible_len = len(data)
        
        self.update()
        self.viewChanged.emit()
    
    def set_view(self, start_sample: int, length_samples: int) -> None:
        """Set visible range of the waveform."""
        if self._visible_start != start_sample or self._visible_len != length_samples:
            self._visible_start = start_sample
            self._visible_len = length_samples
            self._cached_waveform = None  # Invalidate cache
            self.update()
            self.viewChanged.emit()
    
    def toggle_spectrogram(self, enabled: bool) -> None:
        """Toggle spectrogram display mode."""
        self._show_spectrogram = enabled
        self.update()
    
    def get_selection_range_samples(self) -> Optional[tuple[int, int]]:
        """Get selection range in samples."""
        if self._selection_start is None or self._audio_data is None:
            return None
        
        width = self.rect().width()
        if self._visible_len == 0 or width == 0:
            return None
        
        start_x = min(self._selection_start, self._selection_end or 0)
        end_x = max(self._selection_start, self._selection_end or 0)
        
        samples_per_pixel = self._visible_len / width
        
        start_sample = int(self._visible_start + start_x * samples_per_pixel)
        end_sample = int(self._visible_start + end_x * samples_per_pixel)
        
        # Clamp
        data_len = len(self._audio_data)
        start_sample = max(0, min(start_sample, data_len))
        end_sample = max(0, min(end_sample, data_len))
        
        return start_sample, end_sample
    
    # =========================================================================
    # RENDERING
    # =========================================================================
    
    def paintEvent(self, event) -> None:
        """Main paint event - optimized rendering."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        
        if self._audio_data is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Audio Loaded")
            return
        
        rect = self.rect()
        width, height = rect.width(), rect.height()
        
        if self._show_spectrogram:
            self._draw_spectrogram(painter, width, height)
        else:
            self._draw_waveform(painter, width, height)
        
        # Overlays
        self._draw_selection(painter, height, width)
        self._draw_playhead(painter, height, width)
        self._draw_splits(painter, height, width)
    
    def _draw_waveform(self, painter: QPainter, width: int, height: int) -> None:
        """Optimized waveform rendering with downsampling."""
        channel_data = self._audio_data[:, 0] if self._audio_data.ndim > 1 else self._audio_data
        total_samples = len(channel_data)
        
        # Visible range
        v_start = self._visible_start
        v_len = self._visible_len
        
        d_start = max(0, int(v_start))
        d_end = min(total_samples, int(v_start + v_len))
        
        if d_end <= d_start:
            return
        
        view_data = channel_data[d_start:d_end]
        
        # Adaptive downsampling based on view width
        step = max(1, int(v_len / width))
        
        # Use min/max envelope for better visualization at high zoom-out
        if step > 2:
            # Calculate envelope (min/max per block)
            n_blocks = len(view_data) // step
            if n_blocks > 0:
                reshaped = view_data[:n_blocks * step].reshape(n_blocks, step)
                mins = reshaped.min(axis=1)
                maxs = reshaped.max(axis=1)
                
                # Draw envelope
                mid_y = height / 2
                painter.setPen(self._color)
                
                for i in range(n_blocks):
                    abs_sample = d_start + (i * step)
                    x = ((abs_sample - v_start) / v_len) * width
                    
                    y_min = mid_y - (mins[i] * mid_y * 0.9)
                    y_max = mid_y - (maxs[i] * mid_y * 0.9)
                    
                    painter.drawLine(int(x), int(y_min), int(x), int(y_max))
        else:
            # Direct plotting for zoomed-in view
            plot_data = view_data[::step]
            mid_y = height / 2
            painter.setPen(self._color)
            
            poly = QPolygonF()
            for i, val in enumerate(plot_data):
                abs_sample = d_start + (i * step)
                x = ((abs_sample - v_start) / v_len) * width
                y = mid_y - (val * mid_y * 0.9)
                poly.append(QPointF(x, y))
            
            painter.drawPolyline(poly)
    
    def _draw_spectrogram(self, painter: QPainter, width: int, height: int) -> None:
        """Draw cached spectrogram."""
        if self._spectrogram_dirty or self._spectrogram_pixmap is None:
            self._spectrogram_pixmap = self._generate_spectrogram_pixmap(width, height)
            self._spectrogram_dirty = False
        
        if self._spectrogram_pixmap is None:
            return
        
        total_len = len(self._audio_data)
        spec_w = self._spectrogram_pixmap.width()
        
        inter_start = max(0, self._visible_start)
        inter_end = min(total_len, self._visible_start + self._visible_len)
        
        if inter_end > inter_start:
            from PyQt6.QtCore import QRect
            
            src_x = int((inter_start / total_len) * spec_w)
            src_w = int(((inter_end - inter_start) / total_len) * spec_w)
            
            target_x = int(((inter_start - self._visible_start) / self._visible_len) * width)
            target_w = int(((inter_end - inter_start) / self._visible_len) * width)
            
            painter.drawPixmap(
                QRect(target_x, 0, target_w, height),
                self._spectrogram_pixmap,
                QRect(src_x, 0, src_w, self._spectrogram_pixmap.height())
            )
    
    def _generate_spectrogram_pixmap(self, width: int, height: int) -> Optional[QPixmap]:
        """Generate spectrogram image."""
        if self._audio_data is None:
            return None
        
        try:
            import librosa
            
            y = self._audio_data[:, 0] if self._audio_data.ndim > 1 else self._audio_data
            
            # STFT
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(
                    y,
                    n_fft=SPECTROGRAM_CONFIG.n_fft,
                    hop_length=SPECTROGRAM_CONFIG.hop_length
                )),
                ref=np.max
            )
            
            # Normalize to 0-255
            D_norm = ((D - D.min()) / (D.max() - D.min() + 1e-6) * 255).astype(np.uint8)
            D_norm = np.flipud(D_norm)
            D_norm = np.ascontiguousarray(D_norm)
            
            h, w = D_norm.shape
            img = QImage(D_norm.data, w, h, w, QImage.Format.Format_Grayscale8)
            
            return QPixmap.fromImage(img)
            
        except Exception as e:
            logger.error("Spectrogram error: %s", e, exc_info=True)
            return None
    
    def _draw_selection(self, painter: QPainter, height: int, width: int) -> None:
        """Draw selection highlight."""
        if self._selection_start is not None and self._selection_end is not None:
            s = min(self._selection_start, self._selection_end)
            w = abs(self._selection_end - self._selection_start)
            painter.fillRect(int(s), 0, int(w), height, self._selection_color)
    
    def _draw_playhead(self, painter: QPainter, height: int, width: int) -> None:
        """Draw playhead indicator."""
        offset = self._playhead_sample - self._visible_start
        if 0 <= offset <= self._visible_len:
            ph_x = (offset / self._visible_len) * width
            painter.setPen(QPen(self._playhead_color, 1))
            painter.drawLine(int(ph_x), 0, int(ph_x), height)
    
    def _draw_splits(self, painter: QPainter, height: int, width: int) -> None:
        """Draw split point markers."""
        painter.setPen(QPen(self._split_color, 1, Qt.PenStyle.DashLine))
        for split in self._splits:
            offset = split - self._visible_start
            if 0 <= offset <= self._visible_len:
                sx = (offset / self._visible_len) * width
                painter.drawLine(int(sx), 0, int(sx), height)
    
    # =========================================================================
    # MOUSE EVENTS
    # =========================================================================
    
    def mouseDoubleClickEvent(self, event) -> None:
        """Handle double-click to select segment."""
        x = event.position().x()
        width = self.rect().width()
        
        if self._visible_len == 0 or self._audio_data is None:
            return
        
        ratio = x / width
        click_sample = int(self._visible_start + ratio * self._visible_len)
        
        # Find segment containing click
        start = 0
        end = len(self._audio_data)
        
        for split in self._splits:
            if split <= click_sample:
                start = split
            else:
                end = split
                break
        
        # Select segment
        self._selection_start = ((start - self._visible_start) / self._visible_len) * width
        self._selection_end = ((end - self._visible_start) / self._visible_len) * width
        self.update()
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse press for selection."""
        x = event.position().x()
        self._selection_start = x
        self._selection_end = x
        self.update()
    
    def mouseMoveEvent(self, event) -> None:
        """Handle mouse drag for selection."""
        x = event.position().x()
        width = self.rect().width()
        x = max(0, min(x, width))
        
        if self._selection_start is not None:
            self._selection_end = x
            self.update()
    
    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release."""
        if self._selection_start is not None:
            # Small drag = click to seek
            if abs(self._selection_start - (self._selection_end or 0)) < 3:
                width = self.rect().width()
                ratio = self._selection_start / width
                offset_samples = ratio * self._visible_len
                new_sample = int(self._visible_start + offset_samples)
                
                self.set_playhead(new_sample)
                self.seekRequested.emit(new_sample)
                
                self._selection_start = None
                self._selection_end = None
            
            self.update()
    
    def wheelEvent(self, event) -> None:
        """Handle zoom via mouse wheel."""
        if self._audio_data is None:
            return
        
        angle = event.angleDelta().y()
        factor = 0.8 if angle > 0 else 1.25
        
        center_x = event.position().x()
        width = self.rect().width()
        
        # Calculate cursor sample before zoom
        samples_per_pixel_old = self._visible_len / width
        cursor_sample_offset = center_x * samples_per_pixel_old
        cursor_sample_abs = self._visible_start + cursor_sample_offset
        
        # Apply zoom
        new_len = self._visible_len * factor
        new_len = max(WAVEFORM_CONFIG.min_visible_samples, new_len)
        
        # Re-center
        samples_per_pixel_new = new_len / width
        new_visible_start = cursor_sample_abs - (center_x * samples_per_pixel_new)
        
        self._visible_start = int(new_visible_start)
        self._visible_len = int(new_len)
        self._cached_waveform = None
        
        self.update()
        self.viewChanged.emit()
