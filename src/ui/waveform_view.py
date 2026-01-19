"""
Waveform visualization widget for PyAudioEditor.
Optimized for performance with large audio files.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRect, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QPalette, QPolygonF, QImage, QPixmap, QPolygon, QBrush, QPainterPath

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
    segmentMoved = pyqtSignal(int, int, int, int)  # old_start, old_end, new_start, new_end
    
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
        self._split_color = QColor(255, 255, 255, 110)
        self._segment_outline_color = QColor(30, 80, 160)  # Dark blue outline
        self._selection_color = QColor(self._color)
        self._selection_color.setAlpha(WAVEFORM_CONFIG.selection_alpha)
        
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
        self._cached_envelope: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._cached_envelope_view: tuple[int, int, int] = (0, 0, 0)

        # Cached waveform pixmap for fast redraws
        self._waveform_pixmap: Optional[QPixmap] = None
        self._waveform_dirty: bool = True
        
        # Segment dragging state
        self._dragging_segment: Optional[tuple[int, int]] = None  # (start_sample, end_sample)
        self._drag_start_x: Optional[float] = None
        self._drag_offset_samples: int = 0
        self._last_playhead_x: Optional[int] = None
        
        # Middle button panning state
        self._middle_dragging: bool = False
        self._middle_drag_start_x: float = 0
        self._middle_drag_start_visible: int = 0
        
        # Timeline extension (can be larger than audio data)
        self._timeline_duration_samples: int = 0  # 0 = use audio length
        
        # Audio position offset in timeline
        self._start_offset: int = 0 # Offset in samples from timeline start

        self.setStyleSheet("background-color: transparent;")
        self.setMouseTracking(True)
    
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
    
    def _get_timeline_duration(self) -> int:
        """Get the total timeline duration in samples (can be larger than audio)."""
        if self._timeline_duration_samples > 0:
            return self._timeline_duration_samples
        if self._audio_data is not None:
            return len(self._audio_data)
        return 0
    
    def set_timeline_duration(self, duration_samples: int) -> None:
        """Set the timeline duration (can be larger than audio data)."""
        self._timeline_duration_samples = max(0, duration_samples)
        self._waveform_dirty = True
        self.update()
        self.viewChanged.emit()
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    def set_playhead(self, sample_index: int) -> None:
        """Update playhead position."""
        if self._playhead_sample == sample_index:
            return

        self._playhead_sample = sample_index

        if self._visible_len > 0 and self.rect().width() > 1:
            draw_width = self.rect().width() - 1
            ph_x = int(round(((sample_index - self._visible_start) / self._visible_len) * draw_width))
            if self._last_playhead_x is not None:
                x_min = min(self._last_playhead_x, ph_x) - 2
                x_max = max(self._last_playhead_x, ph_x) + 2
                self.update(QRect(x_min, 0, x_max - x_min + 1, self.rect().height()))
            else:
                self.update()
            self._last_playhead_x = ph_x
        else:
            self.update()
    
    def set_data(self, data: Optional[AudioArray], splits: Optional[list[int]] = None, start_offset: int = 0, clips: Optional[list] = None) -> None:
        """Set the audio data for visualization."""
        self._audio_data = data
        self._splits = splits if splits is not None else []
        self._start_offset = start_offset
        self._clips = clips
        
        # Invalidate caches
        self._spectrogram_pixmap = None
        self._spectrogram_dirty = True
        self._cached_waveform = None
        self._cached_envelope = None
        self._waveform_pixmap = None
        self._waveform_dirty = True
        
        # Reset view if not set
        if self._visible_len == 0:
            if self._clips:
                max_end = max(c.end_offset for c in self._clips)
                self._visible_len = max(1000, max_end)
                self._visible_start = 0
            elif self._audio_data is not None:
                self._visible_len = len(self._audio_data)
                self._visible_start = 0
            else:
                self._visible_len = 44100 * 10
        
        self.update()
        self.viewChanged.emit()
    
    def set_view(self, start_sample: int, length_samples: int) -> None:
        """Set visible range of the waveform."""
        if self._visible_start != start_sample or self._visible_len != length_samples:
            self._visible_start = start_sample
            self._visible_len = length_samples
            self._cached_waveform = None  # Invalidate cache
            self._cached_envelope = None
            self._waveform_dirty = True
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
        painter.fillRect(self.rect(), QColor(15, 18, 24))
        
        rect = self.rect()
        width, height = rect.width(), rect.height()

        if self._clips:
            # Draw clips
            for clip in self._clips:
                self._draw_clip_item(painter, clip, width, height)
                
            # Draw ghost for dragging
            if self._dragging_segment and isinstance(self._dragging_segment, (list, tuple)) is False: # Check if it's an object, not just tuple
                 clip = self._dragging_segment
                 if hasattr(clip, 'start_offset'):
                     self._draw_ghost_clip(painter, clip, width, height)
        elif self._audio_data is not None:
            # Legacy mode
            if self._show_spectrogram:
                self._draw_spectrogram(painter, width, height)
            else:
                self._draw_waveform(painter, width, height)
            
            # Legacy overlays
            self._draw_segment_outlines(painter, height, width)
            self._draw_splits(painter, height, width)
        else:
             painter.setPen(QColor(154, 163, 173))
             painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Audio Loaded")

        # Common overlays
        self._draw_selection(painter, height, width)
        self._draw_playhead(painter, height, width)

    def _draw_ghost_clip(self, painter: QPainter, clip, width: int, height: int):
        """Draw outline of clip at dragged position."""
        original_start = getattr(self, '_drag_original_start', clip.start_offset)
        new_start = original_start + self._drag_offset_samples
        length = len(clip.data)
        
        vis_start = self._visible_start
        vis_len = self._visible_len
        
        start_x = int(((new_start - vis_start) / vis_len) * width)
        end_x = int(((new_start + length - vis_start) / vis_len) * width)
        
        if end_x > 0 and start_x < width:
            w = max(1, end_x - start_x)
            ghost_rect = QRect(start_x, 0, w, height)
            painter.setPen(QPen(QColor(255, 255, 255, 150), 2, Qt.PenStyle.DashLine))
            painter.drawRect(ghost_rect)

    def _draw_clip_item(self, painter: QPainter, clip, width: int, height: int):
        """Draw a single clip item."""
        start_sample = clip.start_offset
        if hasattr(clip, 'data'):
            data = clip.data
        else:
            return
            
        length = len(data)
        end_sample = start_sample + length
        
        vis_start = self._visible_start
        vis_len = self._visible_len
        
        # Check intersection
        if end_sample < vis_start or start_sample > vis_start + vis_len:
            return
            
        # Screen coords
        x_start = int(((start_sample - vis_start) / vis_len) * width)
        x_end = int(((end_sample - vis_start) / vis_len) * width)
        w = max(1, x_end - x_start)
        
        # Clip area
        painter.save()
        # Ensure we don't draw outside widget if clip extends way out
        painter.setClipRect(max(0, x_start), 0, w, height)
        
        # Background
        painter.fillRect(x_start, 0, w, height, QColor(40, 44, 52))
        
        # Draw waveform for this clip
        # Simplify Logic: Extract data visible
        rel_vis_start = max(0, vis_start - start_sample)
        rel_vis_end = min(length, vis_start + vis_len - start_sample)
        
        if rel_vis_end > rel_vis_start:
            view_data = data[rel_vis_start:rel_vis_end]
            if view_data.ndim > 1:
                view_data = view_data[:, 0] # Mono for viz
                
            # Draw
            # Calculate step
            pixels = max(1, int((len(view_data) / vis_len) * width))
            step = max(1, len(view_data) // pixels)
            
            mid_y = height / 2
            painter.setPen(QPen(self._color, 1))
            
            # Simple decimation for standard zoom
            plot_data = view_data[::step]
            
            # Adjust x mapping
            # This is tricky: we are mapping rel_vis_start...rel_vis_end to screen
            # screen_start for this chunk = x_start + offset into clip
            
            # Actually simpler: mapping sample index to x
            # idx i in plot_data corresponds to view_data[i*step]
            # absolute sample = start_sample + rel_vis_start + i*step
            
            poly = QPolygon()
            # Optimisation: use QPolygon instead of individual lines
            # But QPolygon takes integers, QPolygonF floats
            
            # Precompute constants
            x_factor = width / vis_len
            y_factor = height / 2 * 0.9
            base_sample = start_sample + rel_vis_start
            
            for i, val in enumerate(plot_data):
                abs_sample = base_sample + (i * step)
                px = int((abs_sample - vis_start) * x_factor)
                py = int(mid_y - (val * y_factor))
                poly.append(QPoint(px, py))
                
            painter.drawPolyline(poly)
            
        # Border
        painter.setPen(QPen(QColor(60, 64, 72), 1))
        painter.drawRect(x_start, 0, w, height)
        
        # Clip Name Label
        if hasattr(clip, 'name') and w > 20:
             painter.setPen(QColor(200, 200, 200))
             painter.drawText(QRect(x_start + 5, 2, w - 10, 20), Qt.AlignmentFlag.AlignLeft, clip.name)

        painter.restore()

    def _draw_waveform(self, painter: QPainter, width: int, height: int) -> None:
        """Legacy: Draw waveform from cached pixmap."""
        pixmap = self._get_waveform_pixmap(width, height)
        if pixmap is not None:
            painter.drawPixmap(0, 0, pixmap)

    def _get_waveform_pixmap(self, width: int, height: int) -> Optional[QPixmap]:
        if self._audio_data is None or width <= 0 or height <= 0:
            return None

        if self._waveform_pixmap is None or self._waveform_dirty:
            if (
                self._waveform_pixmap is None
                or self._waveform_pixmap.width() != width
                or self._waveform_pixmap.height() != height
            ):
                self._waveform_pixmap = QPixmap(width, height)

            self._waveform_pixmap.fill(Qt.GlobalColor.transparent)
            pm_painter = QPainter(self._waveform_pixmap)
            pm_painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            self._render_waveform(pm_painter, width, height)
            pm_painter.end()
            self._waveform_dirty = False

        return self._waveform_pixmap

    def _render_waveform(self, painter: QPainter, width: int, height: int) -> None:
        """Render waveform into a cached pixmap."""
        channel_data = self._audio_data[:, 0] if self._audio_data.ndim > 1 else self._audio_data
        total_samples = len(channel_data)

        v_start = self._visible_start
        v_len = self._visible_len

        # bounds in channel_data
        d_start = max(0, int(v_start - self._start_offset))
        d_end = min(total_samples, int(v_start + v_len - self._start_offset))

        if d_end <= d_start:
            return

        view_data = channel_data[d_start:d_end]

        # Actual timeline start for the data we are drawing
        draw_timeline_start = d_start + self._start_offset

        step = max(1, int(v_len / width))
        mid_y = height / 2
        painter.setPen(self._color)

        if step > 2:
            current_view = (d_start, d_end, step)
            if self._cached_envelope_view == current_view and self._cached_envelope is not None:
                mins, maxs = self._cached_envelope
            else:
                n_blocks = len(view_data) // step
                if n_blocks > 0:
                    try:
                        reshaped = view_data[:n_blocks * step].reshape(n_blocks, step)
                        mins = reshaped.min(axis=1)
                        maxs = reshaped.max(axis=1)
                        self._cached_envelope = (mins, maxs)
                        self._cached_envelope_view = current_view
                    except (ValueError, MemoryError):
                        mins = view_data[::step]
                        maxs = view_data[::step]
                else:
                    mins = view_data
                    maxs = view_data

            for i in range(len(mins)):
                timeline_sample = draw_timeline_start + (i * step)
                x = ((timeline_sample - v_start) / v_len) * width
                y_min = mid_y - (mins[i] * mid_y * 0.9)
                y_max = mid_y - (maxs[i] * mid_y * 0.9)
                painter.drawLine(int(x), int(y_min), int(x), int(y_max))
        else:
            plot_data = view_data[::step]
            poly = QPolygonF()
            for i, val in enumerate(plot_data):
                timeline_sample = draw_timeline_start + (i * step)
                x = ((timeline_sample - v_start) / v_len) * width
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

            max_duration = SPECTROGRAM_CONFIG.max_duration_for_full_stft
            if max_duration > 0:
                max_samples = int(max_duration * self._sr)
                if max_samples > 0 and len(y) > max_samples:
                    stride = max(1, int(np.ceil(len(y) / max_samples)))
                    y = y[::stride]

            n_fft = SPECTROGRAM_CONFIG.n_fft
            hop_length = SPECTROGRAM_CONFIG.hop_length
            if max_duration > 0 and len(y) < SPECTROGRAM_CONFIG.n_fft:
                n_fft = max(256, len(y) // 2)
                hop_length = max(64, n_fft // 4)
            
            # STFT
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)),
                ref=np.max,
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
            draw_width = max(1, width - 1)
            ph_x = (offset / self._visible_len) * draw_width
            painter.setPen(QPen(self._playhead_color, 1))
            painter.drawLine(int(round(ph_x)), 0, int(round(ph_x)), height)
    
    def _draw_splits(self, painter: QPainter, height: int, width: int) -> None:
        """Draw split point markers."""
        painter.setPen(QPen(self._split_color, 1, Qt.PenStyle.DashLine))
        for split in self._splits:
            timeline_pos = split + self._start_offset
            offset = timeline_pos - self._visible_start
            if 0 <= offset <= self._visible_len:
                sx = (offset / self._visible_len) * width
                painter.drawLine(int(sx), 0, int(sx), height)
    
    def _draw_segment_outlines(self, painter: QPainter, height: int, width: int) -> None:
        """Draw dark blue outlines around each segment."""
        if self._audio_data is None or self._visible_len == 0:
            return
        
        total_samples = len(self._audio_data)
        pen = QPen(self._segment_outline_color, 2)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Build list of segment boundaries (relative to audio start)
        boundaries_rel = [0] + sorted(self._splits) + [total_samples]
        
        for i in range(len(boundaries_rel) - 1):
            seg_start_rel = boundaries_rel[i]
            seg_end_rel = boundaries_rel[i + 1]
            
            # Map to timeline
            seg_start = seg_start_rel + self._start_offset
            seg_end = seg_end_rel + self._start_offset
            
            # Check if this is the segment being dragged
            is_dragging = (self._dragging_segment is not None and 
                          self._dragging_segment[0] == seg_start_rel and 
                          self._dragging_segment[1] == seg_end_rel)
            
            # Calculate visible portion of this segment
            vis_start = max(seg_start, self._visible_start)
            vis_end = min(seg_end, self._visible_start + self._visible_len)
            
            if vis_start >= vis_end:
                continue  # Segment not visible
            
            # Convert to pixel coordinates
            x1 = int(((vis_start - self._visible_start) / self._visible_len) * width)
            x2 = int(((vis_end - self._visible_start) / self._visible_len) * width)
            
            # Draw segment outline rectangle with padding
            margin = 2
            rect_x = x1 + margin
            rect_y = margin
            rect_w = max(1, x2 - x1 - margin * 2)
            rect_h = height - margin * 2
            
            if is_dragging:
                # Highlight the segment being dragged with a brighter outline
                highlight_pen = QPen(QColor(80, 140, 220), 3)
                painter.setPen(highlight_pen)
                painter.drawRect(rect_x, rect_y, rect_w, rect_h)
                
                # Draw ghost preview at new position
                if self._drag_offset_samples != 0:
                    ghost_start = seg_start + self._drag_offset_samples
                    ghost_end = seg_end + self._drag_offset_samples
                    
                    vis_ghost_start = max(ghost_start, self._visible_start)
                    vis_ghost_end = min(ghost_end, self._visible_start + self._visible_len)
                    
                    if vis_ghost_start < vis_ghost_end:
                        gx1 = int(((vis_ghost_start - self._visible_start) / self._visible_len) * width)
                        gx2 = int(((vis_ghost_end - self._visible_start) / self._visible_len) * width)
                        
                        ghost_color = QColor(100, 160, 255, 120)
                        painter.setPen(QPen(ghost_color, 2, Qt.PenStyle.DashLine))
                        painter.drawRect(gx1 + margin, rect_y, max(1, gx2 - gx1 - margin * 2), rect_h)
            else:
                painter.setPen(pen)
                painter.drawRect(rect_x, rect_y, rect_w, rect_h)
    
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
        rel_sample = click_sample - self._start_offset
        
        # Find segment containing click (relative to audio start)
        start_rel = 0
        end_rel = len(self._audio_data)
        
        for split in self._splits:
            if split <= rel_sample:
                start_rel = split
            else:
                end_rel = split
                break
        
        # Select segment (in timeline coordinates)
        start_timeline = start_rel + self._start_offset
        end_timeline = end_rel + self._start_offset
        
        self._selection_start = ((start_timeline - self._visible_start) / self._visible_len) * width
        self._selection_end = ((end_timeline - self._visible_start) / self._visible_len) * width
        self.update()
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse press for selection, segment dragging, or panning."""
        x = event.position().x()
        width = self.rect().width()
        
        # Middle button for panning
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_dragging = True
            self._middle_drag_start_x = x
            self._middle_drag_start_visible = self._visible_start
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        
        # Left button only for dragging/selection
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        # Shift+Click for area selection
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._selection_start = x
            self._selection_end = x
            self.update()
            return
        
        # Left click (no modifier) to start segment dragging
        if self._visible_len == 0:
             return
             
        ratio = x / width
        click_sample = int(self._visible_start + ratio * self._visible_len)
        
        # NEW: Clip Dragging
        if self._clips:
            for clip in self._clips:
                if clip.start_offset <= click_sample < clip.end_offset:
                    self._dragging_segment = clip
                    self._drag_start_x = x
                    self._drag_offset_samples = 0
                    self._drag_original_start = clip.start_offset
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return
        
        # Legacy Dragging
        if self._audio_data is not None:
             rel_sample = click_sample - self._start_offset
             total_samples = len(self._audio_data)
             boundaries = [0] + sorted(self._splits) + [total_samples]
            
             for i in range(len(boundaries) - 1):
                seg_start = boundaries[i]
                seg_end = boundaries[i + 1]
                if seg_start <= rel_sample < seg_end:
                    self._dragging_segment = (seg_start, seg_end)
                    self._drag_start_x = x
                    self._drag_offset_samples = 0
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return
    
    def mouseMoveEvent(self, event) -> None:
        """Handle drag."""
        x = event.position().x()
        width = self.rect().width()
        x = max(0, min(x, width))
        
        # Pan
        if self._middle_dragging:
            if self._visible_len == 0 or width == 0: return
            
            samples_per_pixel = self._visible_len / width
            delta_x = self._middle_drag_start_x - x
            new_visible_start = int(self._middle_drag_start_visible + delta_x * samples_per_pixel)
            
            total = self._get_timeline_duration()
            new_visible_start = max(0, min(new_visible_start, total - self._visible_len))
            
            if new_visible_start != self._visible_start:
                self._visible_start = new_visible_start
                self._waveform_dirty = True
                self._cached_waveform = None
                self.update()
                self.viewChanged.emit()
            return

        # Drag Segment
        if self._dragging_segment is not None and self._drag_start_x is not None:
             delta_x = x - self._drag_start_x
             self._drag_offset_samples = int((delta_x / width) * self._visible_len)
             self.update()
             return
             
        # Selection
        if self._selection_start is not None:
             self._selection_end = x
             self.update()

    def mouseReleaseEvent(self, event) -> None:
        """Finish operations."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
            
        if self._dragging_segment is not None:
            # Handle Clip
            if not isinstance(self._dragging_segment, (tuple, list)):
                 clip = self._dragging_segment
                 original_start = getattr(self, '_drag_original_start', clip.start_offset)
                 new_start = original_start + self._drag_offset_samples
                 
                 # Clamp
                 new_start = max(0, new_start)
                 
                 if new_start != original_start:
                     self.segmentMoved.emit(original_start, clip.end_offset, new_start, new_start + len(clip.data))
                     
            else:
                # Handle Legacy
                old_start, old_end = self._dragging_segment
                new_start = old_start + self._drag_offset_samples
                new_end = old_end + self._drag_offset_samples
                
                # Logic for legacy clamp was relative to start_offset?
                # Actually legacy move_segment updated track.start_offset.
                # Just emit what we have.
                if self._drag_offset_samples != 0:
                     self.segmentMoved.emit(old_start, old_end, new_start, new_end)
            
            self._dragging_segment = None
            self._drag_start_x = None
            self._drag_offset_samples = 0
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return
            
        # Selection / Seek
        if self._selection_start is not None:
            if abs(self._selection_start - (self._selection_end or self._selection_start)) < 3:
                # Click = Seek
                ratio = self._selection_start / self.rect().width()
                new_sample = int(self._visible_start + ratio * self._visible_len)
                self.set_playhead(new_sample)
                self.seekRequested.emit(new_sample)
                self._selection_start = None
                self._selection_end = None
                self.update()

    def resizeEvent(self, event) -> None:
        """Handle resize to refresh cached visuals."""
        self._waveform_dirty = True
        self._spectrogram_dirty = True
        self._waveform_pixmap = None
        self._last_playhead_x = None
        super().resizeEvent(event)
    
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
        self._waveform_dirty = True
        
        self.update()
        self.viewChanged.emit()
