from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QSize, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPalette, QPolygonF, QImage, QPixmap
import numpy as np
import librosa

class WaveformWidget(QWidget):
    viewChanged = pyqtSignal()
    seekRequested = pyqtSignal(int) # Emits sample index
    splitRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self.splits = []
        self.sr = 44100
        self.setMinimumHeight(100)
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.Base)
        self.setSizePolicy(
            self.sizePolicy().Policy.Expanding,
            self.sizePolicy().Policy.Expanding
        )
        self.color = QColor(0, 255, 255) # Cyan neon
        self.playhead_color = QColor(255, 50, 50) # Red playhead
        self.split_color = QColor(255, 255, 255, 150)
        
        # Spectrogram State
        self.show_spectrogram = False
        self.spectrogram_pixmap = None
        self.spectrogram_dirty = False
        
        # Selection state (in pixels for now -> mapped to samples in get_selection)
        self.selection_start = None
        self.selection_end = None
        
        # Playhead state (in samples)
        self.playhead_sample = 0
        self.dragging_playhead = False
        
        # View state
        self.visible_start = 0  # Sample index
        self.visible_len = 0    # Number of samples visible
        
    def set_playhead(self, sample_index):
        if self.playhead_sample != sample_index:
            self.playhead_sample = sample_index
            self.update() # Trigger repaint

    def set_data(self, data, sr=44100, splits=None):
        """Sets the audio data for visualization."""
        self.audio_data = data
        self.sr = sr
        self.splits = splits if splits is not None else []
        self.spectrogram_pixmap = None # Reset cache
        self.spectrogram_dirty = True
        if data is not None:
            # Don't reset view if we are just updating data (e.g. effect applied)
            if self.visible_len == 0:
                self.visible_start = 0
                self.visible_len = len(data)
        self.update()
        self.viewChanged.emit()

    def set_view(self, start_sample, length_samples):
        """Sets the visible range of the waveform."""
        self.visible_start = start_sample
        self.visible_len = length_samples
        self.update()
        self.viewChanged.emit()

    def toggle_spectrogram(self, enabled):
        self.show_spectrogram = enabled
        self.update()

    def generate_spectrogram_pixmap(self, width, height):
        if self.audio_data is None: return None
        
        # Take first channel
        y = self.audio_data
        if y.ndim > 1: y = y[:, 0]
        
        # Downsample for speed if very long? 
        # Librosa stft is fast enough for reasonable lengths, but full song might be slow.
        # Let's generate a low-res version for UI
        
        # Convert to Mel Spectrogram
        try:
            # Simple STFT is faster
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
            
            # Normalize to 0-255
            D_norm = (D - D.min()) / (D.max() - D.min()) * 255
            D_norm = D_norm.astype(np.uint8)
            
            # Flip Y (freqs up)
            D_norm = np.flipud(D_norm)
            
            # CRITICAL: Ensure data is valid and contiguous in memory before passing to Qt
            D_norm = np.ascontiguousarray(D_norm)
            
            # Map to color (heatmap) - simple grayscale for now or generate QImage
            # Creating QImage from numpy array
            h, w = D_norm.shape
            
            # Ensure 32-bit aligned stride if needed, but simple bytes works
            # We must keep D_norm alive while QImage exists, but we convert immediately to Pixmap (copy)
            img = QImage(D_norm.data, w, h, w, QImage.Format.Format_Grayscale8)
            
            # Apply a colormap? Qt doesn't make it easy directly on 8bit without lookup table
            # For "Heatmap", we can convert to RGB manually or just use Grayscale for v1
            
            # Resize to widget size only when drawing? No, pixmap should hold full data? 
            # Actually better to hold full spectrogram image and draw slice.
            return QPixmap.fromImage(img)
            
        except Exception as e:
            # Catching all to prevent crash
            print(f"Spectrogram Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if self.audio_data is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Audio Loaded")
            return

        rect = self.rect()
        width, height = rect.width(), rect.height()
        
        if self.show_spectrogram:
            if self.spectrogram_dirty or self.spectrogram_pixmap is None:
                 self.spectrogram_pixmap = self.generate_spectrogram_pixmap(width, height)
                 self.spectrogram_dirty = False
            
            if self.spectrogram_pixmap:
                 total_len = len(self.audio_data)
                 spec_w = self.spectrogram_pixmap.width()
                 
                 inter_start = max(0, self.visible_start)
                 inter_end = min(total_len, self.visible_start + self.visible_len)
                 
                 if inter_end > inter_start:
                     src_x = int((inter_start / total_len) * spec_w)
                     src_w = int(((inter_end - inter_start) / total_len) * spec_w)
                     
                     target_x = int(((inter_start - self.visible_start) / self.visible_len) * width)
                     target_w = int(((inter_end - inter_start) / self.visible_len) * width)
                     
                     painter.drawPixmap(QRect(target_x, 0, target_w, height), 
                                        self.spectrogram_pixmap, 
                                        QRect(src_x, 0, src_w, self.spectrogram_pixmap.height()))
        else:
            # Optimized Waveform Rendering
            channel_data = self.audio_data[:, 0] if self.audio_data.ndim > 1 else self.audio_data
            total_samples = len(channel_data)
            
            # Visible range in samples
            v_start = self.visible_start
            v_len = self.visible_len
            
            # Data range to actually draw (intersection of visible window and audio data)
            d_start = max(0, int(v_start))
            d_end = min(total_samples, int(v_start + v_len))
            
            if d_end <= d_start: return
            
            view_data = channel_data[d_start:d_end]
            # Step should be based on the visible window size to maintain consistent density
            step = max(1, int(v_len / width))
            plot_data = view_data[::step]
            
            mid_y = height / 2
            painter.setPen(self.color)
            
            # Use QPolygonF for faster drawing of many points
            poly = QPolygonF()
            for i, val in enumerate(plot_data):
                # Absolute sample index of this point
                abs_sample = d_start + (i * step)
                # Pixel x relative to the visible window's start
                x = ((abs_sample - v_start) / v_len) * width
                y = mid_y - (val * mid_y * 0.9)
                poly.append(QPointF(x, y))
            painter.drawPolyline(poly)

        # Overlays
        self._draw_selection(painter, height, width)
        self._draw_playhead(painter, height, width)
        self._draw_splits(painter, height, width)

    def _draw_selection(self, painter, height, width):
        if self.selection_start is not None and self.selection_end is not None:
            s = min(self.selection_start, self.selection_end)
            w = abs(self.selection_end - self.selection_start)
            painter.fillRect(int(s), 0, int(w), height, QColor(255, 255, 255, 50))

    def _draw_playhead(self, painter, height, width):
        offset = self.playhead_sample - self.visible_start
        if 0 <= offset <= self.visible_len:
            ph_x = (offset / self.visible_len) * width
            painter.setPen(QPen(self.playhead_color, 1))
            painter.drawLine(int(ph_x), 0, int(ph_x), height)

    def _draw_splits(self, painter, height, width):
        painter.setPen(QPen(self.split_color, 1, Qt.PenStyle.DashLine))
        for split in self.splits:
            offset = split - self.visible_start
            if 0 <= offset <= self.visible_len:
                sx = (offset / self.visible_len) * width
                painter.drawLine(int(sx), 0, int(sx), height)

    def mouseDoubleClickEvent(self, event):
        x = event.position().x()
        width = self.rect().width()
        
        if self.visible_len == 0: return
        
        # Map click to sample
        ratio = x / width
        click_sample = int(self.visible_start + ratio * self.visible_len)
        
        # Find the segment containing this sample
        # Splits are sorted
        start = 0
        end = len(self.audio_data) if self.audio_data is not None else 0
        
        for split in self.splits:
            if split <= click_sample:
                start = split
            else:
                end = split
                break
        
        # Select this segment
        # Map samples back to pixels for visualization
        self.selection_start = ((start - self.visible_start) / self.visible_len) * width
        self.selection_end = ((end - self.visible_start) / self.visible_len) * width
        self.update()

    def mousePressEvent(self, event):
        x = event.position().x()
        y = event.position().y()
        width = self.rect().width()
        
        # Start Selection
        self.selection_start = x
        self.selection_end = x
        self.update()

    def mouseMoveEvent(self, event):
        x = event.position().x()
        width = self.rect().width()
        
        # Clamp x
        x = max(0, min(x, width))

        # Selection Drag
        if self.selection_start is not None:
            self.selection_end = x
            self.update()

    def mouseReleaseEvent(self, event):
        if self.selection_start is not None:
            # Finalize selection or Treat as simple click-to-seek if tiny drag
            if abs(self.selection_start - self.selection_end) < 3:
                 # It was a click (jump playhead here)
                 # Map click x to sample
                 width = self.rect().width()
                 ratio = self.selection_start / width
                 offset_samples = ratio * self.visible_len
                 new_sample = int(self.visible_start + offset_samples)
                 
                 self.set_playhead(new_sample)
                 self.seekRequested.emit(new_sample)
                 
                 self.selection_start = None
                 self.selection_end = None
            
            self.update()

    def wheelEvent(self, event):
        if self.audio_data is None: return
        
        angle = event.angleDelta().y()
        factor = 0.8 if angle > 0 else 1.25
        
        center_x = event.position().x()
        width = self.rect().width()
        
        # Calculate cursor sample position before zoom
        samples_per_pixel_old = self.visible_len / width
        cursor_sample_offset = center_x * samples_per_pixel_old
        cursor_sample_abs = self.visible_start + cursor_sample_offset
        
        # Apply zoom
        new_len = self.visible_len * factor
        # Limit zoom in to 100 samples, but allow zoom out
        new_len = max(100, new_len)
        
        # Re-center
        samples_per_pixel_new = new_len / width
        new_visible_start = cursor_sample_abs - (center_x * samples_per_pixel_new)
        
        self.visible_start = int(new_visible_start)
        self.visible_len = int(new_len)
        self.update()
        self.viewChanged.emit()

    def get_selection_range_samples(self):
        """ Returns tuple (start_sample, end_sample) or None """
        if self.selection_start is None or self.audio_data is None:
            return None
        
        width = self.rect().width()
        
        # Map pixel x to sample index relative to View
        if self.visible_len == 0: return None
        
        start_x = min(self.selection_start, self.selection_end)
        end_x = max(self.selection_start, self.selection_end)
        
        samples_per_pixel = self.visible_len / width
        
        start_sample = int(self.visible_start + start_x * samples_per_pixel)
        end_sample = int(self.visible_start + end_x * samples_per_pixel)
        
        # Clamp
        start_sample = max(0, min(start_sample, len(self.audio_data)))
        end_sample = max(0, min(end_sample, len(self.audio_data)))
        
        return start_sample, end_sample
