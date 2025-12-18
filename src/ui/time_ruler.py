from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
import math

class TimeRulerWidget(QWidget):
    seekRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setMinimumWidth(100)
        
        self.visible_start = 0
        self.visible_len = 0
        self.total_samples = 0
        self.samplerate = 44100
        
        self.bg_color = QColor(30, 30, 30)
        self.text_color = QColor(200, 200, 200)
        self.line_color = QColor(100, 100, 100)
        self.playhead_color = QColor(255, 50, 50)
        
        self.left_offset = 140 # Match TrackWidget controls width
        self.playhead_sample = 0
        self.dragging_playhead = False

    def set_view(self, start, length, total, sr=44100):
        self.visible_start = start
        self.visible_len = length
        self.total_samples = total
        self.samplerate = sr
        self.update()

    def set_playhead(self, sample):
        self.playhead_sample = sample
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Left offset area (empty or label)
        painter.fillRect(0, 0, self.left_offset, self.height(), QColor(40, 40, 40))
        
        if self.visible_len <= 0 or self.samplerate <= 0:
            return

        width = self.width() - self.left_offset
        if width <= 0:
            return

        painter.translate(self.left_offset, 0)
        
        # Calculate time range
        start_time = self.visible_start / self.samplerate
        end_time = (self.visible_start + self.visible_len) / self.samplerate
        duration = end_time - start_time
        
        # Determine interval
        # We want a tick every ~60-100 pixels
        pixels_per_second = width / duration
        
        # Possible intervals in seconds
        intervals = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 3600]
        interval = intervals[-1]
        for i in intervals:
            if i * pixels_per_second >= 60:
                interval = i
                break
        
        # Safety: if interval is still too small for the duration, scale it up
        # Max 100 major ticks to prevent freezing
        if duration / interval > 100:
            interval = duration / 10
        
        # Start drawing from the first interval multiple before start_time
        first_tick = math.floor(start_time / interval) * interval
        
        painter.setPen(QPen(self.line_color, 1))
        painter.setFont(QFont("Arial", 8))
        
        curr_tick = first_tick
        while curr_tick <= end_time:
            if curr_tick >= start_time:
                x = (curr_tick - start_time) * pixels_per_second
                
                # Draw major tick
                painter.drawLine(int(x), self.height() - 10, int(x), self.height())
                
                # Draw label
                minutes = int(curr_tick // 60)
                seconds = curr_tick % 60
                if interval < 1:
                    label = f"{minutes}:{seconds:04.1f}"
                else:
                    label = f"{minutes}:{int(seconds):02d}"
                
                painter.setPen(self.text_color)
                painter.drawText(int(x) + 2, self.height() - 12, label)
                painter.setPen(self.line_color)
                
                # Draw minor ticks (only if they won't be too crowded)
                minor_interval = interval / 5
                if minor_interval * pixels_per_second > 5:
                    for j in range(1, 5):
                        minor_tick = curr_tick + j * minor_interval
                        if minor_tick <= end_time:
                            mx = (minor_tick - start_time) * pixels_per_second
                            painter.drawLine(int(mx), self.height() - 5, int(mx), self.height())

            curr_tick += interval
            # Emergency break if something goes wrong
            if interval <= 0: break 

        # Draw Playhead indicator
        playhead_offset = self.playhead_sample - self.visible_start
        if 0 <= playhead_offset <= self.visible_len:
            ph_x = (playhead_offset / self.visible_len) * width
            
            # Draw a subtle vertical line
            painter.setPen(QPen(self.playhead_color, 1))
            painter.drawLine(int(ph_x), 0, int(ph_x), self.height())
            
            # Playhead Handle (Triangle at the bottom, pointing down)
            handle_w = 12
            handle_h = 10
            triangle = QPolygonF([
                QPointF(ph_x - handle_w/2, self.height() - handle_h),
                QPointF(ph_x + handle_w/2, self.height() - handle_h),
                QPointF(ph_x, self.height())
            ])
            painter.setBrush(self.playhead_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(triangle)
            
            # Optional: Draw a small line at the top too for better visibility
            painter.setPen(QPen(self.playhead_color, 2))
            painter.drawLine(int(ph_x), 0, int(ph_x), 5)

    def mousePressEvent(self, event):
        x = event.position().x()
        if x < self.left_offset:
            return
            
        width = self.width() - self.left_offset
        
        # Check for playhead hit
        playhead_offset = self.playhead_sample - self.visible_start
        ph_x = (playhead_offset / self.visible_len) * width + self.left_offset
        
        if abs(x - ph_x) < 15:
            self.dragging_playhead = True
        else:
            # Jump to position
            ratio = (x - self.left_offset) / width
            sample = int(self.visible_start + ratio * self.visible_len)
            sample = max(0, min(sample, self.total_samples))
            self.seekRequested.emit(sample)

    def mouseMoveEvent(self, event):
        if self.dragging_playhead:
            x = event.position().x()
            width = self.width() - self.left_offset
            if width <= 0: return
            
            ratio = (x - self.left_offset) / width
            ratio = max(0, min(1, ratio))
            
            sample = int(self.visible_start + ratio * self.visible_len)
            sample = max(0, min(sample, self.total_samples))
            
            self.seekRequested.emit(sample)

    def mouseReleaseEvent(self, event):
        self.dragging_playhead = False
