import sys
from PyQt6.QtWidgets import QApplication
import qdarktheme

# Import the actual MainWindow
from src.ui.main_window import MainWindow

CUSTOM_STYLESHEET = """
QMainWindow {
    background-color: #11151c;
}
QWidget {
    color: #e6e8eb;
    font-family: "Bahnschrift", "Segoe UI";
    font-size: 12px;
}
QMenuBar {
    background-color: #151922;
    border-bottom: 1px solid #252c3a;
}
QMenuBar::item {
    padding: 6px 10px;
    background: transparent;
}
QMenuBar::item:selected {
    background-color: #202736;
}
QMenu {
    background-color: #181c25;
    border: 1px solid #2b3446;
}
QMenu::item {
    padding: 6px 22px 6px 20px;
}
QMenu::item:selected {
    background-color: #232b3d;
}
QToolBar {
    background-color: #151922;
    border-bottom: 1px solid #252c3a;
    spacing: 6px;
    padding: 4px;
}
QToolButton {
    background-color: #1f2533;
    border: 1px solid #2c3444;
    border-radius: 6px;
    padding: 6px;
}
QToolButton:hover {
    background-color: #283145;
}
QToolButton:pressed {
    background-color: #1a202d;
}
QStatusBar {
    background-color: #151922;
    border-top: 1px solid #252c3a;
    padding: 4px;
}
QScrollBar:horizontal {
    background-color: #11151c;
    height: 12px;
    margin: 2px 6px;
    border: 1px solid #242b39;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background-color: #2b3446;
    border-radius: 5px;
    min-width: 40px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #3b4760;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0px;
}
QDialog {
    background-color: #151922;
}
QPushButton {
    background-color: #202736;
    border: 1px solid #2f384b;
    border-radius: 6px;
    padding: 6px 10px;
}
QPushButton:hover {
    background-color: #283145;
}
QPushButton:pressed {
    background-color: #1a202d;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #10131a;
    border: 1px solid #2c3444;
    border-radius: 6px;
    padding: 4px 6px;
}
QProgressBar {
    background-color: #10131a;
    border: 1px solid #2c3444;
    border-radius: 6px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2cc7c9;
    border-radius: 6px;
}
"""

def main():
    app = QApplication(sys.argv)
    
    # Apply modern dark theme
    base_stylesheet = qdarktheme.load_stylesheet(theme="dark")
    app.setStyleSheet(base_stylesheet + CUSTOM_STYLESHEET)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
