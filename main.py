import sys
from PyQt6.QtWidgets import QApplication
import qdarktheme

# Import the actual MainWindow
from src.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Apply modern dark theme
    app.setStyleSheet(qdarktheme.load_stylesheet(theme="dark"))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
