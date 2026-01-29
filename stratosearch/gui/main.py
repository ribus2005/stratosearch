import sys
from PySide6.QtWidgets import QApplication

from stratosearch.gui.UI import SettingWidget


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SettingWidget()
    window.show()
    sys.exit(app.exec())
