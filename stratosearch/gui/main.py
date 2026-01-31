import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from stratosearch.gui.UI import SettingWidget


def get_app_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_dir = get_app_dir()
    with open(app_dir / "UI" / "View" / "style.qss", "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())
    window = SettingWidget(app_dir)
    window.show()
    sys.exit(app.exec())
