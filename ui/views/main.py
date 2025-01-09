from PyQt6.QtWidgets import QWidget, QMainWindow, QApplication
from PyQt6.uic import loadUi
from PyQt6.QtCore import Qt

import colorama
from pathlib import Path
from datetime import datetime

from ui.capturers import LocalCapturer
from ui.core import StitcherWrapper
from ui.utils.common import convert_cv_to_pixmap, resize_image_with_padding, save_image

SERVER_UI = Path(__file__).parent / Path('main.ui')
RECORDS_DIR = Path(__file__).parent.parent / Path('records')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(SERVER_UI, self)

        self.stitcher = StitcherWrapper(False, True, False)
        self.stitcher.warm_up()
        self.frames = []

        self.capturer = LocalCapturer()
        self.capturer.set_handle('CameraPlayer')

        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self._init_ui()

    def _init_ui(self):
        for cls in ('stitch', 'undo', 'reset'):
            getattr(self, f'button_{cls}').clicked.connect(getattr(self, f'_button_{cls}'))

    def show_canvas(self):
        image = self.stitcher.stitcher.image
        image = resize_image_with_padding(image, 480, 480)
        self.label.setPixmap(convert_cv_to_pixmap(image))

    def _button_stitch(self):
        frame = self.capturer.get_frame()
        self.frames.append(frame.copy())
        self.stitcher + frame
        self.show_canvas()

    def _button_undo(self):
        self.stitcher.stitcher.undo()
        self.show_canvas()

    def _button_reset(self):
        result_dir = RECORDS_DIR / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_dir.mkdir(exist_ok=True)
        self.stitcher.stitcher.image.save(result_dir / 'stitched.png')
        for idx, image in enumerate(self.stitcher.stitcher.previous):
            image.save(result_dir / f'stitched_{idx}.png')
        for idx, image in enumerate(self.frames):
            save_image(image, result_dir / f'frame_{idx}.png')
        print(colorama.Fore.GREEN + f'All images have saved in the `{result_dir}`.')

        self.stitcher.stitcher.image = None


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
