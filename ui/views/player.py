import sys
from PyQt6.QtCore import Qt, QPoint, QRect, QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QCommandLinkButton

from ui.utils.common import convert_cv_to_pixmap, get_camera_devices
from ui.capturers import CameraCapturer
from ui.components import DraggableZoomableLabel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CameraPlayer")

        self.devices_buttons = []
        self.create_devices_buttons()

        self.camera = CameraCapturer()
        self.device = None
        self.label = None

        self.timer = QTimer(self)  # 创建 QTimer
        self.timer.timeout.connect(self.update_frame)  # 绑定定时更新事件

        self.resize(400, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

    def update_frame(self):
        """更新 QLabel 上的摄像头画面"""
        if self.camera is not None:
            frame = self.camera.get_frame()
            if frame is not None:
                self.label.set_image(frame)  # 显示最新画面

    def create_devices_buttons(self):
        devices = get_camera_devices()
        y = 20
        for device in devices:
            btn = QCommandLinkButton(device, self)
            btn.setGeometry(QRect(20, y, 300, 40))  # 设定按钮的绝对位置
            btn.clicked.connect(lambda checked, t=device: self.on_device_button_clicked(t))  # 绑定点击事件
            self.devices_buttons.append(btn)
            y += 50  # 让下一个按钮往下移动 50 像素

    def on_device_button_clicked(self, device):
        """点击按钮后，移除所有按钮并创建新的 QLabel"""
        # 移除所有按钮
        for btn in self.devices_buttons:
            btn.hide()  # 隐藏按钮
            btn.deleteLater()  # 删除按钮对象

        self.devices_buttons.clear()  # 清空按钮列表

        self.device = device
        self.camera.set_camera(device)

        # 创建 QLabel 显示选中的文本
        self.label = DraggableZoomableLabel(self)
        self.label.set_image(self.camera.get_frame())
        self.label.show()

        # 循环QTimer
        self.timer.start(33)  # 30fps


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
