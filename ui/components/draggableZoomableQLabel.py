import sys
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow

from ui.utils.common import convert_cv_to_pixmap
from ui.capturers import CameraCapturer


class DraggableZoomableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        # 缩放因子
        self.scale_factor = 1.0

        # 拖拽相关标志
        self.dragging = False
        self.start_pos = QPoint()

    def set_image(self, frame):
        image = convert_cv_to_pixmap(frame)
        self.image = image


        # 根据新的缩放因子生成新的 QPixmap
        new_size = self.image.size() * self.scale_factor
        scaled_pixmap = self.image.scaled(
            new_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
        # 调整 QLabel 自身大小，以便在父控件中更好地显示
        self.resize(scaled_pixmap.size())

        self.setPixmap(scaled_pixmap)
        self.setScaledContents(True)  # 允许缩放内容自动填满 QLabel
        # 固定大小，可根据需求调整
        self.resize(scaled_pixmap.size())

    def mousePressEvent(self, event):
        """鼠标按下时，记录初始位置，用于拖拽。"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动时，根据偏移量移动控件本身，实现拖拽效果。"""
        if self.dragging:
            # 计算移动偏移量
            delta = event.pos() - self.start_pos
            # 移动 QLabel
            self.move(self.x() + delta.x(), self.y() + delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放时，停止拖拽。"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """滚轮事件，用于放大 / 缩小 QLabel 显示的图像。"""
        # 获取滚轮滚动方向
        angle_delta = event.angleDelta().y()

        if angle_delta > 0:
            # 放大
            self.scale_factor *= 1.1
        else:
            # 缩小
            self.scale_factor *= 0.9

        # 为防止过度缩放，限制缩放因子范围
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))


