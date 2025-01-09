from pygrabber.dshow_graph import FilterGraph
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np

def get_camera_devices():
    """ 获取摄像头列表 """
    graph = FilterGraph()
    devices = graph.get_input_devices()
    devices = list(devices)
    return devices

def load_image(path):
    """ 修复了opencv不能读取中文路径的bug """
    with open(path, 'rb') as file:
        file_data = file.read()
        image_array = np.frombuffer(file_data, np.uint8)
        image_raw = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image_raw

def save_image(image, path='result.png'):
    """ 保存一张图片 """
    extend = '.' + path.split('.')[-1]
    retval, buffer = cv2.imencode(extend, image.astype('uint8'))
    with open(path, 'wb') as f:
        f.write(buffer)

def show_image(image):
    """ 展示一张图片 """
    image_show = image.copy().astype('uint8')
    if np.amax(image_show) == 1:
        image_show = image_show * 255
    cv2.imshow('Show', image_show)
    cv2.waitKey(0)


def convert_cv_to_pixmap(cv_img):
    """将OpenCV格式的图像转换为QPixmap"""
    # 获取图像尺寸和通道信息
    if len(cv_img.shape) == 3:
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        # 将BGR格式转换为RGB格式
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # 将numpy数组转换为QImage
        q_image = QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        # 将QImage转换为QPixmap
        return QPixmap.fromImage(q_image)
    else:
        height, width = cv_img.shape
        bytes_per_line = width
        # 将numpy数组转换为QImage
        q_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        # 将QImage转换为QPixmap
        return QPixmap.fromImage(q_image)

def resize_image_with_padding(image, target_width, target_height):
    """调整图像大小并添加填充以适应目标尺寸，保持原始长宽比例"""
    # 获取原图的宽度和高度
    h, w = image.shape[:2]

    # 计算缩放比例，保持原图比例
    scale = min(target_width / w, target_height / h)

    # 按照比例缩放原图
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # 创建黑色背景
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    # 使用cv2.copyMakeBorder进行填充
    if len(image.shape) == 2:
        # 灰度图
        color = 0
    else:
        # 彩色图
        color = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_image
