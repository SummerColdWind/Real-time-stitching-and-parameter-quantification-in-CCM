from ui.utils.common import get_camera_devices
from .abstract import VideoStreamCapturer

import cv2

class CameraCapturer(VideoStreamCapturer):
    def __init__(self):
        self.camera = None
        self.frame = None

    def set_camera(self, name):
        devices = get_camera_devices()
        try:
            if self.camera is not None:
                self.camera.relese()
            idx = devices.index(name)
            self.camera = cv2.VideoCapture(idx)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            return True
        except Exception as e:
            print(e)
            return False


    def get_frame(self):
        if self.camera is None:
            return False
        ret, frame = self.camera.read()
        if not ret:
            return False
        self.frame = frame
        return frame

    def show_frame(self, wait=0):
        frame = self.get_frame()
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(wait)




