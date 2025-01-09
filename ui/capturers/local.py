from ui.utils.handle import capture_one, filter_handle_by_title_part, get_client_size
from .abstract import VideoStreamCapturer

import cv2

class LocalCapturer(VideoStreamCapturer):
    def __init__(self):
        self.handle = None
        self.frame = None

    def set_handle(self, title):
        try:
            handle = filter_handle_by_title_part(title)
            handle = list(handle)[0]
            self.handle = handle
            return True
        except Exception as e:
            print(e)
            return False

    def get_frame(self):
        if self.handle is None:
            return False
        _, _, w, h = get_client_size(self.handle)
        frame = capture_one(self.handle)
        frame = frame[:h, :w]
        self.frame = frame
        return frame

    def show_frame(self, wait=0):
        frame = self.get_frame()
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(wait)

