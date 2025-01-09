from abc import ABC, abstractmethod


class VideoStreamCapturer(ABC):
    @abstractmethod
    def get_frame(self):
        """ 获取最新一帧画面 """
