from stitcher import Stitcher, CCMImage
from stitcher.globals import reference_image

import cv2
import colorama


class StitcherWrapper:
    def __init__(self, auto_preprocess=True, auto_gray=True, auto_scale=True, scale_shape=(384, 384)):
        self.stitcher = Stitcher(mode='affine', backend='aliked')
        self.auto_scale = auto_scale
        self.auto_gray = auto_gray
        self.scale_shape = scale_shape
        self.auto_preprocess = auto_preprocess

    def warm_up(self):
        print(colorama.Fore.BLUE + 'Start warming up ...')
        self.stitcher.add(CCMImage(reference_image))
        self.stitcher.add(CCMImage(reference_image))
        self.stitcher.image = None
        print(colorama.Fore.GREEN + 'All warmed up!')

    def process_frame(self, frame):
        if self.auto_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.auto_scale:
            frame = cv2.resize(frame, self.scale_shape)
        return CCMImage(frame, auto_preprocess=self.auto_preprocess)

    def __add__(self, frame):
        image = self.process_frame(frame)
        ret = self.stitcher + image
        return ret


