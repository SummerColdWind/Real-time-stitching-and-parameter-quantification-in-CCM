import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import gc
import weakref

from pathlib import Path
from numpy import ndarray
from PIL import Image

from lightglue.utils import numpy_image_to_torch

from .preprocess import vignetting_correction, match_histograms
from .globals import reference_image


def load_gray(path):
    with open(path, 'rb') as file:
        file_data = file.read()
        image_array = np.frombuffer(file_data, np.uint8)
        x = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)  # gray
        return x


class CCMImage(ndarray):
    """
        1.可以输入图片路径, ndarray, PIL.Image
        2.统一转换为opencv灰度图
        3.auto_preprocess指示是否进行暗角校正
    """
    def __new__(cls, input_, auto_preprocess=False):
        match input_:
            case str():
                if os.path.exists(input_):
                    x = load_gray(input_)
                else:
                    raise ValueError('Not a valid path!')
            case Path():
                x = load_gray(input_)
            case ndarray():
                x = input_
            case Image.Image():
                x = np.array(input_)
            case _:
                raise TypeError('Not a valid type!')

        return x.view(cls)

    def __init__(self, input_, auto_preprocess=False):
        self.input_ = input_
        self.tensor = numpy_image_to_torch(self)
        self.features = None
        self.matches = None
        if auto_preprocess:
            self.preprocess()


    def show(self):
        cv2.imshow('Show', self)
        cv2.moveWindow('Show', 400, 200)
        cv2.waitKey(0)

    def show_plt(self):
        if len(self.shape) == 3:
            plt.imshow(self)
        else:
            # plt.imshow(cv2.cvtColor(self, cv2.COLOR_GRAY2BGR))
            plt.imshow(self, cmap='gray', vmin=0, vmax=255)
        plt.show()

    def save(self, path):
        extend = Path(path).suffix
        retval, buffer = cv2.imencode(extend, self.astype('uint8'))

        with open(path, 'wb') as f:
            f.write(buffer)


    def preprocess(self):
        x = match_histograms(self, reference_image)
        x = vignetting_correction(x)
        self[:] = x
        self.tensor = numpy_image_to_torch(self)


