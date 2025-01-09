import natsort
import colorama
import tqdm
from typing import Literal

from lightglue import LightGlue, SuperPoint, SIFT, DISK, ALIKED, DoGHardNet

from .image import CCMImage, Path
from .feature import extract, match
from .transform import affine_ransac, translation_ransac
from .registration import rough_registration, fine_registration
from .globals import *

STITCH_MODE = Literal['translation', 'affine', 'tps']
BACKEND = Literal['superpoint', 'disk', 'aliked', 'sift', 'doghardnet']


class Stitcher:
    def __init__(self, backend: BACKEND = 'superpoint', mode: STITCH_MODE = 'translation'):
        self.backend = backend
        extractor = {
            'superpoint': SuperPoint,
            'disk': DISK,
            'aliked': ALIKED,
            'sift': SIFT,
            'doghardnet': DoGHardNet,
        }
        self.extractor = extractor[backend](max_num_keypoints=2048).eval().to(device)
        self.matcher = LightGlue(features=backend).eval().to(device)

        self.mode = mode

        self.image: CCMImage | None = None
        self.previous: list[CCMImage] = []
        self.count = 0

    def init(self, img_src: CCMImage):
        self.image = img_src
        self.previous = []
        self.count = 0

    def _stitch(self, img_add: CCMImage):
        # 特征提取和匹配
        extract(self.image, self.extractor)
        extract(img_add, self.extractor)
        cnt_features, similarity = match(self.image, img_add, self.matcher)
        print(colorama.Fore.GREEN + f'特征点匹配数量: {cnt_features}')
        if cnt_features < MIN_FEATURE_POINTS_COUNT:
            raise RuntimeError('匹配到的特征点对过少')

        match self.mode:
            case 'translation':
                H = translation_ransac(self.image, img_add)
                img_new = rough_registration(self.image, img_add, H)
            case 'affine':
                H = affine_ransac(self.image, img_add)
                img_new = rough_registration(self.image, img_add, H)
            case 'tps':
                H = affine_ransac(self.image, img_add)
                img_new = fine_registration(self.image, img_add, H)
            case _:
                raise TypeError('Wrong mode!')

        # 提取透视变换矩阵的各项参数
        x1, y1, x2, y2, x3, y3 = H[0, 0] - 1, H[1, 1] - 1, H[0, 1], H[1, 0], H[2, 0], H[2, 1]
        ransac_params = (x1, y1, x2, y2)
        # 打印并检查每个参数值
        param_names = ['X缩放', 'Y缩放', 'X旋转', 'Y旋转']
        for param_name, value in zip(param_names, ransac_params):
            rounded_value = round(value, 3)
            print(colorama.Fore.GREEN + f"{param_name}: {rounded_value}")

            # 检查透视变换值是否超出限制
            if abs(value) > MAX_PERSPECTIVE_RATIO:
                raise RuntimeError(f'过强的透视变换: {param_name}{rounded_value}')

        return img_new

    def add(self, img_add: CCMImage):
        if self.image is None:
            self.init(img_add)
            return True
        try:
            previous = self.image.copy()
            self.image = self._stitch(img_add)
            self.previous.append(previous)
            self.count += 1
            print(colorama.Back.GREEN + 'A successful stitching was performed.')
            return True
        except Exception as e:
            print(colorama.Fore.RED + str(e))
            print(colorama.Back.RED + 'Therefore, this stitching was ignored.')
            return False

    def __add__(self, other: CCMImage):
        return self.add(other)

    def iter_dir(self, img_dir: Path | str, auto_preprocess=False):
        if isinstance(img_dir, str):
            img_dir = Path(img_dir)

        images = natsort.natsorted(img_dir.iterdir())
        for item in images:
            img_add = CCMImage(item, auto_preprocess=auto_preprocess)
            if self.image is None:
                self.init(img_add)
            else:
                flag = self.add(img_add)

    def undo(self):
        if self.previous:
            previous = self.previous.pop()
            self.image = CCMImage(previous.copy())


    def show(self):
        self.image.show()

    def show_plt(self):
        self.image.show_plt()
