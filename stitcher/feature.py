from stitcher.image import CCMImage

from lightglue import LightGlue
from lightglue.utils import rbd, Extractor

import colorama

from .globals import *


def extract(image: CCMImage, extractor: Extractor) -> None:
    """
    使用SuperPoint对图像提取特征，储存在CCMImage的feature属性中
    :param image: CCMImage
    :param extractor: SuperPoint实例
    :return: 无
    """
    image.features = extractor.extract(image.tensor.to(device))


def match(image1: CCMImage, image2: CCMImage, matcher: LightGlue) -> tuple[int, float]:
    """
    使用LightGlue对图像进行特征匹配，储存在CCMImage的matches属性中
    :param image1: 提取特征后的CCMImage
    :param image2: 提取特征后的CCMImage
    :param matcher: LightGlue实例
    :return: (匹配点对数量, 相似度)
    """
    matches01 = matcher({"image0": image1.features, "image1": image2.features})
    feats0, feats1, matches01 = [
        rbd(x) for x in [image1.features, image2.features, matches01]
    ]  # remove batch dimension
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    image1.matches = m_kpts0
    image2.matches = m_kpts1

    similarity = m_kpts1.shape[0] / kpts1.shape[0]
    return m_kpts1.shape[0], similarity



