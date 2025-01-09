from stitcher.image import CCMImage, Path

from .transform import tps_warp

import numpy as np
import cv2


def _pre_registration(img_src: CCMImage, H: np.ndarray):
    h1, w1 = img_src.shape[:2]
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_img1, np.linalg.inv(H))  # 使用 H 的逆矩阵
    all_corners = np.concatenate((corners_img1, transformed_corners), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    # 平移变换矩阵
    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])

    # 使用 H 的逆矩阵
    H_translation = H_translation @ np.linalg.inv(H)
    return H_translation, (xmin, ymin, xmax, ymax)

def _post_registration(img_src_wrap, img_add_warp):
    # 填充黑色部分
    mask = ((img_add_warp != 0) * 255).astype('uint8')
    # 腐蚀掉黑边
    mask = cv2.erode(mask, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    stitched_image = cv2.bitwise_and(img_src_wrap, img_src_wrap, mask=cv2.bitwise_not(mask))
    stitched_image = cv2.add(stitched_image, cv2.bitwise_and(img_add_warp, img_add_warp, mask=mask))

    # 裁掉黑边
    mask = (stitched_image == 0)
    binary_mask = 1 - mask.astype(np.uint8)
    # 找到非黑色区域的边界（最外层矩形框）
    coords = cv2.findNonZero(binary_mask)  # 找到所有非零（即非黑色）的像素位置
    x, y, w, h = cv2.boundingRect(coords)  # 计算最小外接矩形

    # 使用这个矩形框裁剪图像
    stitched_image = stitched_image[y:y + h, x:x + w]
    return CCMImage(stitched_image)


def rough_registration(img_src: CCMImage, img_add: CCMImage, H: np.ndarray) -> np.ndarray:
    """ 粗配准 """
    H_translation, (xmin, ymin, xmax, ymax) = _pre_registration(img_src, H)
    dsize = (xmax - xmin, ymax - ymin)
    dsize = abs(dsize[0] * dsize[1])
    print(dsize)
    if dsize > 1e8:
        raise RuntimeError('内存开销过大!')
    h1, w1 = img_src.shape[:2]
    # 投影图像2到新的范围
    img_add_warp = cv2.warpPerspective(img_add, H_translation, (xmax - xmin, ymax - ymin))

    canvas = np.zeros_like(img_add_warp)
    canvas[-ymin:-ymin + h1, -xmin:-xmin + w1] = img_src

    return _post_registration(canvas, img_add_warp)

def fine_registration(img_src: CCMImage, img_add: CCMImage, H: np.ndarray) -> np.ndarray:
    """ 精细配准 """
    H_translation, (xmin, ymin, xmax, ymax) = _pre_registration(img_src, H)
    h1, w1 = img_src.shape[:2]
    # 投影图像2到新的范围
    img_add_warp = cv2.warpPerspective(img_add, H_translation, (xmax - xmin, ymax - ymin))

    canvas = np.zeros_like(img_add_warp)
    img_src, max_tps_error = tps_warp(img_add, img_src, np.linalg.inv(H))
    canvas[-ymin:-ymin + h1, -xmin:-xmin + w1] = img_src

    return _post_registration(canvas, img_add_warp)
