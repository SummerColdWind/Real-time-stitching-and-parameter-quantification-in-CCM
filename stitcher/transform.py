import numpy as np
import cv2
import thinplate

import colorama

from .image import CCMImage
from .globals import *


def affine_ransac(image1: CCMImage, image2: CCMImage, *, rtn_inliers_num=False) -> np.ndarray:
    """ 仿射变换的单应性矩阵ransac估计 """
    src_points = image1.matches.cpu().numpy()
    dst_points = image2.matches.cpu().numpy()

    H, mask = cv2.findHomography(
        src_points, dst_points, cv2.RHO, ransacReprojThreshold=RANSAC_THRESHOLD, maxIters=1000)

    # print(f"Number of inliers: {np.sum(mask)} / {len(mask)}")
    if rtn_inliers_num:
        return H, np.sum(mask)
    return H


def translation_ransac(image1: CCMImage, image2: CCMImage, threshold=RANSAC_THRESHOLD, max_iterations=1000):
    """ 平移变换的单应性矩阵ransac估计 """
    pts1 = image1.matches.cpu().numpy()
    pts2 = image2.matches.cpu().numpy()

    best_tx, best_ty = 0, 0
    max_inliers = 0

    for _ in range(max_iterations):
        # 随机选择一个匹配点
        idx = np.random.randint(0, pts1.shape[0])
        tx_candidate = pts2[idx, 0] - pts1[idx, 0]
        ty_candidate = pts2[idx, 1] - pts1[idx, 1]

        # 计算所有点的平移误差
        translated_pts1 = pts1 + np.array([tx_candidate, ty_candidate])
        errors = np.linalg.norm(translated_pts1 - pts2, axis=1)

        # 找到内点
        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_tx, best_ty = tx_candidate, ty_candidate

    # 创建平移变换矩阵 H (3x3)
    H = np.array([[1, 0, best_tx],
                  [0, 1, best_ty],
                  [0, 0, 1]])

    return H


def _tps_transform(img, src_points, dst_points):
    """
    使用薄板样条变换 (TPS) 对图像进行变换。

    参数:
    - img (numpy.ndarray): 输入的灰度图像 (H, W)。
    - src_points (numpy.ndarray): 源坐标，形状为 (N, 2)，即源点坐标。
    - dst_points (numpy.ndarray): 目标坐标，形状为 (N, 2)，即变换后的目标点坐标。

    返回:
    - warped_img (numpy.ndarray): 变换后的图像。
    """
    # 确保源坐标和目标坐标都已归一化到 [0, 1] 区间
    src_points_normalized = src_points / np.array([img.shape[1], img.shape[0]])
    dst_points_normalized = dst_points / np.array([img.shape[1], img.shape[0]])

    # 计算TPS的theta参数
    theta = thinplate.tps_theta_from_points(src_points_normalized, dst_points_normalized, reduced=True)

    # 使用TPS变换生成网格
    dshape = img.shape[:2]  # 图像的高度和宽度
    grid = thinplate.tps_grid(theta, dst_points_normalized, dshape)

    # 计算变换后的坐标
    mapx, mapy = thinplate.tps_grid_to_remap(grid, img.shape)

    # 执行图像变换
    warped_img = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_CUBIC)

    return warped_img

def tps_warp(img_src: CCMImage, img_add: CCMImage, H):
    # 获取匹配的源点和目标点
    try:
        src_points = img_src.matches.cpu().numpy()  # 源图像的匹配点
        dst_points = img_add.matches.cpu().numpy()  # 目标图像的匹配点
    except Exception:
        src_points = img_src.matches
        dst_points = img_add.matches

    # 将源点转换为齐次坐标
    src_homogeneous = np.hstack([src_points, np.ones((src_points.shape[0], 1))])

    # 应用单应性矩阵 H
    projected_points_homogeneous = (H @ src_homogeneous.T).T

    # 转换回笛卡尔坐标
    projected_points = projected_points_homogeneous[:, :2] / projected_points_homogeneous[:, 2, np.newaxis]

    # 计算欧氏距离
    errors = np.linalg.norm(dst_points - projected_points, axis=1)

    # 筛选出距离小于等于 16 的点对
    valid_indices = errors <= MAX_TPS_DISTANCE
    filtered_src_points = src_points[valid_indices]
    filtered_dst_points = dst_points[valid_indices]
    filtered_projected_points = projected_points[valid_indices]


    # 应用 TPS 变换
    if len(filtered_src_points) > 0:
        canvas = img_add.copy()
        canvas = _tps_transform(canvas, filtered_dst_points, filtered_projected_points)
    else:
        print("No valid points for TPS transform")
        return img_add, 0

    return canvas, np.max(errors)
