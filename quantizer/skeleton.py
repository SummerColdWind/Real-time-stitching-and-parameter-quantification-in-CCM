from skimage.morphology import skeletonize
from scipy.ndimage import label
import networkx as nx
import cv2
import numpy as np

from .common import split

def get_skeleton(image, filter_threshold=64):
    image = image > 0
    skeleton = skeletonize(image)
    skeleton = skeleton.astype('uint8')
    skeleton = skeleton * 255

    # 设置边缘像素为0
    skeleton[0, :] = 0  # 上边缘
    skeleton[-1, :] = 0  # 下边缘
    skeleton[:, 0] = 0  # 左边缘
    skeleton[:, -1] = 0  # 右边缘

    # 过滤噪点
    segments, _ = split(skeleton, True)

    for segment in segments:
        if cv2.countNonZero(segment) < filter_threshold:
            skeleton[segment > 0] = 0

    # 滤除小分支
    skeleton = prune_skeleton(skeleton)
    return skeleton

def prune_skeleton(skeleton_image, min_length=10):
    """
    优化二值化骨架图像，滤除小的分支。

    参数:
    - skeleton_image (numpy.ndarray): 输入的二值化骨架图像（灰度图），骨架像素值为255，背景为0。
    - min_length (int): 要保留的最小分支长度。长度小于此阈值的分支将被删除。

    返回:
    - pruned_skeleton (numpy.ndarray): 优化后的二值化骨架图像（灰度图）。
    """

    # 确保输入是二值化的
    binary_skeleton = skeleton_image > 0

    # 构建图
    G = nx.Graph()
    ys, xs = np.where(binary_skeleton)
    for y, x in zip(ys, xs):
        G.add_node((y, x))

    # 定义8连通性
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    for y, x in zip(ys, xs):
        for dy, dx in directions:
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < binary_skeleton.shape[0] and 0 <= nx_ < binary_skeleton.shape[1]:
                if binary_skeleton[ny, nx_]:
                    G.add_edge((y, x), (ny, nx_))

    # 识别端点和分支点
    endpoints = [node for node, degree in G.degree() if degree == 1]
    branchpoints = [node for node, degree in G.degree() if degree > 2]

    pruned = binary_skeleton.copy()
    G_pruned = G.copy()

    for endpoint in endpoints:
        if endpoint not in G_pruned:
            continue
        path = [endpoint]
        current = endpoint
        prev = None
        while True:
            neighbors = list(G_pruned.neighbors(current))
            if prev is not None:
                neighbors = [n for n in neighbors if n != prev]
            if not neighbors:
                break
            next_node = neighbors[0]
            path.append(next_node)
            if next_node in branchpoints:
                break
            prev, current = current, next_node
        if len(path) < min_length:
            for node in path:
                pruned[node] = False
                G_pruned.remove_node(node)

    # 转换回灰度图
    pruned_skeleton = (pruned * 255).astype(np.uint8)

    return pruned_skeleton

