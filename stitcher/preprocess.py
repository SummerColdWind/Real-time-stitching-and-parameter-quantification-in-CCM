import numpy as np
import cv2
import torch
import math


def vignetting_correction(img, a=0.0625, b=1.75, c=-0.75, device='cuda'):
    """
    使用 PyTorch 对图像进行晕影矫正，使用固定参数 (a, b, c)。

    参数:
        img: 源图像 (BGR 或 灰度图) 的 NumPy 数组
        a, b, c: 校正参数
        device: 'cuda' 或 'cpu'

    返回:
        矫正后的灰度图像的 NumPy 数组
    """
    # 转换为灰度图（在 CPU 上执行）
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 将灰度图转换为浮点型并创建 PyTorch 张量
    gray_tensor = torch.from_numpy(gray).float().to(device)

    # 获取图像尺寸
    h, w = gray.shape
    cx, cy = w / 2, h / 2

    # 创建坐标网格
    y = torch.arange(h, device=device).float().view(h, 1).repeat(1, w)
    x = torch.arange(w, device=device).float().view(1, w).repeat(h, 1)

    # 计算归一化距离矩阵
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    norm = math.sqrt(cx ** 2 + cy ** 2)  # 使用 math.sqrt 计算标量
    r_normalized = r / norm

    # 计算增益
    gain = 1 + a * r_normalized ** 2 + b * r_normalized ** 4 + c * r_normalized ** 6

    # 应用增益
    corrected_tensor = gray_tensor * gain

    # 限制范围并转换为 uint8
    corrected_tensor = torch.clamp(corrected_tensor, 0, 255).byte()

    # 将结果从 GPU 传回 CPU（如果使用 GPU）
    corrected = corrected_tensor.cpu().numpy()

    return corrected


def match_histograms(image, reference_image, device='cuda'):
    """
    使用 PyTorch 将输入图像的直方图匹配到参考图像的直方图。

    参数：
        image: 源图像 (BGR 或 灰度图) 的 NumPy 数组
        reference_image: 参考图像 (BGR 或 灰度图) 的 NumPy 数组
        device: 'cuda' 或 'cpu'

    返回：
        matched_image: 匹配后的灰度图像的 NumPy 数组
    """
    # 转换为灰度图（在 CPU 上执行）
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()

    if len(reference_image.shape) == 3:
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference_image.copy()

    # 将图像转换为 uint8 并创建 PyTorch 张量
    image_tensor = torch.from_numpy(image_gray).to(device)
    reference_tensor = torch.from_numpy(reference_gray).to(device)

    # 计算源图像和参考图像的直方图
    src_hist = torch.histc(image_tensor.float(), bins=256, min=0, max=255)
    ref_hist = torch.histc(reference_tensor.float(), bins=256, min=0, max=255)

    # 归一化直方图
    src_hist = src_hist / src_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()

    # 计算累积分布函数 (CDF)
    src_cdf = torch.cumsum(src_hist, dim=0)
    ref_cdf = torch.cumsum(ref_hist, dim=0)

    # 将 CDF 转换为相同的数据类型以避免计算问题
    src_cdf = src_cdf.float()
    ref_cdf = ref_cdf.float()

    # 使用 torch.searchsorted 找到映射关系
    # 需要确保 ref_cdf 是升序的
    mapping = torch.searchsorted(ref_cdf, src_cdf)

    # 处理可能的边界情况
    mapping = torch.clamp(mapping, 0, 255).byte()

    # 将映射表应用到源图像
    matched_image_tensor = mapping[image_tensor.long()]

    # 将结果从 GPU 传回 CPU（如果使用 GPU）
    matched_image = matched_image_tensor.cpu().numpy()

    return matched_image


if __name__ == '__main__':
    import timeit

    image = cv2.imread('static/ref.png', 0)


    def single():
        match_histograms(image, image)
        vignetting_correction(image)


    single()  # warm-up
    code = "single()"
    times = timeit.repeat(stmt=code, number=1000, repeat=5, setup="from __main__ import single")
    print(f"每次执行时间: {times} 秒")
    print(f"最短执行时间: {min(times)} 秒")
