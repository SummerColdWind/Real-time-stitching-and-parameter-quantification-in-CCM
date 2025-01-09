import cv2
import time
import numpy as np
import win32gui, win32con, win32ui, win32api
from ctypes import windll

windll.user32.SetProcessDPIAware()

def get_client_size(handle):
    """获取窗口的客户区尺寸（不包括标题栏和边框）"""
    rect = win32gui.GetClientRect(handle)
    width, height = rect[2] - rect[0], rect[3] - rect[1]

    # 获取窗口左上角坐标（物理像素）
    point = win32gui.ClientToScreen(handle, (0, 0))
    x, y = point[0], point[1]

    return x, y, width, height


def capture_one(handle):
    """通过win32方式截图，并且无视硬件加速"""
    rect = win32gui.GetWindowRect(handle)
    width, height = rect[2] - rect[0], rect[3] - rect[1]
    hwnd_dc = win32gui.GetWindowDC(handle)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    save_bit_map = win32ui.CreateBitmap()
    save_bit_map.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(save_bit_map)
    windll.user32.PrintWindow(handle, save_dc.GetSafeHdc(), 3)
    bmpinfo = save_bit_map.GetInfo()
    bmpstr = save_bit_map.GetBitmapBits(True)
    image = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
    image = np.ascontiguousarray(image)[..., :-1]
    win32gui.DeleteObject(save_bit_map.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(handle, hwnd_dc)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def get_all_handles():
    """获取当前所有窗口的句柄"""
    parent_hwnd_list = []
    win32gui.EnumWindows(lambda hwnd, param: param.append(hwnd), parent_hwnd_list)
    return parent_hwnd_list


def get_child_handles(parent_handle):
    """获取指定父窗口句柄的所有子窗口句柄"""
    child_hwnd_list = []
    win32gui.EnumChildWindows(parent_handle, lambda hwnd, param: param.append(hwnd), child_hwnd_list)
    return child_hwnd_list

def get_title(handle):
    """获取窗口的标题"""
    return win32gui.GetWindowText(handle)

def get_cls(handle):
    """获取窗口的类名"""
    return win32gui.GetClassName(handle)


def filter_handle_by_title_part(title_part, parent=None):
    """根据部分窗口标题过滤句柄"""
    if parent is None:
        return set(h for h in get_all_handles() if title_part in get_title(h))
    return set(h for h in get_child_handles(parent) if title_part in get_title(h))

def filter_handle_by_cls_part(cls_part, parent=None):
    """根据部分窗口类名过滤句柄"""
    if parent is None:
        return set(h for h in get_all_handles() if cls_part in get_cls(h))
    return set(h for h in get_child_handles(parent) if cls_part in get_cls(h))

