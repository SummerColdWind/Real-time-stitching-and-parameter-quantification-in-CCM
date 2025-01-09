from .stitch import Stitcher, CCMImage

import warnings
import os
import colorama

# 忽略所有警告
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

colorama.init(autoreset=True)
