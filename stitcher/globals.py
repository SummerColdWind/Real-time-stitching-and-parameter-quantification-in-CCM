import torch
from PIL import Image
from pathlib import Path
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reference_image = np.array(Image.open(Path(__file__).parent / Path('./static/ref.png')))
# reference_image = np.array(Image.open(Path(__file__).parent / Path('./static/ref2.jpg')))


MIN_FEATURE_POINTS_COUNT = 25
RANSAC_THRESHOLD = 8.0
MAX_TPS_DISTANCE = 16.0
MAX_TPS_ERROR = 16.0
MAX_PERSPECTIVE_RATIO = 0.15
MIN_INCREASED_LENGTH = 32.0
MIN_INCREASED_SIZE = 100.0
EDGE_DILATE_SIZE = 31



