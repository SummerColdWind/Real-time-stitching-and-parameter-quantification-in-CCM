from stitcher import CCMImage, Stitcher
from quantizer import Quantizer

from pathlib import Path

s = Stitcher('aliked', 'affine')
q = Quantizer()

for item in Path('assets/dataset0').iterdir():
    image = CCMImage(item, auto_preprocess=True)
    s + image

s.show_plt()
rst = q.quantize(s.image)
print(rst)
