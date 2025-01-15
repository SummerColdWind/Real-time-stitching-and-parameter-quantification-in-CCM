# A deep learning tool for real-time stitching and parameter quantification in corneal confocal microscopy

### Brief Introduction
> This is a tool developed in pure Python that has the following features:
> 1. It can stitch together multiple corneal confocal microscope (CCM) images into a single wide-field image.
> 2. It can be carried out in real time during the subject's CCM examination.
> 3. Binarize the spliced image, extract the center line and key points, and calculate quantitative parameters.
---

## Environmental preparation

This is our conda environment, which you can reproduce with the following command:
```shell
conda env create -f environment.yml
```
More concise environment requirements will be provided in the future.

## How to use?
### 1. For datasets that have already been acquired

`assets/dataset0` provides 28 high-quality, continuously captured CCM images, and we provide a sample code based on this.

> Origin data source: https://github.com/LiTianYu6/NerveStitcher

See the usage in `example.py`.

#### (1) First, import the required classes.

```python
from stitcher import CCMImage, Stitcher
from quantizer import Quantizer

from pathlib import Path
```
CCMImage inherits from numpy's ndarray, adding functionality dedicated to this task analysis. 

Stitcher is used to stitch multiple images. 

Quantizer is used to quantify the stitched results.

#### (2) Instantiate these processors.

```python
s = Stitcher('aliked', 'affine')
q = Quantizer()
```
Stitcher's first argument specifies the feature extractor to be used, which can be `['superpoint', 'disk', 'aliked', 'sift', 'doghardnet']`. We recommend using `aliked`.

The second parameter indicates the method of concatenation, which can be `['translation', 'affine', 'tps']`. `translation` only carries out translation transformation, `affine` carries out affine transformation, `tps` will add non-rigid transformation on the basis of affine transformation.

#### (3) Incremental stitching

```python
for item in Path('assets/dataset0').iterdir():
    image = CCMImage(item, auto_preprocess=True)
    s + image
```

`CCMImage` can accept a path and automatically read the image. The `auto_preprocess` parameter indicates whether preprocessing is performed. For raw images, we recommend setting it to `True`.

We overload the `add` operator so that it can operate directly with the `CCMImage` instance, which is very intuitive.

Display the results using `matplotlib`.
```python
s.show_plt()
```
The expected output is:

---

<img src="assets/img.png">

#### (4) Parameter quantization

The resulting image can be processed using the `Quantizer` `quantize` method.

```python
rst = q.quantize(s.image)
print(rst)
```
The expected output is:
`
{'cnfl': 19.338700720950847, 'cnbd': 217.70390413606492, 'area': 0.40421875}
`

### 2. Real-time stitching

1) Make sure the device is receiving a video signal from the CCM, such as a USB camera.
2) Launch `ui/views/player.py` and select the corresponding video stream name.
<img src="assets/img_1.png">
3) Then, launch `ui/views/main.py` and click the `Stitch` button.
