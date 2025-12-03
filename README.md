# Nucleus Segmentation with Classical Computer Vision

Primitive (no-deep-learning) nucleus segmentation on MoNuSeg histopathology slides using stain handling, thresholding, morphology, and watershed. This repo is intended as a lightweight baseline before moving to heavier models.

## Datasets
- MoNuSeg (2018 challenge): https://www.kaggle.com/datasets/upsidedown97/monuseg  
- MoNuSeg 2018 (mirror): https://www.kaggle.com/datasets/tuanledinh/monuseg2018

Place the downloaded data under `data/` (or adjust paths in your scripts). Example layout:
```
data/
  MoNuSeg/
    img/*.png
    masks/*.png   # instance or binary masks
```


