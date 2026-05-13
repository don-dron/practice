from __future__ import annotations

import random

import torchvision.transforms.functional as TF
from PIL import Image


class TrainAugmentation:
    def __init__(self, jitter_brightness: tuple[float, float] = (-0.2, 0.2), jitter_contrast: tuple[float, float] = (0.85, 1.15), p_flip: float = 0.0):
        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast
        self.p_flip = p_flip

    def __call__(self, img: Image.Image) -> Image.Image:
        img = TF.rgb_to_grayscale(img, num_output_channels=1)
        if random.random() < self.p_flip:
            img = TF.hflip(img)
        b = random.uniform(*self.jitter_brightness)
        c = random.uniform(*self.jitter_contrast)
        img = TF.adjust_brightness(img, 1.0 + b)
        img = TF.adjust_contrast(img, c)
        return img
