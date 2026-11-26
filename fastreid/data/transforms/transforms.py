# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

__all__ = ['ToTensor', 'RandomPatch', 'AugMix', 'RandomIllumination', 'AutoGamma', ]

import math
import random
from collections import deque

import numpy as np
import torch

from .functional import to_tensor, augmentations


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1, prob_flip_leftright=0.5,
                 ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = torch.flip(patch, dims=[2])
        return patch

    def __call__(self, img):
        _, H, W = img.size()  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img[..., y1:y1 + h, x1:x1 + w]
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        _, patchH, patchW = patch.size()
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img[..., y1:y1 + patchH, x1:x1 + patchW] = patch

        return img


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    """

    def __init__(self, prob=0.5, aug_prob_coeff=0.1, mixture_width=3, mixture_depth=1, aug_severity=1):
        """
        Args:
            prob: Probability of taking augmix
            aug_prob_coeff: Probability distribution coefficients.
            mixture_width: Number of augmentation chains to mix per augmented example.
            mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
            aug_severity: Severity of underlying augmentation operators (between 1 to 10).
        """
        # fmt: off
        self.prob           = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width  = mixture_width
        self.mixture_depth  = mixture_depth
        self.aug_severity   = aug_severity
        self.augmentations  = augmentations
        # fmt: on

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.

        Returns:
          mixed: Augmented and mixed image.
        """
        if random.random() > self.prob:
            # Avoid the warning: the given NumPy array is not writeable
            return np.asarray(image).copy()

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = np.zeros([image.size[1], image.size[0], 3])
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * image + m * mix
        return mixed.astype(np.uint8)


def _luminance(arr):
    """Mean luminance in [0, 1] of an HxWx3 float array in [0, 1]."""
    return float((0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).mean())


class RandomIllumination(object):
    """L1 low-light augmentation (training only). Bright images (mean luminance above
    bright_thresh) are darkened with probability `prob`: gamma > 1, reduced contrast and
    sensor-like Gaussian noise, imitating CARGO night frames (also desaturated: night frames are grey-green). Dark images (below
    dark_thresh) are brightened with gamma < 1. Uses only the image's own pixels."""

    def __init__(self, prob=0.5, dark_gamma=(2.0, 4.0), bright_gamma=(0.4, 0.75),
                 contrast=(0.3, 0.7), noise=0.02, dark_thresh=0.30, bright_thresh=0.38,
                 saturation=(0.3, 0.8)):
        self.prob, self.dark_gamma, self.bright_gamma = prob, dark_gamma, bright_gamma
        self.contrast, self.noise = contrast, noise
        self.dark_thresh, self.bright_thresh = dark_thresh, bright_thresh
        self.saturation = saturation

    def __call__(self, img):
        if random.random() >= self.prob:
            return img
        from PIL import Image
        x = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.
        m = _luminance(x)
        if m > self.bright_thresh:
            x = x ** random.uniform(*self.dark_gamma)
            c = random.uniform(*self.contrast)
            mu = x.mean()
            x = mu + c * (x - mu)
            gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
            k = random.uniform(*self.saturation)
            x = k * x + (1 - k) * gray
            if self.noise > 0:
                x = x + np.random.normal(0, random.uniform(0, self.noise), x.shape).astype(np.float32)
        elif m < self.dark_thresh:
            x = x ** random.uniform(*self.bright_gamma)
        else:
            return img
        return Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


class AutoGamma(object):
    """L2 deterministic brightness normalisation (training and testing). Images whose mean
    luminance m is below `target` get gamma = log(target)/log(m), clipped to
    [min_gamma, 1]; brighter images are unchanged. Uses only the image's own pixels."""

    def __init__(self, target=0.45, min_gamma=0.4):
        self.target, self.min_gamma = target, min_gamma

    def __call__(self, img):
        from PIL import Image
        x = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.
        m = min(max(_luminance(x), 1e-3), 0.999)
        if m >= self.target:
            return img
        g = max(self.min_gamma, math.log(self.target) / math.log(m))
        return Image.fromarray((np.clip(x ** g, 0, 1) * 255 + 0.5).astype(np.uint8))

    def __repr__(self):
        return f'{self.__class__.__name__}(target={self.target}, min_gamma={self.min_gamma})'
