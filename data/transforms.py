# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random
from PIL import Image

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize the input image so that its longest side and shortest side are within a specified range,
    ensuring that both sides are divisible by a specified stride.

    Args:
        max_size (int): Maximum size for the longest edge of the image.
        min_size (int): Minimum size for the shortest edge of the image.
        stride (int): Value by which the height and width of the image must be divisible.
        max_pixels (int): Maximum pixels for the full image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR``, and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g., ``PIL.Image.BILINEAR`` are also accepted.
        antialias (bool, optional): Whether to apply antialiasing (default is True).
    """

    def __init__(
        self, 
        max_size: int, 
        min_size: int, 
        stride: int, 
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC, 
        antialias=True
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        """
        Args:
            img (PIL Image): Image to be resized.
            img_num (int): Number of images, used to change max_tokens.
        Returns:
            PIL Image or Tensor: Rescaled image with divisible dimensions.
        """
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Ensure the number of pixels does not exceed max_pixels
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        # Ensure longest edge does not exceed max_size
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class ImageTransform:
    def __init__(
        self, 
        max_image_size, 
        min_image_size, 
        image_stride, 
        max_pixels=14*14*9*1024,
        image_mean=[0.5, 0.5, 0.5], 
        image_std=[0.5, 0.5, 0.5],
        enable_augmentation=False,
        augmentation_prob=0.0,
        augmentation_ops=None,
        augmentation_strength=0.4,
        enable_spatial_shift=False,
        spatial_shift_prob=0.0,
        spatial_shift_max_ratio=0.1,
    ):
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size, 
            min_size=min_image_size, 
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std, inplace=True)
        self.augmentation_transform = MedicalGenerationAugmentation(
            enabled=enable_augmentation,
            prob=augmentation_prob,
            ops=augmentation_ops,
            strength=augmentation_strength,
        )
        self.spatial_shift_transform = MedicalSpatialShiftAugmentation(
            enabled=enable_spatial_shift,
            prob=spatial_shift_prob,
            max_shift_ratio=spatial_shift_max_ratio,
        )

    def sample_spatial_shift(self):
        return self.spatial_shift_transform.sample()

    def __call__(self, img, img_num=1, spatial_aug_params=None, apply_augmentation=True):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.spatial_shift_transform(img, spatial_aug_params)
        if apply_augmentation:
            img = self.augmentation_transform(img)
        img = self.normalize_transform(img)
        return img


class MedicalSpatialShiftAugmentation:
    def __init__(self, enabled=False, prob=0.0, max_shift_ratio=0.1):
        self.enabled = enabled
        self.prob = float(prob)
        self.max_shift_ratio = float(max(0.0, max_shift_ratio))

    def sample(self):
        if (not self.enabled) or self.prob <= 0.0 or self.max_shift_ratio <= 0.0:
            return {"apply": False, "dx_ratio": 0.0, "dy_ratio": 0.0}
        if random.random() >= self.prob:
            return {"apply": False, "dx_ratio": 0.0, "dy_ratio": 0.0}
        return {
            "apply": True,
            "dx_ratio": random.uniform(-self.max_shift_ratio, self.max_shift_ratio),
            "dy_ratio": random.uniform(-self.max_shift_ratio, self.max_shift_ratio),
        }

    def __call__(self, img, params=None):
        if params is None:
            params = self.sample()
        if not params or not params.get("apply", False):
            return img

        height, width = img.shape[-2:]
        dx = int(round(float(params.get("dx_ratio", 0.0)) * width))
        dy = int(round(float(params.get("dy_ratio", 0.0)) * height))
        if dx == 0 and dy == 0:
            return img

        return F.affine(
            img,
            angle=0.0,
            translate=[dx, dy],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )


class MedicalGenerationAugmentation:
    SUPPORTED_OPS = {
        "intensity",
        "gamma",
        "noise",
        "blur",
        "lowres",
        "bias",
    }

    def __init__(self, enabled=False, prob=0.0, ops=None, strength=0.4):
        self.enabled = enabled
        self.prob = float(prob)
        self.strength = float(np.clip(strength, 0.0, 1.0))
        if ops is None:
            ops = ["intensity", "gamma", "noise", "blur", "lowres", "bias"]
        elif isinstance(ops, str):
            ops = [item.strip() for item in ops.split(",") if item.strip()]
        else:
            ops = [str(item).strip() for item in ops if str(item).strip()]

        unsupported_ops = sorted(set(ops) - self.SUPPORTED_OPS)
        if unsupported_ops:
            raise ValueError(f"Unsupported augmentation ops: {unsupported_ops}")
        self.ops = ops

    def __call__(self, img):
        if (not self.enabled) or self.prob <= 0.0 or len(self.ops) == 0:
            return img
        if random.random() >= self.prob:
            return img

        max_ops = 1 if self.strength <= 0.4 else 2 if self.strength <= 0.75 else 3
        num_ops = min(len(self.ops), random.randint(1, max_ops))
        for op_name in random.sample(self.ops, k=num_ops):
            img = getattr(self, f"_apply_{op_name}")(img)
            img = self._sanitize(img)
        return img

    def _sanitize(self, img):
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        return img.clamp_(0.0, 1.0)

    def _apply_intensity(self, img):
        brightness = random.uniform(max(0.7, 1.0 - 0.35 * self.strength), 1.0 + 0.20 * self.strength)
        contrast = random.uniform(max(0.75, 1.0 - 0.40 * self.strength), 1.0 + 0.25 * self.strength)
        img = F.adjust_brightness(img, brightness)
        img = F.adjust_contrast(img, contrast)
        return img

    def _apply_gamma(self, img):
        gamma = random.uniform(max(0.7, 1.0 - 0.45 * self.strength), 1.0 + 0.45 * self.strength)
        img = img.clamp(0.0, 1.0)
        return F.adjust_gamma(img, gamma=gamma, gain=1.0)

    def _apply_noise(self, img):
        sigma = random.uniform(0.01, 0.01 + 0.07 * self.strength)
        return img + torch.randn_like(img) * sigma

    def _apply_blur(self, img):
        sigma = random.uniform(0.2, 0.4 + 1.8 * self.strength)
        kernel_size = 5 if min(img.shape[-2:]) >= 64 else 3
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=(sigma, sigma))(img)

    def _apply_lowres(self, img):
        _, height, width = img.shape
        scale = random.uniform(max(0.35, 1.0 - 0.65 * self.strength), 0.9)
        resized_h = max(8, int(round(height * scale)))
        resized_w = max(8, int(round(width * scale)))
        img = F.resize(
            img,
            size=[resized_h, resized_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return F.resize(
            img,
            size=[height, width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _apply_bias(self, img):
        _, height, width = img.shape
        field_h = max(2, height // 24)
        field_w = max(2, width // 24)
        field = torch.rand(1, field_h, field_w, dtype=img.dtype, device=img.device)
        field = F.resize(
            field,
            size=[height, width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        field = (field - field.mean()) / (field.std() + 1e-6)
        field = torch.exp(field * (0.05 + 0.25 * self.strength))
        return img * field


def decolorization(image):
    gray_image = image.convert('L')
    return Image.merge(image.mode, [gray_image] * 3) if image.mode in ('RGB', 'L') else gray_image


def downscale(image, scale_factor):
    new_width = int(round(image.width * scale_factor))
    new_height = int(round(image.height * scale_factor))
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def crop(image, crop_factors):
    target_h, target_w = crop_factors
    img_w, img_h = image.size

    if target_h > img_h or target_w > img_w:
        raise ValueError("Crop size exceeds image dimensions")

    x = random.randint(0, img_w - target_w)
    y = random.randint(0, img_h - target_h)

    return image.crop((x, y, x + target_w, y + target_h)), [[x, y], [x + target_w, y + target_h]]


def motion_blur_opencv(image, kernel_size=15, angle=0):
    # 线性核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = np.ones(kernel_size, dtype=np.float32)

    # 旋转核
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    # 归一化核
    rotated_kernel /= rotated_kernel.sum() if rotated_kernel.sum() != 0 else 1

    img = np.array(image)
    if img.ndim == 2:
        blurred = cv2.filter2D(img, -1, rotated_kernel, borderType=cv2.BORDER_REFLECT)
    else:
        # 对于彩色图像，各通道独立卷积
        blurred = np.zeros_like(img)
        for c in range(img.shape[2]):
            blurred[..., c] = cv2.filter2D(img[..., c], -1, rotated_kernel, borderType=cv2.BORDER_REFLECT)

    return Image.fromarray(blurred.astype(np.uint8))


def shuffle_patch(image, num_splits, gap_size=2):
    """将图像分割为块（允许尺寸不整除），随机打乱后拼接，块间保留间隙"""
    h_splits, w_splits = num_splits
    img_w, img_h = image.size

    base_patch_h = img_h // h_splits
    patch_heights = [base_patch_h] * (h_splits - 1)
    patch_heights.append(img_h - sum(patch_heights))

    base_patch_w = img_w // w_splits
    patch_widths = [base_patch_w] * (w_splits - 1)
    patch_widths.append(img_w - sum(patch_widths))

    patches = []
    current_y = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch_w = patch_widths[j]
            patch = image.crop((current_x, current_y, current_x + patch_w, current_y + patch_h))
            patches.append(patch)
            current_x += patch_w
        current_y += patch_h

    random.shuffle(patches)

    total_width = sum(patch_widths) + (w_splits - 1) * gap_size
    total_height = sum(patch_heights) + (h_splits - 1) * gap_size
    new_image = Image.new(image.mode, (total_width, total_height), color=(255, 255, 255))

    current_y = 0  # 当前行的起始 Y 坐标
    patch_idx = 0  # 当前处理的块索引
    for i in range(h_splits):
        current_x = 0  # 当前列的起始 X 坐标
        patch_h = patch_heights[i]  # 当前行块的高度
        for j in range(w_splits):
            # 取出打乱后的块
            patch = patches[patch_idx]
            patch_w = patch_widths[j]  # 当前列块的宽度
            # 粘贴块（左上角坐标为 (current_x, current_y)）
            new_image.paste(patch, (current_x, current_y))
            # 更新 X 坐标（下一个块的起始位置 = 当前块宽度 + 间隙）
            current_x += patch_w + gap_size
            patch_idx += 1
        # 更新 Y 坐标（下一行的起始位置 = 当前行高度 + 间隙）
        current_y += patch_h + gap_size

    return new_image


def inpainting(image, num_splits, blank_ratio=0.3, blank_color=(255, 255, 255)):
    """
    图像分割后随机空白部分patch，用于inpainting任务
    
    参数：
        image: PIL.Image 输入图像（RGB模式）
        h_splits: int 行分割数（垂直方向分割块数）
        w_splits: int 列分割数（水平方向分割块数）
        blank_ratio: float 空白patch的比例（0~1）
        blank_color: tuple 空白区域的颜色（RGB，如白色(255,255,255)）
    
    返回：
        PIL.Image 处理后拼接的图像
    """
    h_splits, w_splits = num_splits
    img_w, img_h = image.size

    base_patch_h = img_h // h_splits
    patch_heights = [base_patch_h] * (h_splits - 1)
    patch_heights.append(img_h - sum(patch_heights))

    base_patch_w = img_w // w_splits
    patch_widths = [base_patch_w] * (w_splits - 1)
    patch_widths.append(img_w - sum(patch_widths))

    patches = []
    current_y = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch_w = patch_widths[j]
            patch = image.crop((current_x, current_y, current_x + patch_w, current_y + patch_h))
            patches.append(patch)
            current_x += patch_w
        current_y += patch_h

    total_patches = h_splits * w_splits
    num_blank = int(total_patches * blank_ratio)
    num_blank = max(0, min(num_blank, total_patches))
    blank_indices = random.sample(range(total_patches), num_blank)

    processed_patches = []
    for idx, patch in enumerate(patches):
        if idx in blank_indices:
            blank_patch = Image.new("RGB", patch.size, color=blank_color)
            processed_patches.append(blank_patch)
        else:
            processed_patches.append(patch)

    # 创建结果图像（尺寸与原图一致）
    result_image = Image.new("RGB", (img_w, img_h))
    current_y = 0
    patch_idx = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            # 取出处理后的patch
            patch = processed_patches[patch_idx]
            patch_w = patch_widths[j]
            # 粘贴到原位置
            result_image.paste(patch, (current_x, current_y))
            current_x += patch_w
            patch_idx += 1
        current_y += patch_h

    return result_image
