import random
import numpy as np
from PIL import Image
import cv2
import io
import torch
from torchvision.transforms import v2 as T
from torchvision import transforms
from typing import Dict, Any

from src.config import Augmentations


class VideoCompressionArtifacts:
    """Simulate video compression artifacts using JPEG with lower quality"""
    def __init__(self, quality_range=(20, 50)):
        self.quality_range = quality_range
    
    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        # Convert PIL to bytes and back to simulate compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class Downscale:
    """Downscale then upscale to original size"""
    def __init__(self, scale_range=(0.5, 0.9), interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation
    
    def __call__(self, img):
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Downscale
        img_down = img.resize((new_w, new_h), self.interpolation)
        # Upscale back
        img_up = img_down.resize((w, h), self.interpolation)
        return img_up


class ISONoise:
    """Add ISO noise (camera noise) to simulate low-light conditions"""
    def __init__(self, color_shift=0.05, intensity_range=(0.1, 0.5)):
        self.color_shift = color_shift
        self.intensity_range = intensity_range
    
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32) / 255.0
        intensity = random.uniform(*self.intensity_range)
        
        # Add gaussian noise
        noise = np.random.randn(*img_array.shape) * intensity
        
        # Add color shift
        color_noise = np.random.randn(3) * self.color_shift
        
        noisy = img_array + noise + color_noise
        noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)


class MultiplicativeNoise:
    """Multiplicative noise"""
    def __init__(self, multiplier_range=(0.9, 1.1)):
        self.multiplier_range = multiplier_range
    
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32)
        multiplier = random.uniform(*self.multiplier_range)
        
        noisy = img_array * multiplier
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)


class MotionBlur:
    """Apply motion blur"""
    def __init__(self, kernel_size_range=(3, 7)):
        self.kernel_size_range = kernel_size_range
    
    def __call__(self, img):
        img_array = np.array(img)
        kernel_size = random.randrange(*self.kernel_size_range, 2)  # Odd numbers only
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply blur
        blurred = cv2.filter2D(img_array, -1, kernel)
        
        return Image.fromarray(blurred)


class HSVAugmentation:
    """HSV color space augmentation"""
    def __init__(self, hue_shift=20, sat_shift=30, val_shift=20):
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
    
    def __call__(self, img):
        img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Random shifts
        h_shift = random.randint(-self.hue_shift, self.hue_shift)
        s_shift = random.randint(-self.sat_shift, self.sat_shift)
        v_shift = random.randint(-self.val_shift, self.val_shift)
        
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + h_shift) % 180
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + s_shift, 0, 255)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + v_shift, 0, 255)
        
        img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(img_rgb)


class CoarseDropout:
    """Randomly drop rectangular regions"""
    def __init__(self, max_holes=8, max_height=32, max_width=32):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
    
    def __call__(self, img):
        img_array = np.array(img).copy()
        h, w = img_array.shape[:2]
        
        num_holes = random.randint(1, self.max_holes)
        
        for _ in range(num_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)
            
            hole_h = random.randint(1, self.max_height)
            hole_w = random.randint(1, self.max_width)
            
            y1 = max(0, y - hole_h // 2)
            y2 = min(h, y + hole_h // 2)
            x1 = max(0, x - hole_w // 2)
            x2 = min(w, x + hole_w // 2)
            
            img_array[y1:y2, x1:x2] = 0
        
        return Image.fromarray(img_array)


class GridMask:
    """Grid mask augmentation"""
    def __init__(self, ratio=0.6):
        self.ratio = ratio
    
    def __call__(self, img):
        img_array = np.array(img).copy()
        h, w = img_array.shape[:2]
        
        # Random grid size
        grid_size = random.randint(20, 50)
        
        # Create grid mask
        mask = np.ones((h, w), dtype=np.uint8)
        
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                if random.random() < self.ratio:
                    mask[i:i+grid_size//2, j:j+grid_size//2] = 0
        
        # Apply mask
        for c in range(3):
            img_array[:, :, c] = img_array[:, :, c] * mask
        
        return Image.fromarray(img_array)


# ===== Phase 6: Frequency Domain Augmentations =====

class DCTDropout:
    """DCT (Discrete Cosine Transform) Dropout - Drop frequency components"""
    def __init__(self, channels=[0, 1, 2], dropout_ratio_range=(0.1, 0.3)):
        self.channels = channels
        self.dropout_ratio_range = dropout_ratio_range
    
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32)
        h, w, c = img_array.shape
        
        dropout_ratio = random.uniform(*self.dropout_ratio_range)
        
        # Apply DCT to selected channels
        for ch in self.channels:
            if ch >= c:
                continue
            
            # Apply DCT
            dct = cv2.dct(img_array[:, :, ch])
            
            # Create mask to dropout random DCT coefficients
            mask = np.random.rand(h, w) > dropout_ratio
            dct_dropped = dct * mask
            
            # Inverse DCT
            img_array[:, :, ch] = cv2.idct(dct_dropped)
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


class FrequencyFilter:
    """Apply high-pass or low-pass filter in frequency domain"""
    def __init__(self, filter_type='high', cutoff_range=(0.1, 0.3)):
        self.filter_type = filter_type  # 'high' or 'low'
        self.cutoff_range = cutoff_range
    
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        cutoff = random.uniform(*self.cutoff_range)
        
        # Process each channel
        for c in range(img_array.shape[2]):
            # FFT
            f = np.fft.fft2(img_array[:, :, c])
            fshift = np.fft.fftshift(f)
            
            # Create mask
            crow, ccol = h // 2, w // 2
            mask = np.zeros((h, w), dtype=np.float32)
            
            radius = int(min(crow, ccol) * cutoff)
            y, x = np.ogrid[:h, :w]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
            
            if self.filter_type == 'low':
                # Low-pass: keep low frequencies
                mask[mask_area] = 1
            else:
                # High-pass: keep high frequencies
                mask = 1 - mask
                mask[mask_area] = 0
            
            # Apply mask and inverse FFT
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_filtered = np.fft.ifft2(f_ishift)
            img_filtered = np.abs(img_filtered)
            
            img_array[:, :, c] = img_filtered
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


# ===== Phase 7: Geometric & Advanced Color Augmentations =====

class ElasticTransform:
    """Elastic deformation of images"""
    def __init__(self, alpha_range=(30, 50), sigma_range=(4, 6)):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
    
    def __call__(self, img):
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        alpha = random.uniform(*self.alpha_range)
        sigma = random.uniform(*self.sigma_range)
        
        # Random displacement fields
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap
        distorted = cv2.remap(img_array, map_x, map_y, 
                             interpolation=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(distorted)


class RandomGamma:
    """Random gamma correction"""
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.gamma_range = gamma_range
    
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32) / 255.0
        gamma = random.uniform(*self.gamma_range)
        
        # Apply gamma correction
        corrected = np.power(img_array, gamma)
        corrected = (corrected * 255).astype(np.uint8)
        
        return Image.fromarray(corrected)


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size) if isinstance(tile_grid_size, list) else tile_grid_size
    
    def __call__(self, img):
        img_array = np.array(img)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                tileGridSize=self.tile_grid_size)
        l_clahe = clahe.apply(l)
        
        # Merge channels
        lab_clahe = cv2.merge([l_clahe, a, b])
        rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(rgb)


class Posterize:
    """Reduce the number of bits for each color channel"""
    def __init__(self, bits_range=(4, 7)):
        self.bits_range = bits_range
    
    def __call__(self, img):
        img_array = np.array(img)
        bits = random.randint(*self.bits_range)
        
        # Reduce bits
        shift = 8 - bits
        posterized = (img_array >> shift) << shift
        
        return Image.fromarray(posterized)


# ===== Phase 8: Self-Blended Images (SBI) & Advanced =====

class SelfBlendedImages:
    """Self-Blended Images (SBI) - CVPR 2022
    Blend different parts of the same image to create realistic boundaries
    Reference: https://arxiv.org/abs/2204.06974
    """
    def __init__(self, alpha_range=(0.3, 0.7), mask_type='random'):
        self.alpha_range = alpha_range
        self.mask_type = mask_type
    
    def create_mask(self, h, w):
        """Create blending mask based on mask_type"""
        if self.mask_type == 'half':
            # Vertical or horizontal split
            mask = np.zeros((h, w), dtype=np.float32)
            if random.random() < 0.5:
                # Vertical split
                split = random.randint(w // 4, 3 * w // 4)
                mask[:, :split] = 1.0
            else:
                # Horizontal split
                split = random.randint(h // 4, 3 * h // 4)
                mask[:split, :] = 1.0
        
        elif self.mask_type == 'quarter':
            # Quadrant blending
            mask = np.zeros((h, w), dtype=np.float32)
            h_mid, w_mid = h // 2, w // 2
            quadrant = random.randint(0, 3)
            if quadrant == 0:
                mask[:h_mid, :w_mid] = 1.0
            elif quadrant == 1:
                mask[:h_mid, w_mid:] = 1.0
            elif quadrant == 2:
                mask[h_mid:, :w_mid] = 1.0
            else:
                mask[h_mid:, w_mid:] = 1.0
        
        elif self.mask_type == 'grid':
            # Grid pattern
            mask = np.zeros((h, w), dtype=np.float32)
            grid_size = random.randint(20, 60)
            for i in range(0, h, grid_size * 2):
                for j in range(0, w, grid_size * 2):
                    mask[i:i+grid_size, j:j+grid_size] = 1.0
        
        else:  # 'random'
            # Random blob mask
            mask = np.random.rand(h // 4, w // 4).astype(np.float32)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.float32)
        
        # Smooth mask edges
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        return mask
    
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Create mask
        mask = self.create_mask(h, w)
        mask = np.expand_dims(mask, axis=2)  # (H, W, 1)
        
        # Create flipped/transformed version of the same image
        transform_type = random.choice(['flip', 'rotate', 'shift'])
        
        if transform_type == 'flip':
            img_transformed = np.fliplr(img_array)
        elif transform_type == 'rotate':
            angle = random.choice([90, 180, 270])
            img_transformed = np.array(Image.fromarray(img_array.astype(np.uint8)).rotate(angle))
        else:  # shift
            shift_x = random.randint(-w // 4, w // 4)
            shift_y = random.randint(-h // 4, h // 4)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img_transformed = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Blend with random alpha
        alpha = random.uniform(*self.alpha_range)
        blended = mask * img_array + (1 - mask) * img_transformed
        blended = blended * alpha + img_array * (1 - alpha)
        
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)


class PerspectiveTransform:
    """Random perspective transformation"""
    def __init__(self, distortion=0.2):
        self.distortion = distortion
    
    def __call__(self, img):
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Random perspective points
        d = int(min(h, w) * self.distortion)
        
        src_points = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])
        
        dst_points = np.float32([
            [random.randint(0, d), random.randint(0, d)],
            [w - 1 - random.randint(0, d), random.randint(0, d)],
            [w - 1 - random.randint(0, d), h - 1 - random.randint(0, d)],
            [random.randint(0, d), h - 1 - random.randint(0, d)]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img_array, matrix, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(warped)


class Sharpen:
    """Image sharpening"""
    def __init__(self, alpha_range=(0.2, 0.5)):
        self.alpha_range = alpha_range
    
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32)
        
        # Sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        sharpened = cv2.filter2D(img_array, -1, kernel)
        
        # Blend with original
        alpha = random.uniform(*self.alpha_range)
        result = alpha * sharpened + (1 - alpha) * img_array
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)


class RandomShadow:
    """Add random shadows to image"""
    def __init__(self, intensity_range=(0.3, 0.7)):
        self.intensity_range = intensity_range
    
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Random shadow polygon
        num_points = random.randint(3, 6)
        points = []
        for _ in range(num_points):
            points.append([random.randint(0, w), random.randint(0, h)])
        points = np.array(points, dtype=np.int32)
        
        # Create shadow mask
        mask = np.ones((h, w), dtype=np.float32)
        cv2.fillPoly(mask, [points], 0.0)
        
        # Smooth shadow edges
        mask = cv2.GaussianBlur(mask, (51, 51), 30)
        
        # Apply shadow
        intensity = random.uniform(*self.intensity_range)
        shadow_strength = 1.0 - intensity
        
        for c in range(3):
            img_array[:, :, c] = img_array[:, :, c] * (mask + (1 - mask) * shadow_strength)
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


        return Image.fromarray(img_array)


def init_augmentations(augs: Augmentations):
    # TODO: for each augmentation, add a probability parameter to the config
    if augs is None:
        return None

    composed_transforms = []

    # === Geometric Augmentations ===
    if augs.random_horizontal_flip != 0.0:
        composed_transforms.append(T.RandomHorizontalFlip(p=augs.random_horizontal_flip))

    if (
        augs.random_affine_degrees != 0
        or augs.random_affine_translate is not None
        or augs.random_affine_scale is not None
    ):
        composed_transforms.append(
            T.RandomAffine(
                degrees=augs.random_affine_degrees,
                translate=augs.random_affine_translate,
                scale=augs.random_affine_scale,
            )
        )

    # === Blur Augmentations ===
    if augs.gaussian_blur_prob != 0.0:
        ks = augs.gaussian_blur_kernel_size
        if (isinstance(ks, int) and ks != 0) or (isinstance(ks, list) and sum(ks) != 0):
            composed_transforms.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=ks, sigma=augs.gaussian_blur_sigma)],
                    p=augs.gaussian_blur_prob,
                )
            )
    
    if augs.motion_blur_prob != 0.0:
        composed_transforms.append(
            T.RandomApply(
                [MotionBlur(kernel_size_range=augs.motion_blur_kernel_size if isinstance(augs.motion_blur_kernel_size, list) else (3, augs.motion_blur_kernel_size))],
                p=augs.motion_blur_prob,
            )
        )

    # === Color Augmentations ===
    if (augs.color_jitter_brightness != 0.0 or augs.color_jitter_contrast != 0.0 or 
        augs.color_jitter_saturation != 0.0 or augs.color_jitter_hue != 0.0):
        composed_transforms.append(
            T.ColorJitter(
                brightness=augs.color_jitter_brightness,
                contrast=augs.color_jitter_contrast,
                saturation=augs.color_jitter_saturation,
                hue=augs.color_jitter_hue,
            )
        )
    
    if augs.hsv_prob != 0.0:
        composed_transforms.append(
            T.RandomApply(
                [HSVAugmentation(augs.hsv_hue_shift, augs.hsv_sat_shift, augs.hsv_val_shift)],
                p=augs.hsv_prob,
            )
        )

    # === Compression Augmentations (Phase 1) ===
    if (isinstance(augs.jpeg_quality, int) and augs.jpeg_quality != 100) or (
        isinstance(augs.jpeg_quality, list) and augs.jpeg_quality[0] != 100
    ):
        composed_transforms.append(T.JPEG(augs.jpeg_quality))
    
    if augs.video_compression_prob != 0.0:
        composed_transforms.append(
            T.RandomApply(
                [VideoCompressionArtifacts(quality_range=augs.video_compression_quality if isinstance(augs.video_compression_quality, list) else (20, augs.video_compression_quality))],
                p=augs.video_compression_prob,
            )
        )

    # === Downscale Augmentations (Phase 1) ===
    if augs.downscale_prob != 0.0:
        interp_mode = augs.downscale_interpolation
        if isinstance(interp_mode, int):
            if interp_mode == 0:
                interp = Image.NEAREST
            elif interp_mode == 2:
                interp = Image.BILINEAR
            elif interp_mode == 3:
                interp = Image.BICUBIC
            else:
                interp = Image.BILINEAR
        else:
            interp = Image.BILINEAR
        
        scale_range = augs.downscale_scale if isinstance(augs.downscale_scale, list) else (0.5, augs.downscale_scale)
        composed_transforms.append(
            T.RandomApply(
                [Downscale(scale_range=scale_range, interpolation=interp)],
                p=augs.downscale_prob,
            )
        )

    # === Noise Augmentations (Phase 2) ===
    if augs.gaussian_noise_sigma != 0.0:
        composed_transforms.append(
            T.Compose(
                [
                    T.ToTensor(),
                    T.GaussianNoise(0.0, augs.gaussian_noise_sigma),
                    T.ToPILImage(),
                ]
            )
        )
    
    if augs.iso_noise_prob != 0.0:
        intensity_range = augs.iso_noise_intensity if isinstance(augs.iso_noise_intensity, list) else (0.1, augs.iso_noise_intensity)
        composed_transforms.append(
            T.RandomApply(
                [ISONoise(color_shift=augs.iso_noise_color_shift, intensity_range=intensity_range)],
                p=augs.iso_noise_prob,
            )
        )
    
    if augs.multiplicative_noise_prob != 0.0:
        mult_range = augs.multiplicative_noise_multiplier if isinstance(augs.multiplicative_noise_multiplier, list) else (0.9, augs.multiplicative_noise_multiplier)
        composed_transforms.append(
            T.RandomApply(
                [MultiplicativeNoise(multiplier_range=mult_range)],
                p=augs.multiplicative_noise_prob,
            )
        )

    # === Dropout Augmentations (Phase 4) ===
    if augs.coarse_dropout_prob != 0.0:
        composed_transforms.append(
            T.RandomApply(
                [CoarseDropout(max_holes=augs.coarse_dropout_max_holes, 
                              max_height=augs.coarse_dropout_max_height,
                              max_width=augs.coarse_dropout_max_width)],
                p=augs.coarse_dropout_prob,
            )
        )
    
    if augs.grid_mask_prob != 0.0:
        composed_transforms.append(
            T.RandomApply(
                [GridMask(ratio=augs.grid_mask_ratio)],
                p=augs.grid_mask_prob,
            )
        )

    # === Phase 6: Frequency Domain Augmentations ===
    if augs.dct_dropout_prob != 0.0:
        channels = augs.dct_dropout_channels if isinstance(augs.dct_dropout_channels, list) else [0, 1, 2]
        dropout_ratio = augs.dct_dropout_ratio if isinstance(augs.dct_dropout_ratio, list) else (0.1, augs.dct_dropout_ratio)
        composed_transforms.append(
            T.RandomApply(
                [DCTDropout(channels=channels, dropout_ratio_range=dropout_ratio)],
                p=augs.dct_dropout_prob,
            )
        )
    
    if augs.high_pass_prob != 0.0:
        cutoff = augs.high_pass_cutoff if isinstance(augs.high_pass_cutoff, list) else (0.1, augs.high_pass_cutoff)
        composed_transforms.append(
            T.RandomApply(
                [FrequencyFilter(filter_type='high', cutoff_range=cutoff)],
                p=augs.high_pass_prob,
            )
        )
    
    if augs.low_pass_prob != 0.0:
        cutoff = augs.low_pass_cutoff if isinstance(augs.low_pass_cutoff, list) else (0.5, augs.low_pass_cutoff)
        composed_transforms.append(
            T.RandomApply(
                [FrequencyFilter(filter_type='low', cutoff_range=cutoff)],
                p=augs.low_pass_prob,
            )
        )
    
    # === Phase 7: Geometric Transform ===
    if augs.elastic_prob != 0.0:
        alpha = augs.elastic_alpha if isinstance(augs.elastic_alpha, list) else (30, augs.elastic_alpha)
        sigma = augs.elastic_sigma if isinstance(augs.elastic_sigma, list) else (4, augs.elastic_sigma)
        composed_transforms.append(
            T.RandomApply(
                [ElasticTransform(alpha_range=alpha, sigma_range=sigma)],
                p=augs.elastic_prob,
            )
        )
    
    # === Phase 7: Advanced Color Augmentations ===
    if augs.gamma_prob != 0.0:
        gamma_limit = augs.gamma_limit if isinstance(augs.gamma_limit, list) else (0.7, augs.gamma_limit)
        composed_transforms.append(
            T.RandomApply(
                [RandomGamma(gamma_range=gamma_limit)],
                p=augs.gamma_prob,
            )
        )
    
    if augs.clahe_prob != 0.0:
        tile_grid = augs.clahe_tile_grid_size if isinstance(augs.clahe_tile_grid_size, list) else (8, 8)
        composed_transforms.append(
            T.RandomApply(
                [CLAHE(clip_limit=augs.clahe_clip_limit, tile_grid_size=tile_grid)],
                p=augs.clahe_prob,
            )
        )
    
    if augs.posterize_prob != 0.0:
        bits = augs.posterize_bits if isinstance(augs.posterize_bits, list) else (4, augs.posterize_bits)
        composed_transforms.append(
            T.RandomApply(
                [Posterize(bits_range=bits)],
                p=augs.posterize_prob,
            )
        )

    # === Phase 8: Self-Blended Images (SBI) ===
    if augs.sbi_prob != 0.0:
        alpha_range = augs.sbi_alpha_range if isinstance(augs.sbi_alpha_range, list) else (0.3, augs.sbi_alpha_range)
        composed_transforms.append(
            T.RandomApply(
                [SelfBlendedImages(alpha_range=alpha_range, mask_type=augs.sbi_mask_type)],
                p=augs.sbi_prob,
            )
        )
    
    # === Phase 8: Advanced Geometric ===
    if augs.perspective_prob != 0.0:
        composed_transforms.append(
            T.RandomApply(
                [PerspectiveTransform(distortion=augs.perspective_distortion)],
                p=augs.perspective_prob,
            )
        )
    
    # === Phase 8: Advanced Lighting ===
    if augs.sharpen_prob != 0.0:
        sharpen_alpha = augs.sharpen_alpha if isinstance(augs.sharpen_alpha, list) else (0.2, augs.sharpen_alpha)
        composed_transforms.append(
            T.RandomApply(
                [Sharpen(alpha_range=sharpen_alpha)],
                p=augs.sharpen_prob,
            )
        )
    
    if augs.random_shadow_prob != 0.0:
        shadow_intensity = augs.random_shadow_intensity if isinstance(augs.random_shadow_intensity, list) else (0.3, augs.random_shadow_intensity)
        composed_transforms.append(
            T.RandomApply(
                [RandomShadow(intensity_range=shadow_intensity)],
                p=augs.random_shadow_prob,
            )
        )

    # === Advanced Augmentations ===
    if augs.auto_augment_prob != 0.0:
        composed_transforms.append(
            T.RandomApply(
                [transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)],
                p=augs.auto_augment_prob,
            )
        )

    # === Resize (should be last) ===
    if augs.resize is not None:
        composed_transforms.append(T.Resize(augs.resize, augs.resize_interpolation))

    if len(composed_transforms) == 0:
        return None

    return T.Compose(composed_transforms)


# ============================================================================
# CutMix Implementation (Batch-level augmentation)
# ============================================================================

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix
    
    Args:
        size: (H, W) of the image
        lam: lambda value from beta distribution
    
    Returns:
        Bounding box coordinates (bbx1, bby1, bbx2, bby2)
    """
    H, W = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # Uniform sampling of center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix augmentation to a batch
    
    Args:
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        alpha: Beta distribution parameter
    
    Returns:
        Mixed images, labels_a, labels_b, lambda value
    """
    batch_size = images.size(0)
    
    # Generate random lambda from beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Generate random permutation of batch indices
    rand_index = torch.randperm(batch_size).to(images.device)
    
    # Get labels for mixed samples
    labels_a = labels
    labels_b = labels[rand_index]
    
    # Generate bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size()[2:], lam)
    
    # Apply CutMix
    images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match actual area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    
    return images, labels_a, labels_b, lam


def cutmix_collate_fn(batch, cutmix_prob=0.5, cutmix_alpha=1.0):
    """Custom collate function with CutMix augmentation
    
    Args:
        batch: List of samples from dataset
        cutmix_prob: Probability of applying CutMix
        cutmix_alpha: Beta distribution alpha parameter
    
    Returns:
        Dictionary with images, labels, and CutMix info
    """
    # Default collate
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    paths = [item['path'] for item in batch]
    idx = torch.tensor([item['idx'] for item in batch])
    
    # Apply CutMix with probability
    if random.random() < cutmix_prob:
        images, labels_a, labels_b, lam = cutmix_batch(images, labels, cutmix_alpha)
        
        return {
            'image': images,
            'label': labels_a,  # Keep as primary label
            'label_b': labels_b,  # Secondary label for mixed region
            'lam': lam,  # Lambda for weighted loss
            'cutmix_applied': True,
            'path': paths,
            'idx': idx,
        }
    else:
        return {
            'image': images,
            'label': labels,
            'label_b': labels,  # Same as primary
            'lam': 1.0,  # No mixing
            'cutmix_applied': False,
            'path': paths,
            'idx': idx,
        }


def create_cutmix_collate_fn(cutmix_prob=0.5, cutmix_alpha=1.0):
    """Create a collate function with fixed CutMix parameters
    
    Args:
        cutmix_prob: Probability of applying CutMix
        cutmix_alpha: Beta distribution alpha parameter
    
    Returns:
        Collate function
    """
    def collate_fn(batch):
        return cutmix_collate_fn(batch, cutmix_prob, cutmix_alpha)
    
    return collate_fn


# ============================================================================
# Real-Fake Mixup Implementation (DFDC 1st Place Strategy)
# ============================================================================

def mixup_images(img_a: torch.Tensor, img_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Mix two images with alpha blending
    
    Args:
        img_a: First image (C, H, W) or (B, C, H, W)
        img_b: Second image (C, H, W) or (B, C, H, W)
        lam: Mixing coefficient [0, 1]
    
    Returns:
        Mixed image = lam * img_a + (1 - lam) * img_b
    """
    return lam * img_a + (1.0 - lam) * img_b


def real_fake_mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0, 
                          paired_indices: torch.Tensor = None):
    """Apply Real-Fake Mixup: mix real and fake samples
    
    This is the key strategy used by DFDC 1st place team.
    Two modes:
    1. Paired mode (if paired_indices provided): Mix same video's real/fake pairs
    2. Random mode (default): Mix random real/fake samples in batch
    
    Args:
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,) where 0=real, 1=fake
        alpha: Beta distribution parameter for lambda
        paired_indices: Optional (B,) tensor with pair indices for paired mixup
    
    Returns:
        Mixed images, labels_a, labels_b, lambda value
    """
    batch_size = images.size(0)
    device = images.device
    
    # Separate real and fake indices
    real_mask = (labels == 0)
    fake_mask = (labels == 1)
    
    real_indices = torch.where(real_mask)[0]
    fake_indices = torch.where(fake_mask)[0]
    
    # If we don't have both real and fake samples, skip mixup
    if len(real_indices) == 0 or len(fake_indices) == 0:
        return images, labels, labels, 1.0
    
    # Generate random lambda from beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # For each sample, decide whether to mix
    mixed_images = images.clone()
    labels_a = labels.clone()
    labels_b = labels.clone()
    
    # === Paired Mixup Mode (DFDC Original Strategy) ===
    if paired_indices is not None:
        # Build pair mapping: idx -> pair_idx
        pair_map = {}
        for i, pair_idx in enumerate(paired_indices):
            pair_map[i] = pair_idx.item()
        
        # Mix real samples with their paired fake samples
        for real_idx in real_indices:
            pair_idx = pair_map.get(real_idx.item())
            if pair_idx is not None and pair_idx < batch_size:
                # Check if pair is fake
                if labels[pair_idx] == 1:
                    mixed_images[real_idx] = mixup_images(
                        images[real_idx], 
                        images[pair_idx], 
                        lam
                    )
                    labels_a[real_idx] = 0  # real
                    labels_b[real_idx] = 1  # fake (from pair)
        
        # Mix fake samples with their paired real samples
        for fake_idx in fake_indices:
            pair_idx = pair_map.get(fake_idx.item())
            if pair_idx is not None and pair_idx < batch_size:
                # Check if pair is real
                if labels[pair_idx] == 0:
                    mixed_images[fake_idx] = mixup_images(
                        images[fake_idx],
                        images[pair_idx],
                        lam
                    )
                    labels_a[fake_idx] = 1  # fake
                    labels_b[fake_idx] = 0  # real (from pair)
    
    # === Random Batch Mixup Mode ===
    else:
        # Mix real samples with random fake samples
        for real_idx in real_indices:
            # Randomly select a fake sample
            fake_idx = fake_indices[torch.randint(len(fake_indices), (1,)).item()]
            
            # Mix: real + fake
            mixed_images[real_idx] = mixup_images(
                images[real_idx], 
                images[fake_idx], 
                lam
            )
            labels_a[real_idx] = 0  # real
            labels_b[real_idx] = 1  # fake
        
        # Mix fake samples with random real samples
        for fake_idx in fake_indices:
            # Randomly select a real sample
            real_idx = real_indices[torch.randint(len(real_indices), (1,)).item()]
            
            # Mix: fake + real
            mixed_images[fake_idx] = mixup_images(
                images[fake_idx],
                images[real_idx],
                lam
            )
            labels_a[fake_idx] = 1  # fake
            labels_b[fake_idx] = 0  # real
    
    return mixed_images, labels_a, labels_b, lam


def real_fake_mixup_collate_fn(batch, mixup_prob=0.5, mixup_alpha=1.0, use_pairs=False):
    """Custom collate function with Real-Fake Mixup augmentation
    
    Supports two modes:
    1. Paired mode: Mix same video's real/fake pairs (DFDC original)
    2. Random mode: Mix random real/fake from batch
    
    Args:
        batch: List of samples from dataset
        mixup_prob: Probability of applying Real-Fake Mixup
        mixup_alpha: Beta distribution alpha parameter
        use_pairs: If True, use paired mixup (requires pair info in batch)
    
    Returns:
        Dictionary with images, labels, and Mixup info
    """
    # Default collate
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    paths = [item['path'] for item in batch]
    idx = torch.tensor([item['idx'] for item in batch])
    
    # Get pair indices if available (for paired mixup)
    paired_indices = None
    if use_pairs and 'pair_idx' in batch[0]:
        paired_indices = torch.tensor([item.get('pair_idx', -1) for item in batch])
    
    # Apply Real-Fake Mixup with probability
    if random.random() < mixup_prob:
        images, labels_a, labels_b, lam = real_fake_mixup_batch(
            images, labels, mixup_alpha, paired_indices
        )
        
        return {
            'image': images,
            'label': labels_a,  # Primary label
            'label_b': labels_b,  # Secondary label (opposite class)
            'lam': lam,  # Lambda for weighted loss
            'mixup_applied': True,
            'path': paths,
            'idx': idx,
        }
    else:
        return {
            'image': images,
            'label': labels,
            'label_b': labels,  # Same as primary
            'lam': 1.0,  # No mixing
            'mixup_applied': False,
            'path': paths,
            'idx': idx,
        }


def create_real_fake_mixup_collate_fn(mixup_prob=0.5, mixup_alpha=1.0, use_pairs=False):
    """Create a collate function with fixed Real-Fake Mixup parameters
    
    Args:
        mixup_prob: Probability of applying Real-Fake Mixup
        mixup_alpha: Beta distribution alpha parameter
        use_pairs: If True, use paired mixup (DFDC original strategy)
    
    Returns:
        Collate function
    """
    def collate_fn(batch):
        return real_fake_mixup_collate_fn(batch, mixup_prob, mixup_alpha, use_pairs)
    
    return collate_fn
