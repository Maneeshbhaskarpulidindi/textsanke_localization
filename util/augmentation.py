import numpy as np
import math
import cv2
import numpy.random as random


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


class RandomMirror(object):
    def __init__(self):
        pass

    def __call__(self, image, polygons=None):
        if np.random.randint(2):
            image = np.ascontiguousarray(image[:, ::-1])
            _, width, _ = image.shape
            for polygon in polygons:
                polygon.points[:, 0] = width - polygon.points[:, 0]
        return image, polygons


class AugmentColor(object):
    def __init__(self):
        self.U = np.array([[-0.56543481, 0.71983482, 0.40240142],
                      [-0.5989477, -0.02304967, -0.80036049],
                      [-0.56694071, -0.6935729, 0.44423429]], dtype=np.float32)
        self.EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)
        self.sigma = 0.1
        self.color_vec = None

    def __call__(self, img, polygons=None):
        color_vec = self.color_vec
        if self.color_vec is None:
            if not self.sigma > 0.0:
                color_vec = np.zeros(3, dtype=np.float32)
            else:
                color_vec = np.random.normal(0.0, self.sigma, 3)

        alpha = color_vec.astype(np.float32) * self.EV
        noise = np.dot(self.U, alpha.T) * 255
        return np.clip(img + noise[np.newaxis, np.newaxis, :], 0, 255), polygons


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, polygons=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return np.clip(image, 0, 255), polygons


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return np.clip(image, 0, 255), polygons


class Rotate(object):
    def __init__(self, up=30): ## up=30 default
        self.up = up

    def rotate(self, center, pt, theta):  # Rotation of 2D graphics

        xr, yr = center
        yr = -yr
        x, y = pt[:, 0], pt[:, 1]
        y = -y

        theta = theta / 360 * 2 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        _x = xr + (x - xr) * cos - (y - yr) * sin
        _y = yr + (x - xr) * sin + (y - yr) * cos

        return _x, -_y

    def __call__(self, img, polygons=None):
        if np.random.randint(2):
            return img, polygons
        angle = np.random.uniform(-self.up, self.up)  #uniformaly select random angle from -up to +up
        rows, cols = img.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])
        center = cols / 2.0, rows / 2.0
        if polygons is not None:
            for polygon in polygons:
                x, y = self.rotate(center, polygon.points, angle)
                pts = np.vstack([x, y]).T
                polygon.points = pts
        return img, polygons

class SquarePadding(object):

    def __call__(self, image, pts=None):

        H, W, _ = image.shape

        if H == W:
            return image, pts

        padding_size = max(H, W)
        expand_image = np.zeros((padding_size, padding_size, 3), dtype=image.dtype)

        if H > W:
            y0, x0 = 0, (H - W) // 2
        else:
            y0, x0 = (W - H) // 2, 0
        if pts is not None:
            pts[:, 0] += x0
            pts[:, 1] += y0

        expand_image[y0:y0+H, x0:x0+W] = image
        image = expand_image

        return image, pts

class Padding(object):

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image, polygons=None):
        if np.random.randint(2):
            return image, polygons

        height, width, depth = image.shape
        ratio = np.random.uniform(1.0, 3.0)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
          (int(height * ratio), int(width * ratio), depth),
          dtype=image.dtype)
        expand_image[:, :, :] = self.fill
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        if polygons is not None:
            for polygon in polygons:
                polygon.points[:, 0] = polygon.points[:, 0] + left
                polygon.points[:, 1] = polygon.points[:, 1] + top
        return image, polygons


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, pts=None):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        cropped = image[i:i + h, j:j + w, :]
        pts = pts.copy()
        mask = (pts[:, 1] >= i) * (pts[:, 0] >= j) * (pts[:, 1] < (i+h)) * (pts[:, 0] < (j+w))
        pts[~mask, 2] = -1
        scales = np.array([self.size[0]/w, self.size[1]/h])
        pts[:, :2] -= np.array([j, i])
        pts[:, :2] = (pts[:, :2] * scales)
        img = cv2.resize(cropped, self.size)
        return img, pts


class RandomResizedLimitCrop(object):
    def __init__(self, size, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, polygons=None):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)

        cropped = image[i:i + h, j:j + w, :]
        scales = np.array([self.size[0] / w, self.size[1] / h])
        if polygons is not None:
            for polygon in polygons:
                polygon.points[:, 0] = (polygon.points[:, 0] - j) * scales[0]
                polygon.points[:, 1] = (polygon.points[:, 1] - i) * scales[1]

        img = cv2.resize(cropped, self.size)
        return img, polygons


class RandomScaleCrop(object):
    """
    Crops the image with a random scale while maintaining the original aspect ratio,
    then resizes it to a fixed size.
    """
    def __init__(self, size, scale=(0.24, 1.0)):
        self.size = (size, size)
        self.scale = scale

    @staticmethod
    def get_params(img, scale):
        h_img, w_img, _ = img.shape
        img_aspect_ratio = h_img / w_img

        for _ in range(10):
            target_scale = np.random.uniform(*scale)
            target_area = h_img * w_img * target_scale

            w_crop = int(round(math.sqrt(target_area / img_aspect_ratio)))
            h_crop = int(round(w_crop * img_aspect_ratio))

            if w_crop < w_img and h_crop < h_img:
                i = np.random.randint(0, h_img - h_crop + 1)
                j = np.random.randint(0, w_img - w_crop + 1)
                return i, j, h_crop, w_crop

        # Fallback to center crop
        in_ratio = h_img / w_img
        out_ratio = 1.0  # Square output
        if in_ratio > out_ratio:
            h_crop = int(w_img * out_ratio)
            w_crop = w_img
        else:
            h_crop = h_img
            w_crop = int(h_img / out_ratio)

        i = (h_img - h_crop) // 2
        j = (w_img - w_crop) // 2
        return i, j, h_crop, w_crop

    def __call__(self, image, polygons=None):
        i, j, h, w = self.get_params(image, self.scale)

        cropped = image[i:i + h, j:j + w, :]
        scales = np.array([self.size[0] / w, self.size[1] / h])
        
        if polygons is not None:
            for polygon in polygons:
                polygon.points[:, 0] = (polygon.points[:, 0] - j) * scales[0]
                polygon.points[:, 1] = (polygon.points[:, 1] - i) * scales[1]

        img = cv2.resize(cropped, self.size)
        return img, polygons


class RandomFixedSizeCrop(object):
    """
    Takes a random crop of fixed size (self.size).
    If the image is smaller than self.size, it pads the image first (with 0s on bottom/right).
    The crop is then taken from this (potentially padded) image.
    """
    def __init__(self, size):
        self.size = size # Assumed to be a single int for square crop

    def __call__(self, image, polygons=None):
        h_img, w_img, _ = image.shape
        target_h, target_w = self.size, self.size

        # Pad if image is smaller than target crop size
        # Padding will be applied to the bottom and right
        pad_h, pad_w = 0, 0
        if h_img < target_h:
            pad_h = target_h - h_img
        if w_img < target_w:
            pad_w = target_w - w_img
        
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # Polygons do not need to be adjusted for padding that extends the image
            # as long as the points themselves are not moved relative to their origin.
            h_img, w_img, _ = image.shape # Update dimensions after padding

        # Randomly choose crop starting point
        if h_img == target_h:
            i = 0
        else:
            i = np.random.randint(0, h_img - target_h + 1)
        
        if w_img == target_w:
            j = 0
        else:
            j = np.random.randint(0, w_img - target_w + 1)

        cropped = image[i:i + target_h, j:j + target_w, :]

        if polygons is not None:
            for polygon in polygons:
                # Adjust polygon coordinates relative to the crop
                polygon.points[:, 0] -= j
                polygon.points[:, 1] -= i
                
                # IMPORTANT: Polygons might now be entirely or partially outside the crop.
                # The current system will still process them; downstream steps (like disk_cover)
                # might implicitly handle points out of bounds or create empty masks for them.
                # For a robust solution, one might filter out polygons completely outside
                # the crop or clip their coordinates here. For this request, simple adjustment suffices.
        
        return cropped, polygons


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, polygons


class Resize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size,
                                   self.size))
        scales = np.array([self.size / w, self.size / h])

        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons


class Augmentation(object):

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            # Resize(size),
            Padding(),
            RandomFixedSizeCrop(size=size),
            # RandomBrightness(),
            # RandomContrast(),
            RandomMirror(),
            Rotate(),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)



class PadToMultipleOf32(object):
    def __init__(self):
        pass

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        new_h = (h // 32 + (1 if h % 32 != 0 else 0)) * 32
        new_w = (w // 32 + (1 if w % 32 != 0 else 0)) * 32
        
        padded_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)
        padded_image[:h, :w, :] = image
        
        return padded_image, polygons


class InferenceTransform(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            PadToMultipleOf32(),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class BaseTransform(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            RandomFixedSizeCrop(size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)