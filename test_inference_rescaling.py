import cv2
import numpy as np
from util.augmentation import InferenceTransform

image = np.ones((500, 600, 3), dtype=np.uint8) * 128
transform = InferenceTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
out_img, _ = transform(image)

print(f"Original shape: {image.shape}")
print(f"Transformed shape: {out_img.shape}")
