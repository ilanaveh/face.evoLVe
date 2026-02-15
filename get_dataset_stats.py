"""
12/2/26
For normalization - get mean (& std) rgb of dataset.
Dataset: MS-Celeb-1M_Align_112x112
"""

import os
import numpy as np
from PIL import Image


print2file = True

data_dir = '/home/projects/bagon/ilanaveh/data/imgs'
output_path = '/home/projects/bagon/ilanaveh/code/face.evoLVe/img_stats/image_rgb_stats.txt'

# Supported image extensions
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

pixel_sum = np.zeros(3)
pixel_sq_sum = np.zeros(3)
pixel_count = 0
img_count = 0
dir_count = 0

for root, _, files in os.walk(data_dir):
    if (dir_count % 1000) == 0:
        print(f"directory {dir_count}/85,742")
    for file in files:
        if file.lower().endswith(IMG_EXTENSIONS):
            img_count += 1
            img_path = os.path.join(root, file)

            # Open image and convert to RGB
            img = Image.open(img_path).convert('RGB')
            img = np.array(img) / 255.0  # Normalize to [0,1]

            # Reshape to (num_pixels, 3)
            img = img.reshape(-1, 3)

            pixel_sum += img.sum(axis=0)
            pixel_sq_sum += (img ** 2).sum(axis=0)
            pixel_count += img.shape[0]
    dir_count += 1

mean = pixel_sum / pixel_count
std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

if print2file:
    with open(output_path, "w") as f:
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Total image directories: {dir_count}\n")
        f.write(f"Total images: {img_count}\n")
        f.write(f"Mean RGB: {mean.tolist()}\n")
        f.write(f"Std RGB: {std.tolist()}")

print("Mean RGB:", mean)
print("Std RGB:", std)
