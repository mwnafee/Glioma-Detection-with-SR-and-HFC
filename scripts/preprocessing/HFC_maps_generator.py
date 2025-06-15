import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def calculate_pixel_weights(image, alpha=0.2):
    """
    Computes a per-pixel weight map highlighting high-frequency regions.
    Pixels below mean + alpha × std are suppressed.
    """
    image = image.astype(np.float32)
    h, w, _ = image.shape
    weight_map = np.zeros((h, w), dtype=np.float32)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            pc = image[i, j]
            neighbors = [image[i - 1, j], image[i + 1, j], image[i, j - 1], image[i, j + 1]]
            diffs = [np.abs(pc - n) for n in neighbors]
            max_diff = np.max(np.array(diffs), axis=0)
            weight_map[i, j] = np.max(max_diff)

    # Normalize and apply threshold
    min_w, max_w = np.min(weight_map), np.max(weight_map)
    normalized = (weight_map - min_w) / (max_w - min_w + 1e-8)
    mean, std = np.mean(normalized), np.std(normalized)
    normalized[normalized < mean + alpha * std] = 0

    return normalized

def process_images_maps(input_dir, save_dir):
    """
    Generate and save weight maps for all images in a directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    for img_file in tqdm(image_files, desc=f"Processing {input_dir}"):
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        weight_map = calculate_pixel_weights(image)
        weight_map_scaled = cv2.normalize(weight_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir, img_file), weight_map_scaled)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate high-frequency weight maps for images.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing RGB input images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save generated weight maps.")
    args = parser.parse_args()

    process_images_maps(args.input_dir, args.output_dir)
    print("✅ Weight map generation complete.")
