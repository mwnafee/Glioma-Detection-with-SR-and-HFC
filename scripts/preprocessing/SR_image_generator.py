import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import re
from cv2 import dnn_superres

def preprocessing(img_path, model_path):
    """
    Applies super-resolution to an image using OpenCV's DNN superres module.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load SR model
    model_name = os.path.basename(model_path).split('_')[0].lower()
    scale_match = re.search(r'x(\d+)', model_path)
    scale = int(scale_match.group(1)) if scale_match else 4

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name, scale)

    # Apply super-resolution
    img_upscaled = sr.upsample(img_rgb)
    return img_upscaled

def sharpen_image(img):
    """
    Sharpens an image using a standard kernel.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def process_directory(input_dir, output_dir, model_path):
    """
    Applies super-resolution and sharpening to all images in a folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for fname in tqdm(image_files, desc=f"Processing {input_dir}"):
        img_path = os.path.join(input_dir, fname)
        image_sr = preprocessing(img_path, model_path)

        if image_sr is None:
            continue

        image_sharp = sharpen_image(image_sr)
        image_bgr = cv2.cvtColor(image_sharp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, fname), image_bgr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super-resolution + sharpening for medical images.")
    parser.add_argument('--train_dir', type=str, required=True, help="Input directory for training images.")
    parser.add_argument('--test_dir', type=str, required=True, help="Input directory for test images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory to save high-res images.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to .pb super-resolution model.")
    args = parser.parse_args()

    train_out = os.path.join(args.output_dir, "training")
    test_out = os.path.join(args.output_dir, "test")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    process_directory(args.train_dir, train_out, args.model_path)
    process_directory(args.test_dir, test_out, args.model_path)

    print("✅ Super-resolution processing complete.")
