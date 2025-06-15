import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import argparse

def extract_rectangles_from_image(image_path, json_path, dataset_type="training", target_size=(1015, 1015)):
    """
    Extracts cell bounding boxes from annotated WSI using corresponding JSON.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    height = data.get('imageHeight', 0)
    width = data.get('imageWidth', 0)

    if height == 1015 and width == 1015:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    shapes = data.get('shapes', [])
    for shape in shapes:
        for (x, y) in shape.get('points', []):
            if x > 512 or y > 512:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
                break

    extracted_rects = []
    for idx, shape in enumerate(shapes):
        points = shape.get("points", [])
        if not points:
            continue

        pts = np.round(np.array(points, dtype=np.float32)).astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts.reshape((-1, 1, 2)))
        cropped = image[y:y+h, x:x+w].copy()

        if dataset_type.lower() == "training":
            label = shape.get("label", "unknown").lower().replace(" ", "")
        else:
            label = "blank"

        extracted_rects.append((cropped, label, idx))

    return extracted_rects

def process_directory_extract_rectangles(input_dir, output_dir, dataset_type="training", target_size=(1015, 1015)):
    """
    Iterates through image-JSON pairs and saves cropped rectangles to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')])

    for img_file in tqdm(image_files, desc=f"Processing {dataset_type}"):
        image_path = os.path.join(input_dir, img_file)
        base = os.path.splitext(img_file)[0]
        json_path = os.path.join(input_dir, base + ".json")

        if not os.path.exists(json_path):
            print(f"⚠️ Skipping {img_file}: No JSON found.")
            continue

        rectangles = extract_rectangles_from_image(image_path, json_path, dataset_type, target_size)
        for cropped_img, label, idx in rectangles:
            filename = f"{base}_{label}_{idx}.jpg" if dataset_type == "training" else f"{base}_blank{idx+1}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), cropped_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cell images from annotated WSI using bounding boxes.")
    parser.add_argument("--train_input", type=str, required=True, help="Input folder containing training WSI and JSONs.")
    parser.add_argument("--test_input", type=str, required=True, help="Input folder containing test WSI and JSONs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory to save extracted cell images.")
    args = parser.parse_args()

    train_out = os.path.join(args.output_dir, "training_rectangles")
    test_out  = os.path.join(args.output_dir, "testing_rectangles")

    print("📦 Processing training dataset:")
    process_directory_extract_rectangles(args.train_input, train_out, dataset_type="training")

    print("\n📦 Processing testing dataset:")
    process_directory_extract_rectangles(args.test_input, test_out, dataset_type="testing")

    print("\n✅ Done.")
    print(f"🗂️ Training rectangles saved to: {os.path.abspath(train_out)}")
    print(f"🗂️ Testing rectangles saved to:  {os.path.abspath(test_out)}")
