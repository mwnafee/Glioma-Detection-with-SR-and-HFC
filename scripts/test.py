import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.ensemble import ensemble_voting, evaluate_models

def build_generators(img_paths, labels, batch_size, is_training=True, target_size=(224, 224)):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    ) if is_training else ImageDataGenerator(rescale=1./255)

    df = pd.DataFrame({'filename': img_paths, 'class': labels})
    return datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='class',
        class_mode='binary',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

def get_data_paths(data_dir):
    import os
    labels = []
    filepaths = []
    for class_label in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, class_label)
        if os.path.isdir(class_folder):
            for fname in os.listdir(class_folder):
                filepaths.append(os.path.join(class_folder, fname))
                labels.append(class_label)
    return np.array(filepaths), np.array(labels).astype('int')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--superimage_dir', type=str, required=True)
    parser.add_argument('--map_dir', type=str, required=True)
    parser.add_argument('--model1_path', type=str, required=True)
    parser.add_argument('--model2_path', type=str, required=True)
    parser.add_argument('--model3_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    X_img, y_img = get_data_paths(args.superimage_dir)
    X_map, y_map = get_data_paths(args.map_dir)

    test_gen_img = build_generators(X_img, y_img, args.batch_size, is_training=False)
    test_gen_map = build_generators(X_map, y_map, args.batch_size, is_training=False)

    models = [model1, model2, model3]
    test_generators = [test_gen_img, test_gen_img, test_gen_map]
    
    # Evaluate individual models and ensemble
    print("\n📊 Running full model evaluation...")
    evaluate_models(models, test_generators, threshold=0.5)
    
    # Also run ensemble voting with tie-breaking strategy
    print("\n🗳️ Running ensemble voting (with fallback to model3)...")
    ensemble_voting(models, test_gen_img, test_gen_map, threshold=0.5)
