import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from models.efficientnet_model import initialize_model1
from models.resnet_model import initialize_model2, initialize_model3

def get_data_paths(data_dir):
    labels = []
    filepaths = []
    for class_label in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, class_label)
        if os.path.isdir(class_folder):
            for fname in os.listdir(class_folder):
                filepaths.append(os.path.join(class_folder, fname))
                labels.append(class_label)
    return np.array(filepaths), np.array(labels).astype('int')

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
        shuffle=is_training
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--superimage_dir', type=str, required=True)
    parser.add_argument('--map_dir', type=str, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    superimage_paths, superimage_labels = get_data_paths(args.superimage_dir)
    map_paths, map_labels = get_data_paths(args.map_dir)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(superimage_paths, superimage_labels)):
        print(f'📁 Fold {fold + 1}/{args.folds}')

        X_train_img, y_train_img = superimage_paths[train_idx], superimage_labels[train_idx]
        X_val_img, y_val_img = superimage_paths[val_idx], superimage_labels[val_idx]

        X_train_map, y_train_map = map_paths[train_idx], map_labels[train_idx]
        X_val_map, y_val_map = map_paths[val_idx], map_labels[val_idx]

        train_gen1 = build_generators(X_train_img, y_train_img, args.batch_size, is_training=True)
        val_gen1 = build_generators(X_val_img, y_val_img, args.batch_size, is_training=False)

        train_gen2 = build_generators(X_train_img, y_train_img, args.batch_size, is_training=True)
        val_gen2 = build_generators(X_val_img, y_val_img, args.batch_size, is_training=False)

        train_gen3 = build_generators(X_train_map, y_train_map, args.batch_size, is_training=True)
        val_gen3 = build_generators(X_val_map, y_val_map, args.batch_size, is_training=False)

        callbacks = lambda name: [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(patience=3, factor=0.5),
            ModelCheckpoint(f'{name}_fold{fold}.h5', save_best_only=True)
        ]

        print("🔧 Training model1 (EfficientNet)")
        model1 = initialize_model1()
        model1.fit(train_gen1, validation_data=val_gen1, epochs=args.epochs, callbacks=callbacks("model1"))

        print("🔧 Training model2 (ResNet on superimage)")
        model2 = initialize_model2()
        model2.fit(train_gen2, validation_data=val_gen2, epochs=args.epochs, callbacks=callbacks("model2"))

        print("🔧 Training model3 (ResNet on maps)")
        model3 = initialize_model3()
        model3.fit(train_gen3, validation_data=val_gen3, epochs=args.epochs, callbacks=callbacks("model3"))
