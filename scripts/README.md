
# Glioma-MDC 2025: High-Frequency and Super-Resolved Cell Classification Pipeline

This repository provides a full deep learning pipeline for extracting and classifying mitotic cells from whole-slide glioma images using:

- 🧩 Cell-level image extraction from annotated WSIs
- 🔬 Super-resolution preprocessing (EDSR)
- ⚡ High-frequency contrast (HFC) map generation
- 🧠 Ensemble classification with CNNs (EfficientNet, ResNet)
- 📊 Cross-validation and ensemble voting evaluation

---

## 🗂️ Project Structure



.
├── cell\_image\_extractor.py        # Extracts cell images from JSON-annotated WSIs
├── super\_res\_generator.py         # Applies super-resolution and sharpening
├── map\_generation.py              # Generates high-frequency (HFC) contrast maps
├── train.py                       # Trains CNNs (EfficientNet, ResNet) with 5-fold CV
├── test.py                        # Evaluates models and performs ensemble voting
├── models/
│   ├── efficientnet\_model.py
│   └── resnet\_model.py
├── utils/
│   ├── data\_utils.py
│   └── ensemble.py
└── README.md

````

---

## 🧬 Step-by-Step Workflow

### 1️⃣ Extract Cell Images from WSIs

```bash
python cell_image_extractor.py \
  --train_input /path/to/Glioma_MDC_2025_training \
  --test_input /path/to/Glioma_MDC_2025_test \
  --output_dir ./whole
````

* Outputs: `./whole/training_rectangles/`, `./whole/testing_rectangles/`

---

### 2️⃣ Super-Resolution Processing

```bash
python super_res_generator.py \
  --train_dir ./whole/training_rectangles \
  --test_dir ./whole/testing_rectangles \
  --output_dir ./Preprocess_HighRes \
  --model_path /path/to/EDSR_x4.pb
```

* Uses OpenCV's `dnn_superres` to upscale and sharpen cell crops

---

### 3️⃣ Generate HFC (High-Frequency Contrast) Maps

```bash
python map_generation.py \
  --input_dir ./Preprocess_HighRes/training \
  --output_dir ./maps/training

python map_generation.py \
  --input_dir ./Preprocess_HighRes/test \
  --output_dir ./maps/test
```

* Highlights edge and texture differences

---

### 4️⃣ Train Ensemble CNNs

```bash
python train.py \
  --superimage_dir ./Preprocess_HighRes/training \
  --map_dir ./maps/training \
  --folds 5 \
  --epochs 20 \
  --batch_size 32
```

* Trains 3 models:

  * EfficientNet (superimage)
  * ResNet (superimage)
  * ResNet (map)
* Performs 5-fold cross-validation

---

### 5️⃣ Evaluate & Ensemble Predictions

```bash
python test.py \
  --superimage_dir ./Preprocess_HighRes/test \
  --map_dir ./maps/test \
  --model1_path model1_fold0.h5 \
  --model2_path model2_fold0.h5 \
  --model3_path model3_fold0.h5
```

* Outputs per-model metrics and ensemble voting results

---

## 🧪 Models Used

* **EfficientNetB0** – Trained on super-resolution images
* **ResNet50** – Trained separately on superimages and HFC maps
* **Ensemble Voting** – Tie-breaking logic using model3 (map-based)

---

## 📖 Citation

If you use this repository for research or development, please cite:

```bibtex
@INPROCEEDINGS{10980984,
  author={Nafee, Mahmud Wasif and Juicy, Asfina Hassan},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)}, 
  title={Enhanced Mitotic Figure Detection in Glioma Using Super-Resolution Images and High-Frequency Content Maps}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ISBI60581.2025.10980984}
}
```



---

## 🙋 Contact

For questions, feedback, or collaborations:

**Mahmud Wasif Nafee**
📧 [wasifnafee@gmail.com](mailto:wasifnafee@gmail.com)


