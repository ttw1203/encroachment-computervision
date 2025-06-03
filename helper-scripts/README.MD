# `sahi_rfdetr_advanced.py`

A powerful utility script for slicing-based inference using **RF-DETR** models integrated with the **SAHI** framework. It processes datasets split into `train`, `val`, and `test` directories and produces **YOLO-format annotations** compatible with standard training pipelines.

---

## 🚀 Key Features

### 1. **RF-DETR Integration**
- Custom wrapper added to support **RF-DETR** with SAHI.
- Choose between:
  - `RFDETRBase` (29M parameters)
  - `RFDETRLarge` (128M parameters)
- Use **pretrained COCO** models or **custom-trained weights**.
- Experimental support for **SAHI's built-in RT-DETR** also available.

### 2. **Updated `data.yaml` Format**
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
nc: number_of_classes
names: [class1, class2, class3, ...]
```

### 3. **Split-Based Processing**
- Specify `--train`, `--val`, and `--test` directories separately.
- Each split is processed **independently**.
- Output organized as:
  ```
  output_dir/{train,val,test}/{images,labels}
  ```

### 4. **Enhanced Functionality**
- Optional **visualization** of predictions using OpenCV.
- **Error handling** for missing RF-DETR dependencies.
- Logs **class mappings** and **detailed inference stats**.
- Uses `supervision`, `SAHI`, `OpenCV`, and `PyYAML`.

---

## 🧩 Installation

Install required dependencies:
```bash
pip install rfdetr supervision sahi opencv-python tqdm pyyaml numpy pillow
```

---

## 💡 Usage Examples

### ▶️ Inference with pretrained COCO RF-DETR:
```bash
python sahi_rfdetr_advanced.py \
  --train /path/to/train/images \
  --val /path/to/val/images \
  --test /path/to/test/images \
  --output /path/to/output \
  --model coco \
  --model-size base
```

### ▶️ Inference using custom-trained RF-DETR weights:
```bash
python sahi_rfdetr_advanced.py \
  --train /path/to/train/images \
  --val /path/to/val/images \
  --output /path/to/output \
  --model /path/to/custom_weights.pt \
  --model-size large \
  --conf 0.3 \
  --slice-size 512 \
  --visualize
```

### ▶️ Process only the training set:
```bash
python sahi_rfdetr_advanced.py \
  --train /path/to/train/images \
  --output /path/to/output \
  --model coco \
  --save-crops
```

---

## ⚙️ Key Arguments

| Argument              | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `--model`             | Path to RF-DETR weights or `"coco"` for pretrained                       |
| `--model-size`        | Choose between `"base"` (29M) and `"large"` (128M)                       |
| `--train / --val / --test` | Image directories per split                                         |
| `--slice-size`        | Size of image tiles (default: 640)                                       |
| `--overlap`           | Overlap ratio between slices (default: 0.2)                              |
| `--conf`              | Confidence threshold (default: 0.25)                                     |
| `--visualize`         | Visualize predictions with OpenCV                                        |
| `--save-crops`        | Save cropped detection patches                                           |
| `--use-sahi-integration` | Use SAHI’s built-in RT-DETR wrapper (experimental)                   |

---

## 📦 Output Structure

```
output_dir/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
├── classes.txt
├── class_mapping.json
└── inference_stats.json
```

- Labels are in **YOLO format**.
- `data.yaml` is auto-generated.
- Stats include class counts and confidence summaries.

---

## 🛠️ Notes
- If RF-DETR is not installed, the script will raise a descriptive error.
- Supports both local `.pt` models and SAHI-compatible detection models.
- Visualizations can be toggled with `--visualize`.
