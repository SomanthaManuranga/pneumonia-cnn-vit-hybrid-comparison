# Pneumonia Detection - CNN vs ViT and Hybrid Architechture Comparison

Compares ResNet-50, ViT-Base, and TinyViT-21M on chest X-ray classification and Grad-CAM explainability.

## Requirements

```bash
pip install torch torchvision timm scikit-learn pillow tqdm matplotlib pytorch-grad-cam scipy
```

## Dataset

Download [Chest X-Ray Images (Kermany)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle and place it at `Project/Dataset/chest_xray/` with `train/`, `val/`, and `test/` subfolders, each containing `NORMAL/` and `PNEUMONIA/` folders.

## How to Run

**Step 1 - Train (requires GPU, ≥8 GB VRAM):**

Run each notebook top to bottom in this order:

1. `Resnet50_Analysis.ipynb`
2. `ViT_base_Analysis.ipynb`
3. `TinyViT_Analysis.ipynb`

Checkpoints are saved to `checkpoints/`, results to `results/`.

**Step 2 — Explainability:**

After all three models are trained, run `GradCAM_analysis.ipynb`.
Outputs: heatmap grid and metric scores saved to `results/`.
