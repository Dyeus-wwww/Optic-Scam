# 🔬 Optic-Scan — Glaucoma Detection with ResNet18

A deep learning project for automated **glaucoma detection** from retinal/optic disc images. Built with PyTorch and transfer learning, this tool classifies eye scan images into two categories: **Normal** and **Glaucoma** using a fine-tuned ResNet18 model.

---

## 📋 Features

- **Transfer Learning** — Leverages pre-trained ResNet18 (ImageNet) for fast and accurate convergence
- **Binary Classification** — Distinguishes between normal retinal images and glaucoma cases
- **Mixed Precision Training** — Uses `torch.cuda.amp` for faster training with lower memory usage
- **Data Augmentation** — Random resized crops and horizontal flips to improve generalization
- **Training Visualization** — Built-in method to visualize model predictions on validation images
- **Best Model Checkpointing** — Automatically saves the model with the highest validation accuracy

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `model.py` | Core model class (`ResNet18`) containing training loop, data loading, and visualization |
| `main.ipynb` | Jupyter notebook for training the model and running inference |
| `data/` | Dataset directory with `train/` and `val/` subfolders (not included in repo) |

---

## 🧠 Model Architecture

- **Base Model**: `torchvision.models.resnet18` (pre-trained on ImageNet)
- **Modified Layer**: Final fully-connected layer replaced with `nn.Linear(512, 2)` for binary classification
- **Input Size**: 224 × 224 RGB images
- **Output**: 2 classes — `normal` (0) and `glaucoma` (1)

### Data Preprocessing

| Phase | Transformations |
|-------|-----------------|
| **Train** | `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` |
| **Val** | `Resize(256)`, `CenterCrop(224)`, `ToTensor()`, `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended)
- torchvision
- matplotlib
- PIL

```bash
pip install torch torchvision matplotlib pillow
```

### Dataset Setup

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── normal/       # Normal retinal images
│   └── glaucoma/     # Glaucoma retinal images
└── val/
    ├── normal/       # Normal retinal images
    └── glaucoma/     # Glaucoma retinal images
```

> Update the `data_dir` path in `model.py` to point to your dataset location.

### Training

Open `main.ipynb` in Jupyter Notebook and run all cells:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from model import ResNet18

# Load pre-trained ResNet18
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)  # 2 classes: normal, glaucoma

# Setup optimizer and scheduler
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Train
model_ft, class_name, dataloaders = ResNet18().train(
    model=model_ft,
    criterion=criterion,
    optimizer=optimizer_ft,
    scheduler=exp_lr_scheduler,
    num_epochs=60
)
```

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 60 |
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Momentum | 0.9 |
| LR Scheduler | StepLR (step=7, gamma=0.1) |
| Loss Function | CrossEntropyLoss |
| Optimizer | SGD |
| DataLoader Workers | 16 |
| Mixed Precision | Enabled (`GradScaler` + `autocast`) |

---

## 📊 Training Results

Sample training progress (from `main.ipynb`):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0 | 0.5837 | 69.52% | 0.4368 | 79.99% |
| 1 | 0.4702 | 77.64% | 0.3889 | 82.23% |
| 2 | 0.4403 | 79.56% | 0.3539 | 84.06% |
| 3 | 0.4142 | 81.09% | 0.3335 | 85.21% |
| 4 | 0.3994 | 81.51% | 0.3350 | 85.70% |
| 5 | 0.3787 | 82.87% | 0.3152 | 86.36% |
| 6 | 0.3788 | 82.53% | 0.2961 | 87.00% |
| 7 | 0.3701 | 83.63% | 0.2984 | 86.79% |
| 8 | 0.3557 | 84.14% | — | — |

> Training was interrupted at epoch 8. The model showed strong improvement, reaching ~87% validation accuracy.

---

## 🖼️ Visualization

After training, visualize model predictions on validation images:

```python
ResNet18().visualize_model(
    model=model_ft,
    dataloaders=dataloaders,
    class_names=class_name,
    num_images=6
)
```

This displays a grid of validation images with their predicted labels (`normal` or `glaucoma`).

---

## 📝 Notes

- The project uses **ImageNet normalization** statistics, which are standard for transfer learning with ResNet
- **Mixed precision training** (`torch.cuda.amp`) is enabled for faster GPU training
- The best model weights are saved to a temporary directory during training and loaded back at the end
- Ensure your dataset is properly labeled with folder names matching class names (`normal`, `glaucoma`)
- The hardcoded data path in `model.py` is Windows-style; update it for your environment

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

---

## 🙋‍♂️ Author

[Dyeus-wwww](https://github.com/Dyeus-wwww)
