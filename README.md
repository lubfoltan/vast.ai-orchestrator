# Vast.ai Training Orchestrator

A Python desktop GUI application for automating deep learning model training on [Vast.ai](https://vast.ai) cloud GPUs. Supports image classification and tabular regression — bring your own dataset.

---

## Features

- **One-click pipeline**: searches for a GPU, rents it, uploads data, trains, downloads results — fully automated
- **Classification & Regression** — switch task type; metrics and options adjust automatically
- **10 classification metrics**: Accuracy, Loss, Precision, Recall, F1, AUC-ROC, Confusion Matrix, Specificity, Sensitivity, Cohen's Kappa
- **7 regression metrics**: MSE, RMSE, MAE, R², Loss, Explained Variance, MAPE
- **Excel export** — `results.xlsx` with Epoch History, Final Metrics, and Predictions sheets
- **Selectable CNN models**: ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt
- **Optional features**: Early Stopping, Grad-CAM, LR Scheduler, Mixup, Label Smoothing, Data Augmentation
- **Custom training script** — bring your own `.py` instead of the built-in `train.py`
- **Interactive SSH Console** — run commands on the remote server from inside the app
- **Stop / Terminate** — pause billing (stop) or fully destroy the instance

---

## Project Structure

```
├── main.py                  # Entry point
├── gui.py                   # CustomTkinter GUI
├── orchestrator.py          # Pipeline orchestration (search → train → download)
├── config.py                # Experiment configuration dataclass
├── vast_api.py              # Vast.ai Python SDK wrapper
├── ssh_manager.py           # Paramiko SSH / SFTP manager
├── train.py                 # Remote training script (classification + regression)
├── requirements.txt         # Local dependencies
└── CUSTOM_SCRIPT_GUIDE.md   # Guide for writing custom training scripts
```

---

## Requirements

- Python 3.10+
- A [Vast.ai](https://vast.ai) account with credit

Install local dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Vast.ai API Key
Go to [vast.ai/accounts](https://vast.ai/accounts) → **API Key** → copy it into the app.

### 2. SSH Key Pair
Generate a key pair (if you don't have one):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_key
```

Upload the **public key** (`vast_key.pub`) to Vast.ai → Account → SSH Keys.

In the app, provide the path to the **private key** (`vast_key`).

---

## Usage

```bash
python main.py
```

Fill in the config panel:

| Field | Description |
|-------|-------------|
| Task Type | Classification (images) or Regression (CSV/Excel) |
| Train Data | Local folder with training data |
| Test Data | Optional separate test folder |
| Output Path | Where results will be downloaded |
| API Key | Your Vast.ai API key |
| SSH Key | Path to your local private SSH key |
| Custom Script | Optional `.py` file to run instead of built-in `train.py` |

Click **▶ Start Pipeline** — the app will:
1. Find the cheapest GPU matching your filters
2. Rent it and wait for it to start
3. Install dependencies on the remote server
4. Upload your data and training script
5. Run training with live log streaming
6. Download all results (models, plots, `results.xlsx`) to your Output Path

---

## Classification — Expected Data Format

### Option A: ImageFolder structure

```
data/
├── CAT/
│   ├── img001.png
│   └── img002.png
└── DOG/
    └── img001.png
```

### Option B: Flat folder (auto-organized)

If all images are in a single folder, the app **automatically detects the class label from the filename** using a regex that extracts the leading alphabetic prefix:

```
data/
├── CAT_001.png      →  class CAT
├── CAT_002.png      →  class CAT
├── DOG_001.png      →  class DOG
└── DOG_002.png      →  class DOG
```

The regex used is `^([A-Za-z]+)` — everything before the first digit or underscore becomes the class name (uppercased).

> **Important:** For flat folder mode to work correctly, **the class name must appear at the start of the filename**, followed by a digit or underscore. Example: `TUMOR_042.jpg` → class `TUMOR`.

---

## Regression — Expected Data Format

Place a **CSV or Excel file** in the Train Data folder:

```
data/
└── dataset.csv
```

Set **Target Column** (the column to predict) and optionally **Feature Columns** (comma-separated; defaults to all columns except the target).

---

## Custom Training Script

You can provide your own `.py` script instead of the built-in `train.py`.  
See [CUSTOM_SCRIPT_GUIDE.md](CUSTOM_SCRIPT_GUIDE.md) for available remote paths and a minimal template.

---

## Output Files

After training, the following files are downloaded to your Output Path:

| File | Description |
|------|-------------|
| `best_model.pth` | Best model checkpoint (lowest val loss) |
| `final_model.pth` | Final model after all epochs |
| `results.xlsx` | Metrics per epoch + predictions (Excel) |
| `loss_accuracy.png` | Training / validation loss & accuracy curves |
| `metrics.png` | All selected metrics over epochs |
| `confusion_matrix.png` | Confusion matrix (classification) |
| `roc_curve.png` | ROC curve (classification) |
| `gradcam_*.png` | Grad-CAM visualizations (classification, optional) |
| `pred_vs_actual.png` | Predicted vs Actual scatter (regression) |
| `residuals.png` | Residual plot (regression) |

---

## Security

- API keys and SSH keys are **never stored on disk** — entered only in the GUI at runtime
- `orchestrator.log` is listed in `.gitignore` (may contain remote IP addresses)
- Private SSH keys should **never be committed** — add to `.gitignore`

---

## License

MIT


---

## Features

- **One-click pipeline**: searches for a GPU, rents it, uploads data, trains, downloads results — fully automated
- **Classification & Regression** — switch task type; metrics and options adjust automatically
- **10 classification metrics**: Accuracy, Loss, Precision, Recall, F1, AUC-ROC, Confusion Matrix, Specificity, Sensitivity, Cohen's Kappa
- **7 regression metrics**: MSE, RMSE, MAE, R², Loss, Explained Variance, MAPE
- **Excel export** — `results.xlsx` with Epoch History, Final Metrics, and Predictions sheets
- **Selectable CNN models**: ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt
- **Optional features**: Early Stopping, Grad-CAM, LR Scheduler, Mixup, Label Smoothing, Data Augmentation
- **Custom training script** — bring your own `.py` instead of the built-in `train.py`
- **Interactive SSH Console** — run commands on the remote server from inside the app
- **Stop / Terminate** — pause billing (stop) or fully destroy the instance

---

## Project Structure

```
├── main.py                  # Entry point
├── gui.py                   # CustomTkinter GUI
├── orchestrator.py          # Pipeline orchestration (search → train → download)
├── config.py                # Experiment configuration dataclass
├── vast_api.py              # Vast.ai Python SDK wrapper
├── ssh_manager.py           # Paramiko SSH / SFTP manager
├── train.py                 # Remote training script (classification + regression)
├── requirements.txt         # Local dependencies
└── CUSTOM_SCRIPT_GUIDE.md   # Guide for writing custom training scripts
```

---

## Requirements

- Python 3.10+
- A [Vast.ai](https://vast.ai) account with credit

Install local dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Vast.ai API Key
Go to [vast.ai/accounts](https://vast.ai/accounts) → **API Key** → copy it into the app.

### 2. SSH Key Pair
Generate a key pair (if you don't have one):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_key
```

Upload the **public key** (`vast_key.pub`) to Vast.ai → Account → SSH Keys.

In the app, provide the path to the **private key** (`vast_key`).

---

## Usage

```bash
python main.py
```

Fill in the config panel:

| Field | Description |
|-------|-------------|
| Task Type | Classification (images) or Regression (CSV/Excel) |
| Train Data | Local folder with training data |
| Test Data | Optional separate test folder |
| Output Path | Where results will be downloaded |
| API Key | Your Vast.ai API key |
| SSH Key | Path to your local private SSH key |
| Custom Script | Optional `.py` file to run instead of built-in `train.py` |

Click **▶ Start Pipeline** — the app will:
1. Find the cheapest GPU matching your filters
2. Rent it and wait for it to start
3. Install dependencies on the remote server
4. Upload your data and training script
5. Run training with live log streaming
6. Download all results (models, plots, `results.xlsx`) to your Output Path

---

## Classification — Expected Data Format

**ImageFolder structure** (recommended):
```
data/
├── CLASS_A/
│   ├── img001.png
│   └── img002.png
└── CLASS_B/
    └── img001.png
```

**Flat folder** (auto-organized):  
If all images are in a single folder, the app detects class labels from filename prefixes:
```
data/
├── NORMAL_001.png   → class NORMAL
├── PNEUMONIA_001.png → class PNEUMONIA
```

---

## Regression — Expected Data Format

Place a **CSV or Excel file** in the Train Data folder:

```
data/
└── dataset.csv
```

Set **Target Column** (the column to predict) and optionally **Feature Columns** (comma-separated; defaults to all columns except the target).

---

## Custom Training Script

You can provide your own `.py` script instead of the built-in `train.py`.  
See [CUSTOM_SCRIPT_GUIDE.md](CUSTOM_SCRIPT_GUIDE.md) for available remote paths and a minimal template.

---

## Output Files

After training, the following files are downloaded to your Output Path:

| File | Description |
|------|-------------|
| `best_model.pth` | Best model checkpoint (lowest val loss) |
| `final_model.pth` | Final model after all epochs |
| `results.xlsx` | Metrics per epoch + predictions (Excel) |
| `loss_accuracy.png` | Training / validation loss & accuracy curves |
| `metrics.png` | All selected metrics over epochs |
| `confusion_matrix.png` | Confusion matrix (classification) |
| `roc_curve.png` | ROC curve (classification) |
| `gradcam_*.png` | Grad-CAM visualizations (classification, optional) |
| `pred_vs_actual.png` | Predicted vs Actual scatter (regression) |
| `residuals.png` | Residual plot (regression) |

---

## Security

- API keys and SSH keys are **never stored on disk** — entered only in the GUI at runtime
- `orchestrator.log` is listed in `.gitignore` (may contain remote IP addresses)
- Private SSH keys should **never be committed** — add to `.gitignore`

---

## License

MIT

| Variable / Path         | Description                                  |
|------------------------|----------------------------------------------|
| `/workspace/data`      | Your uploaded **training data** (input)       |
| `/workspace/test_data` | Your uploaded **test data** (if provided)     |
| `/workspace/output`    | **Output folder** — put all results here      |

Everything placed in `/workspace/output` will be automatically downloaded
to your local Output Path after training finishes.

## Minimal Custom Script Template

```python
import os

# ── Paths (these are fixed on the remote server) ──
DATA_DIR = "/workspace/data"
TEST_DIR = "/workspace/test_data"   # empty if not provided
OUTPUT_DIR = "/workspace/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Your training code here ──
print(f"Training data: {DATA_DIR}")
print(f"Test data:     {TEST_DIR}")
print(f"Output dir:    {OUTPUT_DIR}")

# Example: list files
for f in os.listdir(DATA_DIR):
    print(f"  {f}")

# Save results to OUTPUT_DIR so they get downloaded
with open(os.path.join(OUTPUT_DIR, "result.txt"), "w") as f:
    f.write("Training complete!\n")
```

## Tips

- **Install extra packages** at the top of your script:
  ```python
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "some-package"])
  ```
- The remote server has **PyTorch, torchvision, scikit-learn, matplotlib,
  pandas, openpyxl, numpy** pre-installed.
- For **regression with tabular data**, place your CSV/Excel file in the
  Train Data folder. It will be uploaded to `/workspace/data/`.
- Save plots as `.png` and tables as `.xlsx` into `OUTPUT_DIR`.
- Use `print()` — all stdout/stderr is streamed live to the GUI log.
