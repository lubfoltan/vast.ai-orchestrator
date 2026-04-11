# Custom Training Script Guide

When you provide a custom script in the app, it is uploaded to the remote server
as `/workspace/train.py` and executed with:

```bash
cd /workspace && python train.py
```

The script receives **no CLI arguments** — use the fixed paths below directly in your code.

---

## Fixed Remote Paths

| Path | Contents |
|------|----------|
| `/workspace/data` | Your uploaded **training data** |
| `/workspace/test_data` | Your uploaded **test data** (empty folder if not provided) |
| `/workspace/output` | **Write all output here** — everything in this folder is downloaded to your local Output Path after training |

---

## Pre-installed Packages

The following packages are available on the remote server without any extra install:

- `torch`, `torchvision`, `timm`
- `scikit-learn`
- `matplotlib`
- `pandas`, `openpyxl`
- `numpy` (pinned `<2` for PyTorch compatibility)
- `pillow`, `tqdm`
- `grad-cam`

---

## Minimal Template

```python
import os

# ── Fixed remote paths ──────────────────────────────────────────────
DATA_DIR   = "/workspace/data"
TEST_DIR   = "/workspace/test_data"
OUTPUT_DIR = "/workspace/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Your training code here ─────────────────────────────────────────
print(f"Training data : {DATA_DIR}")
print(f"Test data     : {TEST_DIR}")
print(f"Output dir    : {OUTPUT_DIR}")

# Example: list files in data dir
for f in sorted(os.listdir(DATA_DIR)):
    print(f"  {f}")

# Save any result to OUTPUT_DIR — it will be downloaded automatically
with open(os.path.join(OUTPUT_DIR, "result.txt"), "w") as f:
    f.write("Training complete!\n")
```

---

## Installing Extra Packages

Add this at the top of your script:

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "some-package"])
```

---

## Saving Results

| Type | Recommended format |
|------|--------------------|
| Metrics / tables | `.xlsx` via `pandas.ExcelWriter` |
| Plots | `.png` via `matplotlib.pyplot.savefig` |
| Model weights | `.pth` via `torch.save` |

All files saved to `OUTPUT_DIR` are downloaded to your local machine at the end of the pipeline.

---

## Tips

- Use `print()` freely — stdout and stderr are streamed live to the GUI log.
- The remote server uses Linux; use `/` path separators (not `\`).
- `os.path.join` works correctly on both Linux and Windows — use it instead of hardcoded separators.
- For image datasets in flat layout, class labels must be embedded in the filename. The built-in auto-organizer uses the regex `^([A-Za-z]+)` to extract the prefix before the first digit or underscore as the class name (e.g. `CAT_001.jpg` → class `CAT`).
