<p align="center">
  <strong>SCOUT</strong> — Smart Cloud Orchestrated Unified Trainer
</p>

<p align="center">
  <em>Configure. Deploy. Harvest.</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#business-cases">Business Cases</a> •
  <a href="#license">License</a>
</p>

---

## Why SCOUT?

Training deep learning models shouldn't require a DevOps degree. SCOUT eliminates the gap between **"I have a dataset"** and **"I have a trained model"** by automating every step of cloud GPU training — from renting the machine to downloading your results.

> **Philosophy: Configure, Deploy, Harvest**
>
> Most ML tools force you to think about infrastructure: SSH tunnels, pip dependencies, CUDA versions, SCP commands, instance cleanup. SCOUT inverts this — you think about your *experiment*, and the infrastructure thinks about itself.
>
> **Configure** your hyperparameters in a GUI, **Deploy** with one click to the cheapest reliable GPU on Vast.ai, and **Harvest** your trained model, metrics, Grad-CAM heatmaps, and Excel reports — all downloaded automatically.

---

## Features

### Core Pipeline
- **One-click orchestration**: Search → Rent → Upload → Train → Download → Done
- **Smart Estimator**: Pre-flight dataset analysis — estimates VRAM needs, training time, and upload time before you spend a penny
- **Dynamic Provisioning**: Selects GPUs by *best value*, not just cheapest — filters by reliability score, network speed, and DL performance benchmarks
- **Live Telemetry**: Real-time Loss/Accuracy charts in the GUI via periodic CSV polling — no W&B dependency
- **Rsync pipeline**: Resume-capable, checksum-verified file transfers via rsync over SSH (falls back to tar+gzip → SFTP automatically)

### Training
- **Classification** (images): Pretrained CNNs — ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt
- **Regression** (tabular CSV/Excel): MLP with configurable target/feature columns
- **17 metrics**: 10 classification (Accuracy, F1, AUC-ROC, Cohen's Kappa…) + 7 regression (RMSE, R², MAPE…)
- **Feature flags**: Early Stopping, Cosine LR Scheduler, Mixup, Label Smoothing, Grad-CAM, Data Augmentation

### AI & Interpretability
- **Automated Grad-CAM**: After training, generates 5 Class Activation Map heatmaps from the test set — shows *what the model is looking at*, not just accuracy numbers
- **Model Checkpointing**: Saves both `best_model.pth` (lowest val loss) and `final_model.pth`
- **REPORT.md**: Auto-generated Markdown report with metrics table, feature flags, and output file manifest
- **Excel export**: `results.xlsx` with Epoch History, Final Metrics, and Predictions sheets

### User Experience
- **Dark-mode GUI** (CustomTkinter) with scrollable config, real-time log streaming
- **Interactive SSH Console** — run ad-hoc commands on the remote server from inside the app
- **Custom training scripts** — bring your own `.py` instead of the built-in `train.py`
- **Stop / Terminate** — pause billing or fully destroy the instance
- **Auto-organize**: Flat image folders sorted into class sub-folders via filename regex

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCOUT Desktop GUI (gui.py)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Config Panel  │  │  Log Console │  │  Live Telemetry Chart  │ │
│  │ (scrollable)  │  │  (Consolas)  │  │  (matplotlib TkAgg)   │ │
│  └──────┬───────┘  └──────────────┘  └────────────────────────┘ │
└─────────┼───────────────────────────────────────────────────────┘
          │
     ┌────▼────────────────────────────────────────────────────┐
     │           Orchestrator (orchestrator.py)                │
     │  Step 0: Smart Estimator (estimator.py)                 │
     │  Step 1: Search + Rent → Dynamic Provisioning           │
     │  Step 2: SSH Connect                                    │
     │  Step 3: Setup Environment (pip install)                │
     │  Step 4: Upload Data (rsync → tar+gz → SFTP)           │
     │  Step 5: Run Training + Telemetry Polling               │
     │  Step 6: Download Results (Harvest)                     │
     └────┬──────────────┬─────────────────────────────────────┘
          │              │
   ┌──────▼──────┐ ┌────▼──────┐
   │  Vast.ai API │ │    SSH    │
   │ (vast_api.py)│ │(ssh_mgr) │
   │  SDK wrapper │ │ Paramiko  │
   └──────────────┘ └────┬─────┘
                         │
                ┌────────▼────────────────────────────────┐
                │      Remote GPU Instance (Vast.ai)      │
                │  ┌──────────────────────────────────┐   │
                │  │  train.py                        │   │
                │  │  • Classification (CNN) or       │   │
                │  │    Regression (MLP)              │   │
                │  │  • Writes telemetry.csv (live)   │   │
                │  │  • Saves best_model.pth          │   │
                │  │  • Generates Grad-CAM heatmaps   │   │
                │  │  • Exports results.xlsx          │   │
                │  │  • Creates REPORT.md             │   │
                │  └──────────────────────────────────┘   │
                └─────────────────────────────────────────┘
```

### Module Responsibilities

| File | Role |
|------|------|
| `main.py` | Entry point, file logging (`orchestrator.log`) |
| `gui.py` | CustomTkinter GUI — config panel, log viewer, live chart, SSH console |
| `config.py` | `ExperimentConfig` dataclass + `build_train_command()` |
| `orchestrator.py` | 7-step pipeline controller with telemetry polling |
| `estimator.py` | Pre-flight dataset profiling and resource estimation |
| `vast_api.py` | Vast.ai SDK wrapper — search (scored), create, stop, destroy |
| `ssh_manager.py` | Paramiko SSH/SFTP + rsync support, tar-based transfers |
| `train.py` | Remote training script — classification CNN + regression MLP |

---

## Quick Start

### Prerequisites
- Python 3.10+
- A [Vast.ai](https://vast.ai) account with credit
- SSH key pair (public key uploaded to Vast.ai)

### Install

```bash
pip install -r requirements.txt
```

### Generate SSH key (if needed)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_key
```

Upload `vast_key.pub` to [Vast.ai → Account → SSH Keys](https://vast.ai/account).

### Run

```bash
python main.py
```

Fill in the config panel and click **▶ Start Pipeline**.

---

## Data Format

### Classification — ImageFolder or Flat

**Option A** — pre-organized:
```
data/
├── CAT/
│   ├── img001.png
│   └── img002.png
└── DOG/
    └── img001.png
```

**Option B** — flat folder (auto-organized):
```
data/
├── CAT_001.png   →  class CAT
├── DOG_001.png   →  class DOG
```

Class label extracted via `^([A-Za-z]+)` regex — alphabetic prefix becomes the class name.

### Regression — CSV / Excel

```
data/
└── dataset.csv
```

Set **Target Column** and optionally **Feature Columns** in the GUI.

---

## Output Files

| File | Description |
|------|-------------|
| `best_model.pth` | Checkpoint with lowest validation loss |
| `final_model.pth` | Model after all epochs |
| `REPORT.md` | Auto-generated Markdown training report |
| `results.xlsx` | Epoch history + final metrics + predictions |
| `telemetry.csv` | Per-epoch live metrics (used by GUI chart) |
| `loss_accuracy.png` | Loss & accuracy curves |
| `metrics.png` | All selected metrics over epochs |
| `confusion_matrix.png` | Confusion matrix (classification) |
| `roc_curve.png` | ROC curves (classification) |
| `gradcam_*.png` | Grad-CAM heatmaps (classification) |
| `pred_vs_actual.png` | Scatter plot (regression) |
| `residuals.png` | Residual plot (regression) |

---

## Benchmarks

### Pipeline Speed: SCOUT vs Manual Setup

| Step | Manual (first time) | SCOUT |
|------|-------------------|-------|
| Find & compare GPU offers | 5–15 min | **instant** (auto-scored) |
| Rent instance + wait for boot | 3–5 min | **~2 min** (auto-polled) |
| SSH in, install deps, mkdir | 5–10 min | **~1 min** (scripted) |
| Upload dataset (1 GB, SCP) | 5–10 min | **~2 min** (rsync/tar) |
| Write training script | 30–120 min | **0** (built-in or custom) |
| Monitor training | constant attention | **live chart, auto-log** |
| Download results | 2–5 min | **~1 min** (auto-harvest) |
| Clean up instance | often forgotten 💸 | **one-click terminate** |
| **Total** | **50–170 min + billing risk** | **~6 min + 0 risk** |

### Training Throughput (ResNet-50, ImageNet-scale)

| GPU Tier | VRAM | ~Images/sec | 10k images × 50 epochs |
|----------|------|-------------|------------------------|
| Low (RTX 3060) | 12 GB | ~120 | ~70 min |
| Mid (RTX 3090) | 24 GB | ~300 | ~28 min |
| High (A100) | 80 GB | ~700 | ~12 min |

*Estimates from Smart Estimator — actual times vary by image resolution and augmentation.*

---

## Business Cases

### 🎓 Academic Research — Fast Prototyping

> *"I just want to test if DenseNet-121 beats ResNet-50 on my dataset. I don't want to spend 3 hours setting up a server."*

SCOUT lets researchers focus on **hypotheses, not infrastructure**. A PhD student can:
1. Compare 4 architectures in one afternoon (change model, click Start, repeat)
2. Get publication-ready plots and Grad-CAM heatmaps automatically
3. Pay only for actual compute (~$0.20–$0.50 per short experiment)

**Value**: Eliminates the DevOps tax on research time. 10× faster iteration cycles.

### 🚀 MVP for Startups — Proof of Concept on a Budget

> *"We need to prove our computer vision idea works before raising seed funding."*

SCOUT as a rapid prototyping tool:
- **No AWS/GCP contracts** — pay-per-minute on Vast.ai (10–50× cheaper than hyperscalers)
- **No ML engineer needed** — a product manager can run experiments via GUI
- Smart Estimator prevents wasting money on oversized GPUs
- Excel + REPORT.md outputs are investor-ready

**Value**: Validate ML feasibility for <$5 instead of $500+ on managed services.

### 📚 AI Education — Bridge the Infrastructure Gap

> *"My students understand backpropagation but can't SSH into a server."*

SCOUT as a teaching tool:
- Students change hyperparameters (LR, batch size, epochs) and **see the impact** in live charts
- Grad-CAM shows *what the model sees* — makes CNN internals tangible
- No terminal knowledge required — everything is GUI-driven
- Metric comparison teaches evaluation beyond just accuracy

**Value**: Students learn ML concepts instead of fighting Linux commands.

---

## Security

- API keys and SSH keys are **entered at runtime only** — never persisted to disk
- `orchestrator.log` is gitignored (may contain remote IPs)
- SSH key files are gitignored
- SSH connections use key-based auth with `StrictHostKeyChecking` disabled only for rsync

---

## Custom Scripts

See [CUSTOM_SCRIPT_GUIDE.md](CUSTOM_SCRIPT_GUIDE.md) for remote paths, pre-installed packages, and a minimal template.

---

## License

MIT

---

<p align="center">
  <strong>SCOUT</strong> — <em>Because infrastructure should be invisible.</em>
</p>
