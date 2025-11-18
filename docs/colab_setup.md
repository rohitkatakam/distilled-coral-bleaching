# Google Colab Training Guide

This guide provides step-by-step instructions for training models in Google Colab. Follow each section carefully.

## Prerequisites

Before starting, ensure you have:
- Google Drive with the coral bleaching dataset uploaded (see [Data Setup](#data-setup-verification) below)
- Weights & Biases (W&B) account (free tier works fine)
- GitHub repository access

---

## Part 1: Initial Setup

### Step 1: Create New Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > New notebook**
3. Rename it to `coral-bleaching-training.ipynb`
4. **Important**: Change runtime to GPU
   - Click **Runtime > Change runtime type**
   - Set **Hardware accelerator** to **GPU** (T4 or better)
   - Click **Save**

### Step 2: Clone Repository

Run this in a code cell:

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/distilled-coral-bleaching.git
%cd distilled-coral-bleaching

# Verify we're in the right directory
!pwd
!ls -la
```

**Expected output**: You should see files like `train_teacher.py`, `configs/`, `models/`, etc.

### Step 3: Install Dependencies

```python
# Install required packages (GPU-enabled PyTorch)
!pip install -q -r requirements-colab.txt

# Verify PyTorch can see GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Expected output**:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA device: Tesla T4 (or similar)
```

If CUDA is not available, check that you selected GPU runtime in Step 1.

---

## Part 2: Google Drive Setup

### Step 4: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Action required**: Click the link, authenticate with your Google account, and paste the authorization code.

### Step 5: Verify Drive Structure

**CRITICAL**: Before proceeding, verify your Google Drive has the correct structure:

```python
import os

# Define base path
DRIVE_BASE = "/content/drive/MyDrive/coral-bleaching"

# Check directory structure
print("Checking Google Drive structure...")
print(f"Base directory exists: {os.path.exists(DRIVE_BASE)}")
print(f"Data directory exists: {os.path.exists(f'{DRIVE_BASE}/data/raw')}")
print(f"Bleached images exist: {os.path.exists(f'{DRIVE_BASE}/data/raw/bleached')}")
print(f"Healthy images exist: {os.path.exists(f'{DRIVE_BASE}/data/raw/healthy')}")
print(f"Checkpoints directory exists: {os.path.exists(f'{DRIVE_BASE}/checkpoints')}")

# Count images
if os.path.exists(f'{DRIVE_BASE}/data/raw/bleached'):
    bleached_count = len(os.listdir(f'{DRIVE_BASE}/data/raw/bleached'))
    print(f"\nBleached images: {bleached_count}")
else:
    print("\n⚠️ WARNING: Bleached directory not found!")

if os.path.exists(f'{DRIVE_BASE}/data/raw/healthy'):
    healthy_count = len(os.listdir(f'{DRIVE_BASE}/data/raw/healthy'))
    print(f"Healthy images: {healthy_count}")
else:
    print("⚠️ WARNING: Healthy directory not found!")
```

**Expected output**:
```
Checking Google Drive structure...
Base directory exists: True
Data directory exists: True
Bleached images exist: True
Healthy images exist: True
Checkpoints directory exists: True

Bleached images: 485
Healthy images: 438
```

**If directories are missing**, you need to create them and upload images:

1. In Google Drive, create folder: `coral-bleaching/`
2. Inside it, create: `data/raw/bleached/` and `data/raw/healthy/`
3. Upload all coral images to respective folders
4. Create empty folder: `checkpoints/`

### Step 6: Create Symbolic Links (Optional)

This makes paths work the same as locally:

```python
# Create symbolic link from repo to Drive data
!mkdir -p /content/distilled-coral-bleaching/data
!ln -sf /content/drive/MyDrive/coral-bleaching/data/raw /content/distilled-coral-bleaching/data/raw

# Verify symbolic link
!ls -la /content/distilled-coral-bleaching/data/
```

---

## Part 3: Weights & Biases Setup

### Step 7: W&B Authentication

```python
import wandb

# Login to W&B
wandb.login()
```

**Action required**:
1. Click the link that appears
2. Copy your W&B API key from the website
3. Paste it into the input field

**Alternative - Using API key directly**:
```python
# If you have your API key
import wandb
wandb.login(key="YOUR_WANDB_API_KEY_HERE")
```

---

## Part 4: Teacher Training

### Step 8: Review Training Configuration

Check the default configuration:

```python
import yaml

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Training Configuration:")
print(f"  Epochs: {config['training']['epochs']}")
print(f"  Batch size: {config['training']['batch_size']}")
print(f"  Learning rate: {config['training']['learning_rate']}")
print(f"  Optimizer: {config['training']['optimizer']}")
print(f"  Scheduler: {config['training']['scheduler']}")
```

### Step 9: Start Teacher Training

**Full training (50 epochs, ~2-3 hours on T4 GPU)**:

```python
# Train teacher model with default settings
!python train_teacher.py \
    --config configs/config.yaml \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/teacher \
    --wandb-project coral-bleaching \
    --wandb-mode online \
    --device cuda
```

**Quick test run (2 epochs, ~5 minutes)**:

```python
# Test with 2 epochs to verify everything works
!python train_teacher.py \
    --config configs/config.yaml \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/teacher \
    --epochs 2 \
    --batch-size 16 \
    --wandb-project coral-bleaching-test \
    --wandb-mode online \
    --device cuda
```

### Step 10: Monitor Training

**In the notebook output**, you'll see:
- Device being used (should be `cuda`)
- Number of batches per epoch
- Training progress with loss and accuracy
- Validation metrics after each epoch

**In Weights & Biases**:
1. Click the W&B link in the output (or go to wandb.ai)
2. Navigate to your project (`coral-bleaching`)
3. Click on the active run
4. Monitor:
   - `train/loss` and `train/accuracy` - training metrics
   - `val/loss`, `val/accuracy`, `val/precision`, `val/recall`, `val/f1` - validation metrics
   - `learning_rate` - LR schedule

**Training will automatically**:
- Save best model to Drive: `.../checkpoints/teacher/best_model.pth`
- Save latest model to Drive: `.../checkpoints/teacher/latest_model.pth`
- Stop early if validation accuracy doesn't improve for 10 epochs
- Log all metrics to W&B

---

## Part 5: Checkpoint Management

### Step 11: Verify Checkpoints Saved

After training completes:

```python
import os

checkpoint_dir = "/content/drive/MyDrive/coral-bleaching/checkpoints/teacher"

print("Saved checkpoints:")
for filename in os.listdir(checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, filename)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  {filename}: {size_mb:.1f} MB")
```

**Expected output**:
```
Saved checkpoints:
  best_model.pth: 97.8 MB
  latest_model.pth: 97.8 MB
```

### Step 12: Inspect Checkpoint Contents

```python
import torch

# Load checkpoint
checkpoint_path = "/content/drive/MyDrive/coral-bleaching/checkpoints/teacher/best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint contents:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
print(f"  Metrics: {checkpoint['metrics']}")
print(f"  Keys: {list(checkpoint.keys())}")
```

---

## Part 6: Resuming Training

### Step 13: Resume from Checkpoint (if interrupted)

If training was interrupted, you can resume:

```python
!python train_teacher.py \
    --config configs/config.yaml \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/teacher \
    --resume /content/drive/MyDrive/coral-bleaching/checkpoints/teacher/latest_model.pth \
    --wandb-project coral-bleaching \
    --wandb-mode online \
    --device cuda
```

The training will:
- Load model weights, optimizer state, and scheduler state
- Continue from the next epoch
- Maintain the best validation accuracy tracker

---

## Part 7: Downloading Results (Optional)

### Step 14: Download to Local Machine

**Option A: Download from Google Drive web interface**
1. Go to Google Drive
2. Navigate to `coral-bleaching/checkpoints/teacher/`
3. Right-click `best_model.pth` → Download

**Option B: Download directly in Colab**
```python
from google.colab import files

# Download best model
files.download('/content/drive/MyDrive/coral-bleaching/checkpoints/teacher/best_model.pth')
```

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size

```python
!python train_teacher.py \
    --batch-size 16 \
    --device cuda \
    ...
```

### Issue: Dataset not found

**Error**: `FileNotFoundError: data/splits/train.csv`

**Solution**: The split CSV files should be in the repo. Verify:

```bash
!ls -la data/splits/
```

If missing, they should be in Git. Pull the latest version:

```bash
!git pull origin main
```

### Issue: Images not loading

**Error**: `FileNotFoundError: data/raw/bleached/...`

**Solution**:
1. Check Drive paths (Step 5)
2. Ensure images are uploaded to Drive
3. Verify symbolic links (Step 6) or update paths in config

### Issue: W&B not logging

**Solution**: Check W&B mode:

```python
# Use offline mode if online doesn't work
!python train_teacher.py --wandb-mode offline ...
```

Logs will be saved locally and can be synced later:

```bash
!wandb sync wandb/offline-run-...
```

### Issue: Colab disconnects during training

**Prevention**:
- Keep the browser tab open
- Checkpoints are saved every epoch to Drive, so you can resume

**Recovery**:
- Restart from Step 9, but use `--resume` (Step 13)

### Issue: GPU not available

**Solution**:
1. Runtime > Change runtime type > GPU
2. Or use CPU (much slower):

```python
!python train_teacher.py --device cpu --batch-size 8 ...
```

---

## Training Hyperparameter Reference

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/config.yaml` | Path to config file |
| `--output-dir` | `checkpoints/teacher` | Checkpoint save directory |
| `--resume` | None | Path to checkpoint to resume from |
| `--epochs` | 50 (from config) | Number of training epochs |
| `--batch-size` | 32 (from config) | Batch size |
| `--lr` | 0.001 (from config) | Learning rate |
| `--device` | cuda (from config) | Device (cuda/cpu) |
| `--wandb-project` | `coral-bleaching` | W&B project name |
| `--wandb-mode` | `online` | W&B mode (online/offline/disabled) |
| `--no-pretrained` | False | Don't use pretrained ImageNet weights |

### Example Configurations

**Fast training (for testing)**:
```bash
--epochs 5 --batch-size 16
```

**High-quality training**:
```bash
--epochs 100 --batch-size 32 --lr 0.0001
```

**Training without pretrained weights**:
```bash
--no-pretrained --epochs 100
```

---

## Expected Training Times (on T4 GPU)

- **Full training (50 epochs)**: ~2-3 hours
- **Quick test (2 epochs)**: ~5 minutes
- **Single epoch**: ~3-4 minutes

---

## Next Steps After Training

1. **Verify checkpoint saved to Drive** (Step 11)
2. **Check W&B dashboard** for training curves
3. **Download checkpoint** to local machine (Step 14)
4. **Run evaluation locally**:
   - Use `scripts/evaluate.py` to compute test metrics
   - Create analysis scripts in `scripts/` for data exploration and visualization
   - See main README for evaluation workflow details

---

## Questions or Issues?

- Check the [main README](../README.md) for project overview
- Review error messages carefully - they usually indicate the issue
- Ensure Google Drive structure matches Part 2
- For training issues, check the troubleshooting section above
