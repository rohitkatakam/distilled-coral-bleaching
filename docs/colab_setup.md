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

## Part 8: Student Baseline Training (Phase 3)

**Prerequisites**: Ensure teacher training (Part 4) is complete before starting student baseline.

### Step 15: Review Student Model Configuration

The student model uses a lightweight MobileNetV3-Small architecture (~1.5M parameters vs ~23.5M for teacher):

```python
import yaml

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Student Model Configuration:")
print(f"  Architecture: {config['model']['student']['name']}")
print(f"  Pretrained: {config['model']['student']['pretrained']}")
print(f"  Parameters: ~1.5M (vs ~23.5M for teacher)")
```

### Step 16: Start Student Baseline Training

**Full training (50 epochs, ~1-1.5 hours on T4 GPU)**:

```python
# Train student baseline model (no distillation)
!python train_student_baseline.py \
    --config configs/config.yaml \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/student_baseline \
    --wandb-project coral-bleaching \
    --wandb-mode online \
    --device cuda
```

**Quick test run (2 epochs, ~3 minutes)**:

```python
# Test with 2 epochs to verify everything works
!python train_student_baseline.py \
    --config configs/config.yaml \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/student_baseline \
    --epochs 2 \
    --batch-size 16 \
    --wandb-project coral-bleaching-test \
    --wandb-mode online \
    --device cuda
```

### Step 17: Monitor Student Training

Monitoring works the same as teacher training (Step 10):
- Check W&B dashboard for metrics
- Student baseline expected performance: ~72-73% test accuracy (vs ~77-78% for teacher)
- Checkpoints auto-save to Drive: `.../checkpoints/student_baseline/best_model.pth`

### Step 18: Verify Student Checkpoint

After training completes:

```python
import os
import torch

checkpoint_dir = "/content/drive/MyDrive/coral-bleaching/checkpoints/student_baseline"

print("Student baseline checkpoints:")
for filename in os.listdir(checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, filename)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  {filename}: {size_mb:.1f} MB")

# Load and inspect checkpoint
checkpoint = torch.load(f"{checkpoint_dir}/best_model.pth", map_location='cpu')
print(f"\nCheckpoint details:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Best val acc: {checkpoint.get('best_val_acc', 0) * 100:.2f}%")
```

**Expected output**:
```
Student baseline checkpoints:
  best_model.pth: ~10 MB (much smaller than teacher's ~97 MB)
  latest_model.pth: ~10 MB

Checkpoint details:
  Epoch: 15-20 (early stopping expected)
  Best val acc: 72-75%
```

### Expected Performance Gap

The student baseline should perform slightly worse than the teacher:
- **Teacher**: ~77-78% test accuracy
- **Student Baseline**: ~72-73% test accuracy (~5% gap)
- **Goal for Phase 4**: Use knowledge distillation to close this gap

### Student Training Times (on T4 GPU)

- **Full training (50 epochs)**: ~1-1.5 hours (faster than teacher due to smaller model)
- **Quick test (2 epochs)**: ~3 minutes
- **Single epoch**: ~2 minutes

---

## Part 9: Knowledge Distillation Training (Phase 4)

**Prerequisites**: Both teacher (Part 4) and student baseline (Part 8) should be complete. Distillation requires a trained teacher checkpoint.

### Step 19: Review Distillation Configuration

Knowledge distillation uses two hyperparameters to blend teacher knowledge with hard labels:

```python
import yaml

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Knowledge Distillation Configuration:")
print(f"  Temperature (T): {config['model']['distillation']['temperature']}")
print(f"  Alpha (α): {config['model']['distillation']['alpha']}")
print(f"  Hard weight (1-α): {1 - config['model']['distillation']['alpha']}")
print("\nWhat these mean:")
print(f"  - Temperature: Softens probability distributions (higher = softer)")
print(f"  - Alpha: Weight for distillation loss (0.7 = 70% KD, 30% hard labels)")
```

**Expected output**:
```
Knowledge Distillation Configuration:
  Temperature (T): 4.0
  Alpha (α): 0.7
  Hard weight (1-α): 0.3

What these mean:
  - Temperature: Softens probability distributions (higher = softer)
  - Alpha: Weight for distillation loss (0.7 = 70% KD, 30% hard labels)
```

### Step 20: Verify Teacher Checkpoint Exists

Before starting KD training, ensure the teacher checkpoint is available:

```python
import os

teacher_checkpoint = "/content/drive/MyDrive/coral-bleaching/checkpoints/teacher/best_model.pth"

if os.path.exists(teacher_checkpoint):
    size_mb = os.path.getsize(teacher_checkpoint) / (1024 * 1024)
    print(f"✓ Teacher checkpoint found: {size_mb:.1f} MB")
else:
    print("✗ ERROR: Teacher checkpoint not found!")
    print("  Please complete Part 4 (Teacher Training) first.")
```

### Step 21: Start Knowledge Distillation Training

Choose your training approach based on your goals:

#### Option A: Quick Start (Single Configuration)

Train with default hyperparameters (T=4.0, α=0.7) to quickly verify KD works:

**Full training (50 epochs, ~1.5-2 hours on T4 GPU)**:

```python
# Train student with knowledge distillation (default hyperparameters)
!python train_student_kd.py \
    --config configs/config.yaml \
    --teacher-checkpoint /content/drive/MyDrive/coral-bleaching/checkpoints/teacher/best_model.pth \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/student_kd \
    --wandb-project coral-bleaching \
    --wandb-mode online \
    --device cuda
```

**Quick test run (2 epochs, ~4 minutes)**:

```python
# Test with 2 epochs to verify everything works
!python train_student_kd.py \
    --config configs/config.yaml \
    --teacher-checkpoint /content/drive/MyDrive/coral-bleaching/checkpoints/teacher/best_model.pth \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/student_kd \
    --epochs 2 \
    --batch-size 16 \
    --wandb-project coral-bleaching-test \
    --wandb-mode online \
    --device cuda
```

---

#### Option B: Strategic Hyperparameter Exploration (Recommended for Paper)

For comprehensive ablation studies, train **4 strategic configurations** that explore the temperature/alpha space. This provides robust evidence for your paper and helps identify optimal hyperparameters.

**Rationale:**
- **T (Temperature)**: Controls softness of probability distributions (1=hard, higher=softer)
- **α (Alpha)**: Balances teacher guidance vs hard labels (0=only labels, 1=only teacher)
- **Strategy**: Sample key points in hyperparameter space to understand sensitivity
- **Cost**: ~6-8 hours total training time (feasible due to fast training)
- **Benefit**: Multiple data points for Phase 4 comparison, sets up Phase 5 ablation analysis

**Before starting:** Check if you've already trained the default configuration:

```python
import os

# Check if default config already trained
default_checkpoint = "/content/drive/MyDrive/coral-bleaching/checkpoints/student_kd/best_model_t4.0_a0.7.pth"
if os.path.exists(default_checkpoint):
    print("✓ Default configuration (T=4.0, α=0.7) already trained")
    print("  You can skip Configuration 1 below and reuse this checkpoint")
else:
    print("→ Need to train all 4 configurations")
```

---

**Configuration 1: Default (T=4.0, α=0.7) - Balanced approach** [~1.5-2 hours]

Moderate temperature with teacher-focused weighting (70% KD, 30% hard labels).

```python
# Config 1: Default - balanced distillation
!python train_student_kd.py \
    --config configs/config.yaml \
    --teacher-checkpoint /content/drive/MyDrive/coral-bleaching/checkpoints/teacher/best_model.pth \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/student_kd \
    --temperature 4.0 \
    --alpha 0.7 \
    --wandb-project coral-bleaching \
    --wandb-mode online \
    --device cuda
```

**Checkpoint**: `best_model_t4.0_a0.7.pth` | **W&B run**: `student-kd-t4.0-a0.7`

---

**Configuration 2: Conservative (T=2.0, α=0.5) - Sharper distributions, balanced loss** [~1.5-2 hours]

Lower temperature (less softening) with equal weight on teacher and labels.

```python
# Config 2: Conservative - less aggressive distillation
!python train_student_kd.py \
    --config configs/config.yaml \
    --teacher-checkpoint /content/drive/MyDrive/coral-bleaching/checkpoints/teacher/best_model.pth \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/student_kd \
    --temperature 2.0 \
    --alpha 0.5 \
    --wandb-project coral-bleaching \
    --wandb-mode online \
    --device cuda
```

**Checkpoint**: `best_model_t2.0_a0.5.pth` | **W&B run**: `student-kd-t2.0-a0.5`
**Hypothesis**: Less aggressive distillation may stay closer to baseline performance.

---

**Configuration 3: Aggressive (T=8.0, α=0.9) - Very soft distributions, teacher-dominant** [~1.5-2 hours]

High temperature (maximum softening) with heavy teacher weighting (90% KD, 10% hard labels).

```python
# Config 3: Aggressive - maximum knowledge transfer
!python train_student_kd.py \
    --config configs/config.yaml \
    --teacher-checkpoint /content/drive/MyDrive/coral-bleaching/checkpoints/teacher/best_model.pth \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/student_kd \
    --temperature 8.0 \
    --alpha 0.9 \
    --wandb-project coral-bleaching \
    --wandb-mode online \
    --device cuda
```

**Checkpoint**: `best_model_t8.0_a0.9.pth` | **W&B run**: `student-kd-t8.0-a0.9`
**Hypothesis**: Maximum soft target transfer, best calibration, may achieve highest accuracy.

---

**Configuration 4: Label-Focused (T=4.0, α=0.3) - Moderate softening, label-dominant** [~1.5-2 hours]

Same temperature as default but prioritizes hard labels (30% KD, 70% hard labels).

```python
# Config 4: Label-focused - tests alpha sensitivity
!python train_student_kd.py \
    --config configs/config.yaml \
    --teacher-checkpoint /content/drive/MyDrive/coral-bleaching/checkpoints/teacher/best_model.pth \
    --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/student_kd \
    --temperature 4.0 \
    --alpha 0.3 \
    --wandb-project coral-bleaching \
    --wandb-mode online \
    --device cuda
```

**Checkpoint**: `best_model_t4.0_a0.3.pth` | **W&B run**: `student-kd-t4.0-a0.3`
**Hypothesis**: Closer to baseline behavior, isolates alpha sensitivity at fixed temperature.

---

**Tips for Sequential Training:**

- **Run one at a time** to avoid memory issues
- **Monitor W&B** between runs to verify completion
- All checkpoints save to same directory (`student_kd/`) with unique filenames
- **Total time**: ~6-8 hours (can split across multiple Colab sessions if needed)
- **Keep browser tab open** during training (or enable Colab Pro background execution)
- Each configuration takes ~10-15 minutes per epoch, with early stopping around 12-20 epochs

**Verification After All 4 Runs:**

```python
import os

checkpoint_dir = "/content/drive/MyDrive/coral-bleaching/checkpoints/student_kd"

print("All KD checkpoints:")
expected_files = [
    "best_model_t4.0_a0.7.pth",
    "best_model_t2.0_a0.5.pth",
    "best_model_t8.0_a0.9.pth",
    "best_model_t4.0_a0.3.pth"
]

for filename in expected_files:
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ {filename}: {size_mb:.1f} MB")
    else:
        print(f"  ✗ {filename}: MISSING")
```

**Expected output:**
```
All KD checkpoints:
  ✓ best_model_t4.0_a0.7.pth: ~10 MB
  ✓ best_model_t2.0_a0.5.pth: ~10 MB
  ✓ best_model_t8.0_a0.9.pth: ~10 MB
  ✓ best_model_t4.0_a0.3.pth: ~10 MB
```

**W&B Dashboard Organization:**

You'll see 4 separate runs in your `coral-bleaching` project:
- `student-kd-t4.0-a0.7` (default)
- `student-kd-t2.0-a0.5` (conservative)
- `student-kd-t8.0-a0.9` (aggressive)
- `student-kd-t4.0-a0.3` (label-focused)

**To compare runs side-by-side in W&B:**
1. Go to your project dashboard
2. Select all 4 KD runs (use checkboxes)
3. Click "Compare runs" to see metrics overlaid
4. Useful comparisons: validation accuracy curves, loss components (kd_loss vs hard_loss)

**Next Steps (Phase 5 Analysis):**
After completing all 4 training runs:
1. Download all 4 checkpoints to local machine
2. Run `scripts/evaluate.py` on each model to get test metrics
3. Create `scripts/analyze_ablations.py` for comprehensive comparison
4. Generate ablation figures for paper (temperature sensitivity, alpha sensitivity)

---

### Step 22: Monitor KD Training

**Key difference from baseline**: KD training logs **three loss components**:
- `train/loss`: Total loss (combined)
- `train/kd_loss`: Knowledge distillation loss (teacher guidance)
- `train/hard_loss`: Cross-entropy loss (hard labels)

**In the notebook output**, you'll see:
```
Loading teacher model from .../teacher/best_model.pth...
Teacher loaded successfully (params: 23,528,522)
Teacher frozen: all parameters have requires_grad=False

Epoch [1] Batch [10/21] Loss: 1.2345 (KD: 0.8901, Hard: 0.3444) Acc: 65.62%
```

**In Weights & Biases**:
1. Navigate to your run in W&B dashboard
2. Monitor these metrics:
   - `train/kd_loss` and `train/hard_loss` - loss components
   - `val/loss` (CE), `val/kd_loss`, `val/hard_loss` - validation losses
   - `val/accuracy` - student performance (should improve over baseline)
3. Compare KD run to baseline student run:
   - KD student should achieve higher validation accuracy
   - Better calibration (probability distributions closer to teacher)

**Training will automatically**:
- Load teacher in eval mode (frozen weights)
- Save best model to Drive with hyperparameters in filename
- Stop early if validation CE loss doesn't improve for 10 epochs

### Step 23: Verify KD Checkpoint Saved

After training completes:

```python
import os
import torch

checkpoint_dir = "/content/drive/MyDrive/coral-bleaching/checkpoints/student_kd"

print("Knowledge distillation checkpoints:")
for filename in os.listdir(checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, filename)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  {filename}: {size_mb:.1f} MB")

# Load and inspect checkpoint
best_checkpoint = f"{checkpoint_dir}/best_model_t4.0_a0.7.pth"
checkpoint = torch.load(best_checkpoint, map_location='cpu')
print(f"\nCheckpoint details:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Best val acc: {checkpoint.get('best_val_acc', 0) * 100:.2f}%")
print(f"  Temperature: {checkpoint.get('temperature', 'N/A')}")
print(f"  Alpha: {checkpoint.get('alpha', 'N/A')}")
print(f"  Teacher checkpoint: {checkpoint.get('teacher_checkpoint', 'N/A')}")
```

**Expected output**:
```
Knowledge distillation checkpoints:
  best_model_t4.0_a0.7.pth: ~10 MB
  latest_model.pth: ~10 MB

Checkpoint details:
  Epoch: 12-18 (early stopping expected)
  Best val acc: 75-77%
  Temperature: 4.0
  Alpha: 0.7
  Teacher checkpoint: .../teacher/best_model.pth
```

**Note**: The checkpoint filename includes T and α to track hyperparameters for Phase 5 ablation studies.

### Step 24: Expected Performance Comparison

Three-way comparison of all models (based on default KD configuration T=4.0, α=0.7):

| Model | Test Accuracy | Parameters | Disk Size | Training Time |
|-------|---------------|------------|-----------|---------------|
| Teacher | ~77-78% | 23.5M | ~97 MB | ~2-3 hours |
| Student Baseline | ~72-73% | 1.5M | ~10 MB | ~1-1.5 hours |
| **Student + KD** | **~75-77%** | **1.5M** | **~10 MB** | **~1.5-2 hours** |

**Key insights**:
- **Performance**: KD closes 60-80% of the teacher-student gap
- **Efficiency**: Same model size as baseline, but better accuracy
- **Calibration**: KD student produces better-calibrated probabilities (matches teacher confidence)

**Note on Strategic Sampling (Option B)**: If you trained all 4 configurations, performance may vary across hyperparameter choices. The strategic sampling approach provides:
- Multiple evidence points for Phase 4 paper results
- Data for Phase 5 ablation analysis (temperature sensitivity, alpha sensitivity curves)
- Identification of optimal hyperparameters for this specific dataset

Actual results will be analyzed locally using `scripts/evaluate.py` and `scripts/analyze_ablations.py`.

### KD Training Times (on T4 GPU)

- **Full training (50 epochs)**: ~1.5-2 hours (slightly slower than baseline due to teacher forward pass)
- **Quick test (2 epochs)**: ~4 minutes
- **Single epoch**: ~3 minutes

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

**Common arguments** (teacher, student baseline, KD):

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

**Knowledge distillation specific** (train_student_kd.py only):

| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher-checkpoint` | **REQUIRED** | Path to trained teacher checkpoint |
| `--temperature` | 4.0 (from config) | Distillation temperature (softens distributions) |
| `--alpha` | 0.7 (from config) | KD loss weight (1-α for hard label weight) |

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

| Training Type | Full (50 epochs) | Quick Test (2 epochs) | Single Epoch |
|--------------|------------------|----------------------|--------------|
| **Teacher** | ~2-3 hours | ~5 minutes | ~3-4 minutes |
| **Student Baseline** | ~1-1.5 hours | ~3 minutes | ~2 minutes |
| **Student + KD** | ~1.5-2 hours | ~4 minutes | ~3 minutes |

**Note**: KD training is slightly slower than baseline due to teacher forward pass, but faster than teacher due to smaller student model.

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
