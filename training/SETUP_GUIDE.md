# Setup Guide: Hybrid Diffusion-Transformer PoC

This guide walks you through setting up and running the Hybrid Diffusion-Transformer PoC on your 2× L40S GPUs.

---

## Quick Start (5 Minutes)

```bash
# 1. Navigate to training directory
cd /home/user/oss_srv/training

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets accelerate tensorboard tqdm

# 3. Run minimal test
python scripts/train_minimal.py

# 4. If test passes, run training
python scripts/train.py --dataset dummy --num-epochs 1
```

**Expected result**: Training starts, loss decreases, no errors

---

## Detailed Setup

### Step 1: Install PyTorch with CUDA 12.8

```bash
# Install PyTorch 2.0+ with CUDA 12.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output**:
```
PyTorch: 2.x.x+cu128
CUDA: True
```

### Step 2: Install Core Dependencies

```bash
# Essential packages
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0
pip install tensorboard>=2.14.0
pip install tqdm

# Optional but recommended
pip install wandb  # For experiment tracking
pip install deepspeed  # For multi-GPU optimization
```

### Step 3: Verify GPU Setup

```bash
# Check GPUs
nvidia-smi

# Run environment test
cd /home/user/oss_srv/training
python test_setup.py
```

**Expected output**:
```
✓ Found 2 GPU(s)
  GPU 0: NVIDIA L40S, 48.0GB VRAM
  GPU 1: NVIDIA L40S, 48.0GB VRAM
✓ Created TimeAdapter with 35,435,520 parameters
✓ All tests passed! Ready to train.
```

### Step 4: Run Minimal Training Test

This validates that everything works before committing to a long training run.

```bash
python scripts/train_minimal.py
```

**What it tests**:
- ✅ Model loads correctly
- ✅ Forward pass works
- ✅ Loss computation works
- ✅ Training step updates parameters
- ✅ Generation works

**Expected output**:
```
TEST 1: Forward Pass
✓ Loading GPT-2...
✓ Creating hybrid model...
✓ Output shape: (1, X, 50257)

TEST 2: Loss Computation
✓ Loss: 7.2345

TEST 3: Training Step
✓ Backward pass completed
  Initial loss: 7.2345
  Final loss:   6.8921
  ✓ Loss decreased! Training is working.

TEST 4: Generation
  Prompt: Once upon a time
  Generated: Once upon a time there was a...

ALL TESTS PASSED! ✅
```

---

## Training Options

### Option 1: Quick Test (Dummy Data)

Test the training loop with dummy data (fastest):

```bash
python scripts/train.py \
    --base-model gpt2 \
    --dataset dummy \
    --batch-size 4 \
    --num-epochs 1 \
    --output-dir checkpoints/test_run
```

**Time**: ~5-10 minutes
**Purpose**: Verify training loop works

### Option 2: Small Model (GPT-2)

Train with real data on a small model (for validation):

```bash
python scripts/train.py \
    --base-model gpt2 \
    --dataset wikipedia \
    --max-samples 10000 \
    --batch-size 4 \
    --num-epochs 3 \
    --output-dir checkpoints/gpt2_test
```

**Time**: ~1-2 hours
**Purpose**: Validate approach before scaling

### Option 3: Medium Model (GPT-2 Large)

Train on larger model (closer to production):

```bash
python scripts/train.py \
    --base-model gpt2-large \
    --dataset wikipedia \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --num-epochs 3 \
    --fp16 \
    --output-dir checkpoints/gpt2_large_v1
```

**Time**: ~4-8 hours
**Purpose**: Production-quality validation

### Option 4: Multi-GPU Training (2× L40S)

Use both GPUs for faster training:

```bash
# Using torchrun (recommended)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --base-model gpt2-large \
    --dataset wikipedia \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --num-epochs 3 \
    --fp16 \
    --output-dir checkpoints/gpt2_large_multigpu
```

**Time**: ~2-4 hours (2× faster)
**Purpose**: Production training

### Option 5: Full Scale (20B Model)

Train on full 20B model (if available):

```bash
# Using GPT-NeoX-20B from EleutherAI
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --base-model EleutherAI/gpt-neox-20b \
    --dataset c4 \
    --streaming \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --num-epochs 1 \
    --fp16 \
    --output-dir checkpoints/gpt_neox_20b_v1
```

**Time**: ~4-6 weeks
**Purpose**: Production PoC

---

## Monitoring Training

### TensorBoard

```bash
# In a separate terminal
tensorboard --logdir logs/ --port 6006

# Open in browser
# http://localhost:6006
```

**What to watch**:
- **Loss**: Should decrease from ~8.0 to ~3.5-4.0
- **Learning rate**: Should stay constant (or follow schedule)
- **GPU utilization**: Should be 70-80%

### Watch GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvtop (if installed)
nvtop
```

**Expected GPU usage**:
- Memory: 30-35GB per GPU (for GPT-2 Large)
- Utilization: 70-80%
- Temperature: <80°C

### Check Logs

```bash
# View training progress
tail -f logs/hybrid_v1/events.out.tfevents.*

# Or check checkpoint directory
ls -lh checkpoints/*/
```

---

## Running Demos

After training completes, test the model:

```bash
# Run all demos
python scripts/demo.py \
    --checkpoint checkpoints/gpt2_large_v1/final \
    --demo all

# Run specific demo
python scripts/demo.py \
    --checkpoint checkpoints/gpt2_large_v1/final \
    --demo editing \
    --text "The quick brown fox jumps over the lazy dog" \
    --edit-positions 3 8

# Custom infilling
python scripts/demo.py \
    --checkpoint checkpoints/gpt2_large_v1/final \
    --demo infilling \
    --masked-text "The capital of [MASK] is Paris, and [MASK] is the capital of Spain"
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python scripts/train.py --batch-size 2

# Increase gradient accumulation
python scripts/train.py --batch-size 2 --gradient-accumulation-steps 8

# Reduce sequence length
python scripts/train.py --max-length 256

# Reduce adapter dimension
python scripts/train.py --adapter-dim 64
```

### Issue 2: Training is Slow

**Solutions**:
```bash
# Enable mixed precision
python scripts/train.py --fp16

# Reduce number of workers
python scripts/train.py --num-workers 2

# Use streaming dataset
python scripts/train.py --dataset c4 --streaming
```

### Issue 3: Loss Not Decreasing

**Checks**:
```python
# Verify base model is frozen
python -c "from models import HybridGPTOSS20B; \
           import torch; \
           from transformers import GPT2LMHeadModel; \
           base = GPT2LMHeadModel.from_pretrained('gpt2'); \
           model = HybridGPTOSS20B(base, freeze_base=True); \
           print('Trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))"
```

**Solutions**:
```bash
# Lower learning rate
python scripts/train.py --learning-rate 5e-5

# Increase training data
python scripts/train.py --num-epochs 5

# Check data quality
python -c "from utils import create_simple_dataloader; \
           from transformers import GPT2Tokenizer; \
           tok = GPT2Tokenizer.from_pretrained('gpt2'); \
           tok.pad_token = tok.eos_token; \
           dl = create_simple_dataloader(['test'], tok); \
           print('Dataloader works!')"
```

### Issue 4: Import Errors

**Error**: `ModuleNotFoundError: No module named 'models'`

**Solution**:
```bash
# Make sure you're in the training directory
cd /home/user/oss_srv/training

# Or set PYTHONPATH
export PYTHONPATH=/home/user/oss_srv/training:$PYTHONPATH
```

### Issue 5: Multi-GPU Not Working

**Error**: `RuntimeError: distributed package not available`

**Solution**:
```bash
# Install distributed training support
pip install torch --upgrade

# Use NCCL backend
export NCCL_DEBUG=INFO
torchrun --nproc_per_node=2 scripts/train.py ...

# Check NCCL is available
python -c "import torch; print('NCCL:', torch.distributed.is_nccl_available())"
```

---

## Expected Results

### After 1 Epoch (GPT-2)
- Loss: ~6.0-7.0
- Time: ~10-30 minutes
- Quality: Model generates coherent short sequences

### After 3 Epochs (GPT-2 Large)
- Loss: ~4.0-5.0
- Time: ~2-4 hours
- Quality: Model edits and fills blanks reasonably well

### After Full Training (20B Model, 4-6 weeks)
- Loss: ~3.5-4.0
- Time: 4-6 weeks
- Quality: Production-ready editing and infilling

---

## Next Steps

Once training is complete:

1. **Test Demos**: Run all 4 demos to validate capabilities
2. **Benchmark Performance**: Measure editing speed and quality
3. **Integrate with TensorRT**: Export model for production inference
4. **Build Applications**: Create practical tools using editing capabilities

---

## Resources

- **Architecture**: `docs/HYBRID_DIFFUSION_ARCHITECTURE.md`
- **Next Steps**: `docs/POC_NEXT_STEPS.md`
- **Training README**: `training/README.md`
- **Code**: `training/models/` and `training/scripts/`

---

## Quick Reference

### Training Commands

```bash
# Minimal test
python scripts/train_minimal.py

# Single GPU
python scripts/train.py --dataset dummy --num-epochs 1

# Multi-GPU
torchrun --nproc_per_node=2 scripts/train.py --dataset wikipedia --num-epochs 3

# Production
torchrun --nproc_per_node=2 scripts/train.py \
    --base-model gpt2-large \
    --dataset c4 --streaming \
    --batch-size 4 --gradient-accumulation-steps 4 \
    --num-epochs 3 --fp16
```

### Demo Commands

```bash
# All demos
python scripts/demo.py --checkpoint checkpoints/final --demo all

# Specific demos
python scripts/demo.py --checkpoint checkpoints/final --demo autoregressive
python scripts/demo.py --checkpoint checkpoints/final --demo editing
python scripts/demo.py --checkpoint checkpoints/final --demo infilling
python scripts/demo.py --checkpoint checkpoints/final --demo parallel
```

### Monitoring Commands

```bash
# TensorBoard
tensorboard --logdir logs/

# GPU usage
watch -n 1 nvidia-smi

# Check checkpoints
ls -lh checkpoints/*/
```

---

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review `docs/POC_NEXT_STEPS.md`
3. Run `python test_setup.py` to diagnose environment
4. Check TensorBoard logs for training issues
5. Open an issue on GitHub

---

**Ready to start?** Run `python scripts/train_minimal.py` to begin! 🚀
