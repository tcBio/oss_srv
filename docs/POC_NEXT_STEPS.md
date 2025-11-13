# Hybrid Diffusion-Transformer PoC: Next Steps

**Status**: ✅ Foundation Complete
**Date**: 2025-11-13
**Goal**: Build working PoC on 2× L40S GPUs in 4-6 weeks

---

## What We've Built

### ✅ Completed

1. **Architecture Design** (`docs/HYBRID_DIFFUSION_ARCHITECTURE.md`)
   - Complete technical specification
   - Dual-mode design (autoregressive + diffusion)
   - Memory-efficient adapter strategy
   - Killer demo concepts

2. **Core Components** (`training/models/`)
   - `time_adapter.py`: FiLM-conditioned adapters (35M params, 0.18% of base)
   - `discrete_diffusion.py`: Token-level diffusion with cosine schedule
   - `hybrid_model.py`: Integrated model with mode switching

3. **Infrastructure** (`training/`)
   - DeepSpeed configuration for 2× L40S
   - Requirements and dependencies
   - Test suite for verification
   - Comprehensive documentation

4. **Feasibility Analyses** (`docs/`)
   - Full diffusion conversion feasibility
   - GPU cloud cost analysis
   - L40S training feasibility
   - Hybrid architecture design

### 📊 Key Metrics

- **Base Model**: 20B parameters (frozen)
- **Adapters**: 35M parameters (0.18%, trainable)
- **Memory**: 30-35GB per GPU (fits in 48GB L40S)
- **Free Capacity**: 13-18GB per GPU for parallel tasks
- **Training Time**: 4-6 weeks on 2× L40S

---

## Immediate Next Steps (Week 1)

### Step 1: Environment Setup

```bash
cd /home/user/oss_srv/training

# Install dependencies
pip install -r requirements.txt

# Verify CUDA and GPUs
nvidia-smi

# Run setup test
python test_setup.py
```

**Expected Output**:
```
✓ Found 2 GPU(s)
  GPU 0: NVIDIA L40S, 48.0GB VRAM
  GPU 1: NVIDIA L40S, 48.0GB VRAM
✓ Created TimeAdapter with 1,476,480 parameters
✓ Created MultiLayerTimeAdapter with 35,435,520 parameters
✓ All tests passed! Ready to train.
```

### Step 2: Convert GPT-OSS-20B to PyTorch

**Challenge**: Current model is in TensorRT format (.engine file)

**Options**:

#### Option A: Export from TensorRT ⚠️ **(May be difficult)**
```bash
# This requires the original PyTorch weights or ONNX model
# TensorRT .engine files are binary and hard to export from
```

#### Option B: Use HuggingFace Pretrained Model ✅ **(Recommended for PoC)**
```python
# Use a similar 20B model as base (e.g., GPT-NeoX-20B)
from transformers import GPTNeoXForCausalLM

base_model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/gpt-neox-20b",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

#### Option C: Train from Scratch with Smaller Model **(For Testing)**
```python
# Use GPT-2 or smaller model to validate approach
from transformers import GPT2LMHeadModel

base_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
```

**Recommendation**: Start with Option C (GPT-2) to validate the approach, then scale to Option B (GPT-NeoX-20B) once working.

### Step 3: Create Minimal Training Script

Create `training/scripts/train_minimal.py`:

```python
#!/usr/bin/env python3
"""
Minimal training script for hybrid diffusion adapters
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models import HybridGPTOSS20B

# 1. Load base model
print("Loading base model...")
base_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

# 2. Create hybrid model
print("Creating hybrid model...")
model = HybridGPTOSS20B(
    base_model=base_model,
    freeze_base=True,
    adapter_dim=128,
    num_timesteps=1000,
    vocab_size=len(tokenizer),
)

# 3. Prepare sample data (replace with real data later)
print("Preparing data...")
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer(text, return_tensors="pt").input_ids

# 4. Test forward pass
print("Testing forward pass...")
timesteps = torch.randint(0, 1000, (1,))
logits = model(tokens, timestep=timesteps, mode="diffusion")
print(f"Output shape: {logits.shape}")

# 5. Compute loss
print("Testing loss computation...")
loss = model.diffusion.loss(model, tokens)
print(f"Loss: {loss.item():.4f}")

print("\n✓ Minimal training script working!")
print("\nNext: Add training loop and real dataset")
```

**Run it**:
```bash
cd /home/user/oss_srv/training
python scripts/train_minimal.py
```

---

## Week 1-2: Validation & Data Preparation

### Goals
1. ✅ Validate hybrid model works with real base model
2. ✅ Prepare training dataset (10-20B tokens)
3. ✅ Implement full training loop
4. ✅ Test on 2× L40S GPUs

### Tasks

#### Task 1.1: Validate with GPT-2
```bash
# Test with smaller model first
python scripts/train_minimal.py --base-model gpt2-large --test-mode
```

**Expected**: No errors, loss decreases on small batch

#### Task 1.2: Prepare Training Data

**Option A: Use Existing Datasets**
```python
from datasets import load_dataset

# Load C4 dataset (800GB, subset available)
dataset = load_dataset("c4", "en", split="train", streaming=True)

# Or use The Pile (800GB)
dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)

# Or use Wikipedia + BookCorpus (16GB)
dataset = load_dataset("bookcorpusopen", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
```

**Option B: Use GPT-OSS-20B Training Data** (if available)
```bash
# If you have the original training data
# Copy to /home/user/oss_srv/data/training/
```

**Recommendation**: Start with Wikipedia + BookCorpus (smaller, faster) for PoC validation

#### Task 1.3: Implement Data Loading

Create `training/utils/data_loader.py`:

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

def create_dataloader(
    dataset_name="wikipedia",
    tokenizer=None,
    batch_size=4,
    max_length=512,
    num_workers=4,
):
    """Create dataloader for training"""

    # Load dataset
    if dataset_name == "wikipedia":
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:10%]")
    elif dataset_name == "bookcorpus":
        dataset = load_dataset("bookcorpusopen", split="train[:10%]")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Create dataloader
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
```

#### Task 1.4: Implement Training Loop

Create `training/scripts/train.py`:

```python
#!/usr/bin/env python3
"""
Training script for hybrid diffusion adapters
"""

import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter
import os

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import HybridGPTOSS20B
from utils.data_loader import create_dataloader


def train(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model
    print(f"Loading base model: {args.base_model}")
    base_model = GPT2LMHeadModel.from_pretrained(args.base_model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Create hybrid model
    print("Creating hybrid model...")
    model = HybridGPTOSS20B(
        base_model=base_model,
        freeze_base=True,
        adapter_dim=args.adapter_dim,
        num_timesteps=args.num_timesteps,
        vocab_size=len(tokenizer),
    ).to(device)

    # Optimizer (only adapters)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Dataloader
    print("Loading data...")
    dataloader = create_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch["input_ids"].to(device)

            # Forward pass
            loss = model.diffusion.loss(model, input_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), args.max_grad_norm)
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            global_step += 1

            if step % args.log_interval == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/avg_loss", avg_loss, global_step)

            # Save checkpoint
            if step % args.save_interval == 0 and step > 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, os.path.join(checkpoint_path, "model.pt"))
                print(f"Saved checkpoint to {checkpoint_path}")

        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} complete. Average loss: {avg_epoch_loss:.4f}")

    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, "model.pt"))
    print(f"Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="gpt2-large", help="Base model name")
    parser.add_argument("--adapter-dim", type=int, default=128, help="Adapter dimension")
    parser.add_argument("--num-timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--dataset", default="wikipedia", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--output-dir", default="checkpoints/hybrid_v1", help="Output directory")
    parser.add_argument("--log-dir", default="logs/hybrid_v1", help="TensorBoard log directory")

    args = parser.parse_args()
    train(args)
```

---

## Week 3-4: Training on 2× L40S

### Launch Training

```bash
cd /home/user/oss_srv/training

# Create directories
mkdir -p checkpoints logs data

# Single GPU test
python scripts/train.py \
    --base-model gpt2-large \
    --batch-size 4 \
    --num-epochs 1 \
    --output-dir checkpoints/test

# Multi-GPU training (2× L40S)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --base-model gpt2-large \
    --batch-size 4 \
    --num-epochs 3 \
    --output-dir checkpoints/hybrid_v1
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir logs/hybrid_v1 --port 6006

# Watch GPU usage
watch -n 1 nvidia-smi
```

**Expected GPU Usage**:
- Memory: 30-35GB per GPU
- Utilization: 70-80%
- Free: 13-18GB for parallel tasks

### Success Criteria

- ✅ Training loss decreases from ~8.0 to ~3.5-4.0
- ✅ No OOM errors
- ✅ GPU utilization 70-80%
- ✅ Checkpoints save successfully

---

## Week 5-6: Demos & Validation

### Build Demo Scripts

Create `training/scripts/demo.py`:

```python
#!/usr/bin/env python3
"""
Demo script for hybrid diffusion-transformer
"""

import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import HybridGPTOSS20B


def demo_autoregressive(model, tokenizer, prompt, max_tokens=50):
    """Demo: Autoregressive generation (original GPT)"""
    print("\n" + "="*60)
    print("DEMO 1: Autoregressive Generation")
    print("="*60)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = model.generate_autoregressive(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_p=0.95,
    )

    output = tokenizer.decode(generated[0])
    print(f"Prompt: {prompt}")
    print(f"Generated: {output}")


def demo_editing(model, tokenizer, text, edit_positions):
    """Demo: Smart editing with diffusion"""
    print("\n" + "="*60)
    print("DEMO 2: Smart Editing")
    print("="*60)

    tokens = tokenizer(text, return_tensors="pt").input_ids
    edit_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for pos in edit_positions:
        if pos < tokens.size(1):
            edit_mask[0, pos] = True

    edited_tokens = model.edit_text(
        tokens,
        edit_mask,
        num_steps=50,
        temperature=0.8,
    )

    original = tokenizer.decode(tokens[0])
    edited = tokenizer.decode(edited_tokens[0])

    print(f"Original: {original}")
    print(f"Edited:   {edited}")
    print(f"Positions edited: {edit_positions}")


def demo_infilling(model, tokenizer, text_with_masks):
    """Demo: Bidirectional infilling"""
    print("\n" + "="*60)
    print("DEMO 3: Bidirectional Infilling")
    print("="*60)

    # Replace [MASK] with actual mask token
    text_processed = text_with_masks.replace("[MASK]", tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else "<|endoftext|>")

    tokens = tokenizer(text_processed, return_tensors="pt").input_ids
    filled_tokens = model.infill(
        tokens,
        num_steps=50,
        temperature=0.7,
    )

    original = tokenizer.decode(tokens[0])
    filled = tokenizer.decode(filled_tokens[0])

    print(f"With masks: {text_with_masks}")
    print(f"Filled:     {filled}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--base-model", default="gpt2-large", help="Base model")
    parser.add_argument("--demo", choices=["all", "autoregressive", "editing", "infilling"], default="all")

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    base_model = GPT2LMHeadModel.from_pretrained(args.base_model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = HybridGPTOSS20B(
        base_model=base_model,
        freeze_base=True,
    )

    # Load checkpoint
    checkpoint = torch.load(os.path.join(args.checkpoint, "model.pt"))
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.cuda()

    print("Model loaded!\n")

    # Run demos
    if args.demo in ["all", "autoregressive"]:
        demo_autoregressive(model, tokenizer, "Once upon a time", max_tokens=50)

    if args.demo in ["all", "editing"]:
        demo_editing(
            model, tokenizer,
            "The quick brown fox jumps over the lazy dog",
            edit_positions=[3, 8]  # Edit "fox" and "dog"
        )

    if args.demo in ["all", "infilling"]:
        demo_infilling(
            model, tokenizer,
            "The capital of France is [MASK], and the capital of Spain is [MASK]."
        )


if __name__ == "__main__":
    main()
```

### Run Demos

```bash
python scripts/demo.py \
    --checkpoint checkpoints/hybrid_v1/final \
    --demo all
```

---

## Timeline Summary

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **1** | Setup, validation, data prep | Working training script, data loaded |
| **2** | Initial training (GPT-2 base) | Trained adapters, loss curves |
| **3** | Scale to larger model (GPT-NeoX) | Production adapters on 20B model |
| **4** | Fine-tuning for editing tasks | Optimized for editing/infilling |
| **5** | Demo development | Working demos, benchmarks |
| **6** | Polish, documentation | Final PoC, documentation |

**Total**: 6 weeks from start to working PoC

---

## Success Metrics

### Must Have (Week 4)
- ✅ Training completes without OOM
- ✅ Loss decreases to <4.0
- ✅ Autoregressive mode matches base model quality
- ✅ Diffusion mode generates coherent text

### Nice to Have (Week 6)
- ✅ Editing demo shows multi-position edits
- ✅ Infilling demo fills blanks correctly
- ✅ <2s latency for 1K token edits (50 steps)
- ✅ Human eval prefers edits over regeneration

---

## Common Issues & Solutions

### Issue 1: OOM on L40S

**Solution**:
```bash
# Reduce batch size
python scripts/train.py --batch-size 2

# Enable gradient checkpointing
python scripts/train.py --gradient-checkpointing

# Reduce adapter dimension
python scripts/train.py --adapter-dim 64
```

### Issue 2: Slow Training

**Solution**:
```bash
# Enable mixed precision
python scripts/train.py --fp16

# Increase num_workers
python scripts/train.py --num-workers 8

# Use smaller dataset for testing
python scripts/train.py --dataset wikipedia --max-samples 10000
```

### Issue 3: Poor Quality

**Solution**:
```python
# Verify base model is frozen
assert not next(model.base_model.parameters()).requires_grad

# Lower learning rate
python scripts/train.py --learning-rate 5e-5

# Increase training data
python scripts/train.py --num-epochs 5
```

---

## Resources

### Documentation
- Architecture: `docs/HYBRID_DIFFUSION_ARCHITECTURE.md`
- Training Guide: `training/README.md`
- This Guide: `docs/POC_NEXT_STEPS.md`

### Code
- Models: `training/models/`
- Scripts: `training/scripts/`
- Tests: `training/test_setup.py`

### References
- DiffuLLaMA: https://github.com/HKUNLP/DiffuLLaMA
- LLaDA: https://github.com/ML-GSAI/LLaDA
- LoRA: https://github.com/microsoft/LoRA

---

## Contact & Support

For questions or issues:
1. Check documentation in `docs/` and `training/README.md`
2. Run `python test_setup.py` to diagnose environment issues
3. Review training logs in `logs/` and TensorBoard
4. Open an issue on GitHub

---

**Ready to start?** Run `python test_setup.py` and follow Week 1 tasks! 🚀
