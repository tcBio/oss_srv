# Hybrid Diffusion-Transformer Training

This directory contains the implementation of the Hybrid GPT-OSS-20B model with diffusion editing capabilities.

## Overview

The hybrid model adds **parallel editing and infilling capabilities** to GPT-OSS-20B while preserving its autoregressive strengths:

- **Mode 1**: Autoregressive generation (original GPT behavior)
- **Mode 2**: Diffusion editing/infilling (new capability)

### Key Features

✅ **Minimal Changes**: Only 1-2% additional parameters (35M adapters on top of 20B base)
✅ **Dual-Mode**: Switch between autoregressive and diffusion modes
✅ **Frozen Base**: GPT-OSS-20B weights remain frozen (preserves quality)
✅ **Efficient Training**: Fits on 2× L40S GPUs (30-35GB memory per GPU)
✅ **Novel Capabilities**: Parallel editing, bidirectional infilling, controlled rewriting

## Architecture

```
GPT-OSS-20B (Frozen 20B params)
    ↓
Time-Conditioned Adapters (Trainable 35M params)
    ↓
Dual-Mode Attention (Causal or Bidirectional)
    ↓
Iterative Denoising (Diffusion Mode Only)
```

### Components

1. **`models/time_adapter.py`**: Time-step conditioning adapters (LoRA-style)
2. **`models/discrete_diffusion.py`**: Discrete token diffusion process
3. **`models/hybrid_model.py`**: Hybrid model wrapper combining base + adapters
4. **`scripts/train.py`**: Training script for adapter pre-training
5. **`scripts/demo.py`**: Demo script showcasing editing/infilling

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 12.8+ (for L40S GPUs)
- Transformers (HuggingFace)
- DeepSpeed (for multi-GPU training)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## Quick Start

### 1. Convert GPT-OSS-20B to PyTorch

First, export the TensorRT model to PyTorch format:

```bash
# TODO: Add conversion script
python scripts/convert_tensorrt_to_pytorch.py \
    --engine-path /path/to/gptoss20b.engine \
    --output-path models/gptoss20b_pytorch.pth
```

### 2. Train Adapters

```bash
# Single GPU (for testing)
python scripts/train.py \
    --base-model models/gptoss20b_pytorch.pth \
    --output-dir checkpoints/hybrid_v1 \
    --batch-size 4 \
    --num-epochs 3

# Multi-GPU (production)
deepspeed scripts/train.py \
    --deepspeed configs/deepspeed_config.json \
    --base-model models/gptoss20b_pytorch.pth \
    --output-dir checkpoints/hybrid_v1 \
    --batch-size 8 \
    --num-epochs 3
```

### 3. Run Demos

```bash
# Demo 1: Smart editing
python scripts/demo.py \
    --checkpoint checkpoints/hybrid_v1/final \
    --demo smart_editing \
    --text "The quick brown fox jumps over the lazy dog" \
    --instruction "Make it about cats and mice"

# Demo 2: Infilling
python scripts/demo.py \
    --checkpoint checkpoints/hybrid_v1/final \
    --demo infilling \
    --text "The meeting is [MASK] [MASK] [MASK], we should [MASK] [MASK]"

# Demo 3: Parallel generation
python scripts/demo.py \
    --checkpoint checkpoints/hybrid_v1/final \
    --demo parallel_generation \
    --prompt "Once upon a time" \
    --length 512
```

## Training

### Phase 1: Adapter Pre-Training (Week 1-2)

**Objective**: Train time adapters with masked language modeling

**Data**: 10-20B tokens from OSS20B training corpus

**Configuration**:
```
GPUs:              2× L40S (48GB each)
Batch Size:        4-8 per GPU
Gradient Accum:    4× (effective batch = 32-64)
Learning Rate:     1e-4
Optimizer:         AdamW
Precision:         Mixed FP16/FP32
Duration:          1-2 weeks
Memory per GPU:    30-35GB
```

**Expected Loss**:
- Initial: ~8.0 (random)
- Final: ~3.5-4.0 (competitive with base model)

### Phase 2: Editing Fine-Tuning (Week 3-4)

**Objective**: Optimize for editing/infilling tasks

**Data**: 1-5B tokens from editing datasets
- GitHub commits (code edits)
- Wikipedia revisions
- Paraphrase datasets
- Style transfer examples

**Configuration**:
```
Same as Phase 1, but:
Learning Rate:     5e-5 (lower for fine-tuning)
Duration:          1-2 weeks
```

### Monitoring

Track metrics with Weights & Biases:

```bash
export WANDB_PROJECT="hybrid-gptoss20b"
export WANDB_RUN_NAME="adapter_pretrain_v1"
python scripts/train.py ...
```

**Key Metrics**:
- Training loss (should decrease to ~3.5-4.0)
- Validation perplexity (target: <10% increase vs. base)
- Denoising accuracy (% of correctly predicted masked tokens)
- Memory usage (should stay <40GB per GPU)

## Inference Modes

### Autoregressive Generation

```python
from models.hybrid_model import HybridGPTOSS20B

# Load model
model = HybridGPTOSS20B.from_pretrained("checkpoints/hybrid_v1/final")

# Generate text
prompt = tokenizer.encode("Once upon a time")
generated = model.generate_autoregressive(
    torch.tensor([prompt]),
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.95
)
print(tokenizer.decode(generated[0]))
```

### Diffusion Editing

```python
# Edit specific positions
text = "The quick brown fox jumps over the lazy dog"
tokens = tokenizer.encode(text)

# Mark positions to edit (e.g., change "fox" and "dog")
edit_mask = torch.zeros(len(tokens), dtype=torch.bool)
edit_mask[3] = True  # "fox"
edit_mask[8] = True  # "dog"

# Edit with diffusion
edited_tokens = model.edit_text(
    torch.tensor([tokens]),
    edit_mask.unsqueeze(0),
    num_steps=50,
    temperature=0.8
)
print(tokenizer.decode(edited_tokens[0]))
```

### Bidirectional Infilling

```python
# Fill in [MASK] tokens
text_with_masks = "The capital of France is [MASK], and the capital of Spain is [MASK]."
tokens = tokenizer.encode(text_with_masks)

# Infill automatically detects [MASK] tokens
filled_tokens = model.infill(
    torch.tensor([tokens]),
    num_steps=50,
    temperature=0.7
)
print(tokenizer.decode(filled_tokens[0]))
```

## Memory Requirements

### Training (2× L40S)

| Component | Memory per GPU |
|-----------|----------------|
| Base Model (frozen, FP16) | ~20GB |
| Adapters (trainable) | ~0.3GB |
| Gradients | ~0.3GB |
| Optimizer States (Adam) | ~0.6GB |
| Activations (batch=4) | ~10-15GB |
| **Total** | **30-35GB** ✅ |
| **Free for parallel tasks** | **13-18GB** |

### Inference

| Mode | Memory per GPU |
|------|----------------|
| Autoregressive | ~22-25GB |
| Diffusion (50 steps) | ~25-30GB |

## Performance Benchmarks

### Target Metrics

| Task | Metric | Target |
|------|--------|--------|
| **Autoregressive Quality** | Perplexity | 100% match with base OSS20B |
| **Diffusion Quality** | Perplexity | >95% of base OSS20B |
| **Editing Speed** | Latency | <2s for 1K tokens (50 steps) |
| **Infilling Accuracy** | Exact Match | >90% on cloze tests |
| **Parallel Generation** | Speedup | 5-10× vs. autoregressive (512 tokens) |

### Preliminary Results

(To be updated after training)

## Troubleshooting

### OOM (Out of Memory) Errors

**Solution 1**: Reduce batch size
```bash
python scripts/train.py --batch-size 2  # Instead of 4
```

**Solution 2**: Enable gradient checkpointing
```bash
python scripts/train.py --gradient-checkpointing
```

**Solution 3**: Use smaller adapter dimension
```bash
python scripts/train.py --adapter-dim 64  # Instead of 128
```

### Quality Degradation

**Check 1**: Verify base model is frozen
```python
assert not next(model.base_model.parameters()).requires_grad
```

**Check 2**: Validate on base model first
```python
# Should match original OSS20B perplexity
base_ppl = evaluate_perplexity(model.base_model, val_data)
```

**Check 3**: Reduce learning rate
```bash
python scripts/train.py --learning-rate 5e-5  # Instead of 1e-4
```

### Slow Training

**Solution 1**: Enable mixed precision
```bash
python scripts/train.py --fp16
```

**Solution 2**: Use gradient accumulation
```bash
python scripts/train.py --gradient-accumulation-steps 4
```

**Solution 3**: Optimize data loading
```bash
python scripts/train.py --num-workers 4 --prefetch-factor 2
```

## Citation

If you use this hybrid diffusion-transformer architecture, please cite:

```bibtex
@software{hybrid_gptoss20b_2025,
  title={Hybrid Diffusion-Transformer for GPT-OSS-20B},
  author={Your Team},
  year={2025},
  url={https://github.com/yourorg/oss_srv}
}
```

## References

1. **DiffuLLaMA** (ICLR 2025): Scaling Diffusion Language Models via Adaptation from Autoregressive Models
2. **LLaDA** (2025): Large Language Diffusion Models
3. **LoRA** (ICLR 2022): Low-Rank Adaptation of Large Language Models
4. **DDPM** (NeurIPS 2020): Denoising Diffusion Probabilistic Models

## License

[Your License Here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
