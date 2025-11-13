#!/usr/bin/env python3
"""
Training script for hybrid diffusion-transformer adapters

Supports:
- Single GPU and multi-GPU training (with torch.distributed)
- Mixed precision training (FP16)
- Gradient accumulation
- Checkpointing and resuming
- TensorBoard logging
- Weights & Biases integration (optional)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HybridGPTOSS20B
from utils import create_dataloader, create_simple_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, epoch, global_step, loss, args, filename="checkpoint.pt"):
    """Save training checkpoint"""
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get model state (unwrap DDP if necessary)
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args),
    }

    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state (handle DDP)
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['global_step'], checkpoint['loss']


def train(args):
    """Main training function"""

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

    if is_main_process:
        print("="*60)
        print("HYBRID DIFFUSION-TRANSFORMER TRAINING")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")

    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    if is_main_process:
        print(f"\nLoading tokenizer: {args.base_model}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    if is_main_process:
        print(f"Loading base model: {args.base_model}")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
        )
    except:
        base_model = GPT2LMHeadModel.from_pretrained(
            args.base_model,
        )
        if args.fp16:
            base_model = base_model.half()

    # Create hybrid model
    if is_main_process:
        print("Creating hybrid model...")

    model = HybridGPTOSS20B(
        base_model=base_model,
        freeze_base=True,
        adapter_dim=args.adapter_dim,
        num_timesteps=args.num_timesteps,
        vocab_size=len(tokenizer),
        mask_token_id=len(tokenizer),  # Use vocab_size as mask token
    )

    model = model.to(device)

    # Wrap with DDP for multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process:
            print(f"✓ Model wrapped with DistributedDataParallel")

    # Count parameters
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Create optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint:
        if is_main_process:
            print(f"\nResuming from checkpoint: {args.resume_from_checkpoint}")
        start_epoch, global_step, _ = load_checkpoint(
            model, optimizer, args.resume_from_checkpoint
        )

    # Create dataloader
    if is_main_process:
        print(f"\nLoading dataset: {args.dataset}")

    if args.dataset == "dummy":
        # Create dummy dataset for testing
        dummy_texts = [
            "The quick brown fox jumps over the lazy dog." * 10,
            "Machine learning is transforming the world." * 10,
            "Artificial intelligence will change everything." * 10,
            "Deep learning models require large amounts of data." * 10,
        ] * 100  # Repeat to get more samples

        dataloader = create_simple_dataloader(
            dummy_texts,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
        )
    else:
        dataloader = create_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            streaming=args.streaming,
        )

    # TensorBoard writer (only on main process)
    writer = None
    if is_main_process and args.use_tensorboard:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"✓ TensorBoard logging to: {log_dir}")

    # Weights & Biases (optional)
    if is_main_process and args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            print(f"✓ W&B logging enabled: {args.wandb_project}/{args.wandb_run_name}")
        except ImportError:
            print("⚠️  wandb not installed, skipping W&B logging")
            args.use_wandb = False

    # Training loop
    if is_main_process:
        print(f"\nStarting training for {args.num_epochs} epochs...")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * world_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print("="*60)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Progress bar (only on main process)
        pbar = tqdm(dataloader, disable=not is_main_process, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(device)

            # Forward pass
            # Get model (unwrap DDP if necessary)
            model_unwrapped = model.module if hasattr(model, 'module') else model

            loss = model_unwrapped.diffusion.loss(model_unwrapped, input_ids)

            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights every N steps
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                num_batches += 1

                if is_main_process:
                    # Update progress bar
                    avg_loss = epoch_loss / num_batches
                    pbar.set_postfix({'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}', 'avg': f'{avg_loss:.4f}'})

                    # TensorBoard logging
                    if writer is not None and global_step % args.log_interval == 0:
                        writer.add_scalar("train/loss", loss.item() * args.gradient_accumulation_steps, global_step)
                        writer.add_scalar("train/avg_loss", avg_loss, global_step)
                        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], global_step)

                    # W&B logging
                    if args.use_wandb and global_step % args.log_interval == 0:
                        import wandb
                        wandb.log({
                            "train/loss": loss.item() * args.gradient_accumulation_steps,
                            "train/avg_loss": avg_loss,
                            "train/learning_rate": optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        })

                # Save checkpoint
                if is_main_process and global_step % args.save_interval == 0 and global_step > 0:
                    checkpoint_path = save_checkpoint(
                        model, optimizer, epoch, global_step,
                        loss.item() * args.gradient_accumulation_steps,
                        args,
                        filename=f"checkpoint-step-{global_step}.pt"
                    )
                    print(f"\n✓ Saved checkpoint: {checkpoint_path}")

        # End of epoch
        if is_main_process:
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch} complete. Average loss: {avg_epoch_loss:.4f}")

            # Save epoch checkpoint
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, global_step,
                avg_epoch_loss, args,
                filename=f"checkpoint-epoch-{epoch}.pt"
            )
            print(f"✓ Saved epoch checkpoint: {checkpoint_path}")

    # Save final model
    if is_main_process:
        final_dir = Path(args.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_unwrapped = model.module if hasattr(model, 'module') else model
        torch.save(model_unwrapped.state_dict(), final_dir / "model.pt")

        # Save config
        config = {
            'base_model': args.base_model,
            'adapter_dim': args.adapter_dim,
            'num_timesteps': args.num_timesteps,
            'vocab_size': len(tokenizer),
        }
        with open(final_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Training complete! Final model saved to: {final_dir}")

    # Cleanup
    if writer is not None:
        writer.close()

    if args.use_wandb:
        import wandb
        wandb.finish()

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train hybrid diffusion-transformer adapters")

    # Model arguments
    parser.add_argument("--base-model", type=str, default="gpt2", help="Base model name or path")
    parser.add_argument("--adapter-dim", type=int, default=128, help="Adapter bottleneck dimension")
    parser.add_argument("--num-timesteps", type=int, default=1000, help="Number of diffusion timesteps")

    # Data arguments
    parser.add_argument("--dataset", type=str, default="dummy", help="Dataset name (dummy, wikipedia, bookcorpus, c4)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples (for testing)")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for large datasets")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam-epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")

    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="checkpoints/hybrid_v1", help="Output directory")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save checkpoint every N steps")

    # Logging
    parser.add_argument("--log-dir", type=str, default="logs/hybrid_v1", help="TensorBoard log directory")
    parser.add_argument("--log-interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--use-tensorboard", action="store_true", default=True, help="Use TensorBoard logging")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="hybrid-gptoss20b", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    # Distributed training (set automatically by torchrun)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    # Set wandb run name if not provided
    if args.use_wandb and args.wandb_run_name is None:
        args.wandb_run_name = f"{args.base_model.replace('/', '-')}_adapter{args.adapter_dim}_bs{args.batch_size}"

    train(args)


if __name__ == "__main__":
    main()
