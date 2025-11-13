"""
Data loading utilities for hybrid diffusion-transformer training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List
import numpy as np


class TextDataset(Dataset):
    """Simple text dataset for training"""

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        mask_token_id: Optional[int] = None,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_token_id = mask_token_id or tokenizer.vocab_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def create_dataloader(
    dataset_name: str = "wikipedia",
    tokenizer = None,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    streaming: bool = False,
):
    """
    Create dataloader for training

    Args:
        dataset_name: Name of dataset ("wikipedia", "bookcorpus", "c4", etc.)
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        max_samples: Maximum number of samples to use (for testing)
        streaming: Use streaming mode for large datasets

    Returns:
        DataLoader instance
    """
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name}")

    # Load dataset
    if dataset_name == "wikipedia":
        dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train",
            streaming=streaming,
        )
        text_column = "text"

    elif dataset_name == "bookcorpus":
        dataset = load_dataset(
            "bookcorpusopen",
            split="train",
            streaming=streaming,
        )
        text_column = "text"

    elif dataset_name == "c4":
        dataset = load_dataset(
            "c4",
            "en",
            split="train",
            streaming=streaming,
        )
        text_column = "text"

    elif dataset_name == "openwebtext":
        dataset = load_dataset(
            "openwebtext",
            split="train",
            streaming=streaming,
        )
        text_column = "text"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Limit samples if specified
    if max_samples is not None and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt" if not streaming else None,
        )

    if streaming:
        # For streaming datasets, map on the fly
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else [text_column],
        )
    else:
        # For non-streaming, tokenize all at once
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

    # Create dataloader
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=not streaming,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"✓ Dataloader created: batch_size={batch_size}, max_length={max_length}")

    return dataloader


def create_simple_dataloader(
    texts: List[str],
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 2,
):
    """
    Create simple dataloader from list of texts (for testing)

    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    dataset = TextDataset(texts, tokenizer, max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"✓ Simple dataloader created: {len(texts)} samples, batch_size={batch_size}")

    return dataloader


# Test the data loader
if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("Testing data loader...")

    # Create tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Test with simple texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a high-level programming language.",
        "Deep learning models require large amounts of data.",
    ]

    # Create dataloader
    dataloader = create_simple_dataloader(
        texts,
        tokenizer,
        batch_size=2,
        max_length=128,
    )

    # Test iteration
    for batch in dataloader:
        print(f"Batch shape: {batch['input_ids'].shape}")
        print(f"Sample tokens: {batch['input_ids'][0][:20]}")
        break

    print("\n✓ Data loader test passed!")
