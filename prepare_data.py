"""
BitStateLM Data Preparation
Downloads TinyStories, tokenizes via Phi-3.5 tokenizer → data/tokens.bin
"""
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_data(
    output_path: str = "data/tokens.bin",
    dataset_name: str = "roneneldan/TinyStories",
    tokenizer_name: str = "microsoft/Phi-3.5-mini-instruct",
    max_tokens: int = 100_000_000,  # ~100M tokens for start
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train", streaming=True)
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    
    # Determine dtype for tokens
    dtype = np.uint16 if vocab_size < 65535 else np.uint32
    
    # Preliminary pass to estimate size
    print(f"\nTokenizing (target: ~{max_tokens:,} tokens)...")
    all_tokens = []
    total = 0
    
    for i, example in enumerate(ds):
        if total >= max_tokens:
            break
        
        text = example.get("text", example.get("story", ""))
        if not text:
            continue
            
        # Tokenization with eos_token
        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_tokens.extend(tokens)
        total += len(tokens)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,} stories, {total:,} tokens...")
    
    # Save
    tokens_array = np.array(all_tokens, dtype=dtype)
    
    # Create memmap and write
    fp = np.memmap(output_path, dtype=dtype, mode='w+', shape=(len(tokens_array),))
    fp[:] = tokens_array[:]
    fp.flush()
    
    print(f"\nSaved {len(tokens_array):,} tokens to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
    print(f"Vocab size (for BitStateConfig): {vocab_size}")
    
    # Verify reading
    verify = np.memmap(output_path, dtype=dtype, mode='r')
    print(f"Verification: first 10 tokens = {verify[:10].tolist()}")
    print(f"Verification: last 10 tokens = {verify[-10:].tolist()}")
    
    return vocab_size

if __name__ == "__main__":
    vocab_size = prepare_data()
    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"Update train.py: BitStateConfig(vocab_size={vocab_size})")
    print(f"{'='*60}")
