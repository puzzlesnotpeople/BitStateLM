"""
BitStateLM MCU Binary Export with 2-bit Ternary Weight Packing
Converts bitstate_int8.pt to packed binary for embedded inference

Packing format: 4 ternary weights per byte (2 bits each)
- 00 -> 0 (zero)
- 01 -> 1 (+1)  
- 10 -> 2 (-1, treated as -1.0 in inference)
- 11 -> reserved

This reduces weight storage from 8 bits to 2 bits per parameter (4x compression).
"""
import os
import sys
import struct
import torch
import numpy as np
from model import BitStateLM, BitStateConfig


def write_header(f, cfg: BitStateConfig, use_int8_emb: bool = True):
    """Write config as int32 values (9 fields) + version info"""
    header = [
        cfg.vocab_size,
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.head_size,
        cfg.ff_mult,
        cfg.max_seq_len,
        int(cfg.use_bitnet),
        int(cfg.tie_weights),
    ]
    
    # Magic number for validation
    f.write(struct.pack('<I', 0x42495453))  # "BITS"
    # Version 1.2: packed weights + INT8 embeddings
    version = 0x00010002 if use_int8_emb else 0x00010001
    f.write(struct.pack('<I', version))
    f.write(struct.pack('<I', len(header)))
    for val in header:
        f.write(struct.pack('<i', val))
    # dropout as float
    f.write(struct.pack('<f', cfg.dropout))
    print(f"Header written: version=1.2 (packed+int8_emb), {len(header)} config values")


def write_fp32_tensor(f, tensor: torch.Tensor, name: str):
    """Write FP32 tensor: [shape_dims] + [shape] + [data]"""
    arr = tensor.detach().cpu().float().numpy()
    shape = arr.shape
    
    # Write dimensions and shape
    f.write(struct.pack('<I', len(shape)))
    for dim in shape:
        f.write(struct.pack('<I', dim))
    
    # Write data as float32
    data = arr.flatten().astype('float32')
    f.write(data.tobytes())
    
    print(f"  FP32: {name} {shape} = {data.nbytes} bytes")
    return data.nbytes


def quantize_and_write_embedding(f, tensor: torch.Tensor, name: str):
    """
    Quantize embedding to INT8 and write.
    Returns (bytes_written, scale).
    """
    arr = tensor.detach().cpu().float().numpy()
    shape = arr.shape
    
    # Per-channel quantization (per token)
    # Find max abs value per row for scaling
    abs_max = np.abs(arr).max(axis=1, keepdims=True).clip(min=1e-8)
    scale = abs_max / 127.0
    
    # Quantize to INT8
    arr_quantized = np.round(arr / scale).clip(-127, 127).astype(np.int8)
    
    # Write shape
    f.write(struct.pack('<I', len(shape)))
    for dim in shape:
        f.write(struct.pack('<I', dim))
    
    # Write quantized data
    data = arr_quantized.flatten()
    f.write(data.tobytes())
    
    # Write scale (per-token scales)
    scale_data = scale.flatten().astype('float32')
    f.write(struct.pack('<I', 1))  # 1D tensor
    f.write(struct.pack('<I', shape[0]))  # vocab_size scales
    f.write(scale_data.tobytes())
    
    original_bytes = arr.nbytes
    new_bytes = data.nbytes + scale_data.nbytes
    
    print(f"  INT8: {name} {shape} = {new_bytes} bytes (was {original_bytes}, {original_bytes/new_bytes:.1f}x)")
    return new_bytes


def pack_ternary_weights(w_int8: np.ndarray) -> np.ndarray:
    """
    Pack ternary weights (-1, 0, 1) into 2 bits per weight.
    4 weights per byte: [w0:2b | w1:2b | w2:2b | w3:2b]
    
    Encoding:
        0 -> 00 (zero)
        1 -> 01 (+1)
        -1 -> 10 (-1)
        
    Returns uint8 array with packed weights.
    """
    flat = w_int8.flatten()
    n = len(flat)
    
    # Pad to multiple of 4 if needed
    if n % 4 != 0:
        pad_len = 4 - (n % 4)
        flat = np.pad(flat, (0, pad_len), mode='constant', constant_values=0)
        n = len(flat)
    
    packed_len = n // 4
    packed = np.zeros(packed_len, dtype=np.uint8)
    
    # Encode: -1 -> 2, 0 -> 0, 1 -> 1
    # Use bitwise operations for speed
    for i in range(4):
        chunk = flat[i::4]
        # Encode values to 2-bit codes
        encoded = np.where(chunk == 1, 1, 
                         np.where(chunk == -1, 2, 0)).astype(np.uint8)
        # Shift to position and OR
        packed |= (encoded << (i * 2))
    
    return packed


def write_packed_weights(f, tensor: torch.Tensor, name: str) -> tuple[int, int]:
    """
    Write packed ternary weights.
    Returns (packed_bytes, original_elements) for stats.
    """
    arr = tensor.detach().cpu().numpy().astype(np.int8)
    shape = arr.shape
    n_elements = arr.size
    
    # Pack the weights
    packed = pack_ternary_weights(arr)
    
    # Write shape info
    f.write(struct.pack('<I', len(shape)))
    for dim in shape:
        f.write(struct.pack('<I', dim))
    
    # Write packed data
    f.write(packed.tobytes())
    
    original_bytes = n_elements * 1  # INT8 = 1 byte per element
    packed_bytes = len(packed)
    ratio = original_bytes / packed_bytes
    
    print(f"  PACKED: {name} {shape} = {packed_bytes} bytes (was {original_bytes}, {ratio:.1f}x)")
    return packed_bytes, n_elements


def export_to_mcu(pt_path: str, bin_path: str):
    """
    Main export function with 2-bit weight packing and correct layer ordering.
    
    File structure:
    - Header (magic + version + config)
    - FP32 weights: emb, final_norm
    - For each layer:
        - TM projections (packed + internal LN)
        - CM projections (packed + internal LN)
        - Block LNs (try-catch compatible)
        - log_decay, u (time_first)
        - mu parameters
        - GroupNorm
    - FP32 scales
    """
    print(f"Loading: {pt_path}")
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    state = ckpt['model']
    
    print(f"\nModel config:")
    print(f"  vocab_size={cfg.vocab_size}, n_layer={cfg.n_layer}")
    print(f"  n_embd={cfg.n_embd}, n_head={cfg.n_head}")
    print(f"\nPacking ternary weights to 2 bits + INT8 embeddings...")
    
    def write_proj(f, prefix: str, fp32_bytes: list):
        """Write projection with packed weights and internal LayerNorm."""
        # Get projection weight (packed from _w_i8)
        w_key = f"{prefix}.weight"
        w_i8_key = f"{prefix}._w_i8"
        scale_key = f"{prefix}._w_scale"
        ln_w_key = f"{prefix}.ln.weight"
        ln_b_key = f"{prefix}.ln.bias"
        
        if w_i8_key in state:
            pb, _ = write_packed_weights(f, state[w_i8_key], w_key)
            fp32_bytes[0] += pb
        # Write scale
        if scale_key in state:
            fp32_bytes[0] += write_fp32_tensor(f, state[scale_key], scale_key)
        # Write internal LayerNorm
        if ln_w_key in state:
            fp32_bytes[0] += write_fp32_tensor(f, state[ln_w_key], ln_w_key)
        if ln_b_key in state:
            fp32_bytes[0] += write_fp32_tensor(f, state[ln_b_key], ln_b_key)
    
    with open(bin_path, 'wb') as f:
        # 1. Header
        print(f"\n[1/4] Writing header...")
        write_header(f, cfg, use_int8_emb=True)
        
        # 2. Write FP32 weights (embeddings, head, final_norm)
        print(f"\n[2/4] Writing FP32 weights...")
        fp32_bytes = [0]  # Use list for mutability in nested function
        
        # Embeddings (quantized to INT8)
        if 'emb.weight' in state:
            fp32_bytes[0] += quantize_and_write_embedding(f, state['emb.weight'], 'emb.weight')
        
        # LM head (if not tied) - written before layers
        if 'head.weight' in state and not cfg.tie_weights:
            fp32_bytes[0] += write_fp32_tensor(f, state['head.weight'], 'head.weight')
        
        # 3. Write layer weights in specific order
        print(f"\n[3/4] Writing layer weights...")
        packed_bytes = 0
        total_weights = 0
        
        for layer in range(cfg.n_layer):
            prefix = f"blocks.{layer}"
            
            # TM projections with internal LNs
            for proj in ['r_proj', 'k_proj', 'v_proj', 'g_proj', 'o_proj']:
                write_proj(f, f"{prefix}.tm.{proj}", fp32_bytes)
            
            # CM projections with internal LNs
            for proj in ['k', 'r', 'v']:
                write_proj(f, f"{prefix}.cm.{proj}", fp32_bytes)
            
            # Block-level LNs: ln1 (before TM), ln2 (before CM)
            ln1_w = f"{prefix}.ln1.weight"
            ln1_b = f"{prefix}.ln1.bias"
            ln2_w = f"{prefix}.ln2.weight"
            ln2_b = f"{prefix}.ln2.bias"
            if ln1_w in state:
                fp32_bytes[0] += write_fp32_tensor(f, state[ln1_w], ln1_w)
            if ln1_b in state:
                fp32_bytes[0] += write_fp32_tensor(f, state[ln1_b], ln1_b)
            if ln2_w in state:
                fp32_bytes[0] += write_fp32_tensor(f, state[ln2_w], ln2_w)
            if ln2_b in state:
                fp32_bytes[0] += write_fp32_tensor(f, state[ln2_b], ln2_b)
            
            # WKV parameters: log_decay and u (time_first)
            log_decay_key = f"{prefix}.tm.log_decay"
            u_key = f"{prefix}.tm.u"
            if log_decay_key in state:
                fp32_bytes[0] += write_fp32_tensor(f, state[log_decay_key], log_decay_key)
            if u_key in state:
                fp32_bytes[0] += write_fp32_tensor(f, state[u_key], u_key)
            
            # Time-mixing parameters (μ) for TM
            for mu in ['mu_r', 'mu_k', 'mu_v', 'mu_g']:
                key = f"{prefix}.tm.{mu}"
                if key in state:
                    fp32_bytes[0] += write_fp32_tensor(f, state[key], key)
            
            # Time-mixing parameters (μ) for CM
            for mu in ['mu_k', 'mu_r']:
                key = f"{prefix}.cm.{mu}"
                if key in state:
                    fp32_bytes[0] += write_fp32_tensor(f, state[key], key)
            
            # GroupNorm
            gn_w = f"{prefix}.tm.gn.weight"
            gn_b = f"{prefix}.tm.gn.bias"
            if gn_w in state:
                fp32_bytes[0] += write_fp32_tensor(f, state[gn_w], gn_w)
            if gn_b in state:
                fp32_bytes[0] += write_fp32_tensor(f, state[gn_b], gn_b)
        
        # 4. Final LayerNorm (ln_f) — written AFTER all layers to match C++ load order
        print(f"\n[4/4] Writing final LayerNorm...")
        if 'ln_f.weight' in state:
            fp32_bytes[0] += write_fp32_tensor(f, state['ln_f.weight'], 'ln_f.weight')
        if 'ln_f.bias' in state:
            fp32_bytes[0] += write_fp32_tensor(f, state['ln_f.bias'], 'ln_f.bias')
    
    total_size = os.path.getsize(bin_path)
    
    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"  FP32 data:     {fp32_bytes[0]/1024**2:.1f} MB")
    print(f"  Total size:   {total_size/1024**2:.1f} MB")
    print(f"{'='*60}")
    
    return bin_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='bitstate_int8.pt')
    parser.add_argument('--output', default='bitstate.bin')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found!")
        print("Run: python quantize.py --test")
        sys.exit(1)
    
    export_to_mcu(args.input, args.output)
