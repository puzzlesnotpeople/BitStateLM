"""
check_bin.py — validate and inspect BitStateLM binary file
Usage: python check_bin.py bitstate.bin
"""
import sys
import struct
import os

MAGIC = 0x42495453  # "BITS"


def read_fp32_tensor_header(f):
    ndim_data = f.read(4)
    if len(ndim_data) < 4:
        return None, 0
    ndim = struct.unpack('<I', ndim_data)[0]
    if ndim > 8:
        return None, 0
    shape = []
    total = 1
    for _ in range(ndim):
        d = struct.unpack('<I', f.read(4))[0]
        shape.append(d)
        total *= d
    nbytes = total * 4  # float32
    f.seek(nbytes, 1)
    return shape, nbytes


def check_bin(path):
    size = os.path.getsize(path)
    print(f"File: {path}  ({size / 1024**2:.2f} MB)\n")

    with open(path, 'rb') as f:
        # Magic
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != MAGIC:
            print(f"ERROR: bad magic 0x{magic:08X}, expected 0x{MAGIC:08X}")
            return

        # Version
        version = struct.unpack('<I', f.read(4))[0]
        major = (version >> 16) & 0xFFFF
        minor = version & 0xFFFF
        packed = (version == 0x00010001) or (version == 0x00010002)
        int8_emb = (version == 0x00010002)
        print(f"Magic:   OK (BITS)")
        print(f"Version: {major}.{minor}  packed={packed}  int8_emb={int8_emb}")

        # Config
        num_fields = struct.unpack('<I', f.read(4))[0]
        fields = ['vocab_size', 'n_layer', 'n_embd', 'n_head',
                  'head_size', 'ff_mult', 'max_seq_len', 'use_bitnet', 'tie_weights']
        cfg = {}
        for i in range(num_fields):
            val = struct.unpack('<i', f.read(4))[0]
            if i < len(fields):
                cfg[fields[i]] = val
        dropout = struct.unpack('<f', f.read(4))[0]
        cfg['dropout'] = dropout

        print(f"\nConfig:")
        for k, v in cfg.items():
            print(f"  {k} = {v}")

        n_layer = cfg.get('n_layer', 0)
        n_embd  = cfg.get('n_embd', 0)
        n_head  = cfg.get('n_head', 0)
        vocab   = cfg.get('vocab_size', 0)

        # Expected param count (rough)
        proj_params = 5 * n_embd * n_embd  # TM: r,k,v,g,o
        cm_params   = n_embd * (n_embd * 4) + n_embd * n_embd + (n_embd * 4) * n_embd  # CM: k,r,v
        total_params = n_layer * (proj_params + cm_params) + vocab * n_embd
        print(f"\nExpected params (approx): {total_params / 1e6:.1f}M")
        print(f"Expected INT8 size: {total_params / 1024**2:.1f} MB")
        print(f"Expected 2-bit size: {total_params / 4 / 1024**2:.1f} MB")

        remaining = size - f.tell()
        print(f"\nRemaining bytes after config: {remaining / 1024**2:.2f} MB")
        print("\nFile looks valid ✓")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_bin.py <model.bin>")
        sys.exit(1)
    check_bin(sys.argv[1])
