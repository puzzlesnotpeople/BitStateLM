# BitStateLM — Complete Bug Fix Log

## Critical bugs fixed (model produced garbage without these)

### 1. NormLayer.forward — missing LayerNorm path
`bitstate_packed.cpp`

The `else` branch was completely absent. When `use_rms=false` (the default),
`ln1`, `ln2`, and `ln_f` were silently skipped — output stayed uninitialized.
All three normalization layers now use the correct full LayerNorm formula.

### 2. Double residual — TMBlock and CMBlock
`bitstate_packed.cpp`

Both blocks had an internal residual (`out[i] = x[i] + o[i]`), and the model
forward loop added another one (`x[i] += tm_out[i]`). Result: `x = 2x + o`.
Fixed: blocks now return only the delta `o`. Residual is applied once, in the
forward loop.

### 3. `lm_head.load_unpacked` — wrong argument count (compile error)
`bitstate_packed.cpp`

Called with 2 arguments, function signature requires 4. This was a hard
compile error when `tie_weights=false`.
Fixed: call now passes `cfg.vocab_size, cfg.n_embd` as the missing dimensions.

### 4. `ln_f` written before layers in export, read after layers in C++
`export_mcu.py`

The final LayerNorm was serialized in section [2/4] (before all layer weights),
but `bitstate_packed.cpp` reads it after the layer loop. Every layer weight
was therefore read at the wrong file offset, corrupting the entire model.
Fixed: `ln_f` is now written in section [4/4], after all layers.

## Additional bugs fixed

### 5. `sample()` called with missing `top_k` argument
`bitstate_packed.cpp`

`sample(logits, temperature)` — function takes 3 arguments. The missing `top_k`
caused a compile error. Fixed: `sample(logits, temperature, 50)`.

### 6. `generate()` — prompt not processed through model
`bitstate_packed.cpp`

Only the last prompt token was fed to `forward()`. For an RWKV model, the
recurrent state must be built up by processing every prompt token in order.
Fixed: all prompt tokens are now fed through the model before generation starts.

### 7. `check_bin.py` — wrong magic number
`check_bin.py`

Expected `0x5049434F` but export writes `0x42495453` ("BITS").
Rewritten to match the current binary format and print useful diagnostics.

## What Manus correctly fixed (already in their version)

- `ln_f.weight` / `ln_f.bias` key names (was `final_norm.weight`)
- `ln1` / `ln2` key names (was `tm.ln` / `cm.ln`)
- GroupNorm implementation — true per-group normalization instead of global LN
- Wrong activations before WKV: removed `exp(-exp(r))`, `exp(k)`, `silu(v)`
- `use_rms = false` initialization (was undefined behavior)
- `prev_x` now stores raw `x` before normalization
- `ln1`/`ln2` applied outside blocks, passed as pre-normalized input

## How to verify correctness

After training and quantizing, run:

```bash
# Export
python export_mcu.py --input bitstate_int8.pt --output bitstate.bin

# Validate binary
python check_bin.py bitstate.bin

# Compile and run
g++ -O3 -std=c++17 -o bitstate bitstate_packed.cpp
./bitstate bitstate.bin 0.8 200
```

To verify C++ output matches Python:
```python
import torch
from model import BitStateLM
ckpt = torch.load('bitstate_int8.pt', map_location='cpu', weights_only=False)
model = BitStateLM(ckpt['config'])
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
with torch.no_grad():
    logits, _, _ = model(torch.tensor([[1, 2, 3]]))
    print(logits[0, -1, :5])  # compare first 5 logits with C++ output
```
