# BitStateLM

A small language model built around RWKV-style recurrent attention and BitNet 1.58-bit weight quantization. Runs inference in under 1 GB of RAM. The C++ engine has no external dependencies.

## 🚀 Try it now

- **🌐 [Live WASM Demo](https://puzzlesnotpeople.github.io/BitStateLM/)** — Run the model directly in your browser (zero install)
- **📥 [Download Pre-trained Weights](https://github.com/puzzlesnotpeople/BitStateLM/releases)** — `bitstate.bin` (**8.7 MB**) — true 1.58-bit compression

## 📦 Quick Download & Run

Download the pre-built binary weights and run inference immediately:

```bash
# Download the model (8.7 MB — 2-bit packed weights + INT8 embeddings)
curl -L -o bitstate.bin https://github.com/puzzlesnotpeople/BitStateLM/releases/download/v1.0/bitstate.bin

# Compile the C++ engine
g++ -O3 -std=c++17 -o bitstate bitstate_packed.cpp

# Run inference
./bitstate bitstate.bin 0.8 100
```

Or use wget:
```bash
wget https://github.com/puzzlesnotpeople/BitStateLM/releases/download/v1.0/bitstate.bin
```

## Architecture

- **Attention**: RWKV WKV mechanism. O(1) memory per step at inference time, no KV cache needed
- **Weights**: BitNet 1.58-bit (ternary {-1, 0, 1}) with INT8 activations
- **Normalization**: LayerNorm before each block (pre-norm)
- **Tokenizer**: Phi-3.5 / any HuggingFace tokenizer via `prepare_data.py`

Default config: 4 layers, 256 embedding dim, 4 heads. ~35M parameters quantized to 1.58-bit ternary weights + INT8 embeddings = **8.7 MB total**.

## Files

```
model.py                  - model definition (BitStateLM, BitStateConfig, BitLinear, WKV)
train.py                  - training loop with gradient accumulation and distillation
prepare_data.py           - downloads TinyStories and tokenizes to tokens.bin
quantize.py               - converts FP32 checkpoint to INT8 inference mode
export_bin.py             - exports INT8 .pt to flat binary (legacy format)
export_mcu.py             - exports with 2-bit packed weights (4x compression)
bitstate.cpp              - standalone C++ inference engine (legacy format)
bitstate_packed.cpp       - C++ engine with 2-bit packed weight support
bitstate.bin         - ⬇️ PRE-TRAINED WEIGHTS (8.7 MB, 2-bit packed + INT8 emb)
wasm/                     - 🌐 WebAssembly demo for browser
  bitstate_wasm_packed.cpp  - Emscripten-compatible C++ with packed weights
  index.html              - Interactive web demo
  build.bat / build.sh    - Build scripts for WASM
```

## Quick start

**1. Prepare data**

```bash
pip install torch transformers datasets numpy psutil
python prepare_data.py
# writes data/tokens.bin (~200 MB for 100M tokens of TinyStories)
```

**2. Train**

```bash
python train.py
# checkpoints saved to checkpoints/bitstate_best.pt
# on a single GPU: ~6 hours for 400k steps
```

**3. Quantize**

```bash
python quantize.py --input checkpoints/bitstate_best.pt --output bitstate_int8.pt --test
```

**4. Export to binary (2-bit packed)**

```bash
python export_mcu.py --input bitstate_int8.pt --output bitstate.bin
# Creates 8.7 MB binary with 2-bit packed weights + INT8 embeddings
```

**5. Run C++ inference**

```bash
g++ -O3 -std=c++17 -o bitstate bitstate_packed.cpp
./bitstate bitstate.bin 0.8 200
```

Arguments: `[binary] [model.bin] [temperature] [max_tokens]`

## Training config

Key parameters in `train.py`:

| Parameter | Default | Notes |
|---|---|---|
| `batch_size` | 4 | increase if VRAM allows |
| `grad_accum` | 32 | effective batch = 128 |
| `seq_len` | 1024 | |
| `max_steps` | 400 000 | ~52B tokens |
| `lr` | 3e-4 | cosine decay to 3e-5 |

Knowledge distillation from a teacher model is optional. Set `teacher_path = None` to train from scratch with cross-entropy only.

## Memory usage

| Component | Size |
|---|---|
| **Model on disk (full 1.58-bit)** | **~8.7 MB** |
| Model weights (2-bit packed) | 0.6 MB |
| Embeddings (INT8) | ~8 MB |
| Model on disk (legacy INT8) | ~65 MB |
| RAM at inference | ~50 MB |
| VRAM for training (fp32) | ~4 GB |

**True 1.58-bit compression**:
- Ternary weights (-1, 0, +1): **2 bits per weight** (4× packed)
- Embeddings: **INT8** with per-token scaling (4× vs FP32)
- Total: **8.7 MB** for 35M parameter model

## Inference speed

Tested on CPU (no GPU at inference):

| Hardware | tok/s |
|---|---|
| Python (i7, CPU) | ~53 tok/s |
| C++ (WSL, 1 core) | ~43 tok/s |
| i5-8250U | ~25 tok/s |
| WASM (Chrome) | ~10 tok/s |

## 🌐 WebAssembly Build

Compile and run in the browser with Emscripten:

```bash
cd wasm

# On Windows
build.bat

# On Linux/Mac
./build.sh

# Test locally
python -m http.server 8080
# Open http://localhost:8080
```

The WASM build auto-deploys to GitHub Pages on every push to main.

## ESP32-S3 port (in progress)

The goal is to run a smaller variant of this model on an ESP32-S3 with 8 MB PSRAM:

- Reduced config: 4 layers, 128 embedding dim, byte-level tokenizer (vocab=256)
- Weights in PSRAM (~930 KB), WKV state in SRAM (~64 KB)
- Expected speed: 2–8 tok/s on Xtensa LX7 @ 240 MHz

See the `esp32/` branch (coming soon).

## Requirements

```
torch >= 2.1
transformers
datasets
numpy
psutil
```

C++ engine requires only a C++17 compiler, no libraries.

## License

MIT
