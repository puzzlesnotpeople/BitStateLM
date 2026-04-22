"""
BitStateLM INT8 Inference Export + Test
Converts FP32 checkpoint to INT8 for minimal size (~70MB)
"""
import os
import sys
import time
import psutil
import torch
from model import BitStateLM, BitStateConfig

def export_checkpoint(input_path: str, output_path: str):
    """
    Loads FP32 checkpoint, converts all BitLinear to INT8 mode,
    saves result. Removes FP32 weight copies (INT8 buffers only).
    """
    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Create model with checkpoint config
    model = BitStateLM(ckpt['config'])
    model.load_state_dict(ckpt['model'])
    
    # Convert to INT8 inference mode (removes FP32 weights)
    print("Converting to INT8 inference mode...")
    model.to_inference_mode()
    
    # Check: no FP32 BitLinear weights, only INT8 buffers
    for name, buf in model.named_buffers():
        if '_w_i8' in name:
            assert buf.dtype == torch.int8, f"{name} must be INT8!"
            print(f"  {name}: {buf.dtype} {tuple(buf.shape)}")
    
    # Save
    torch.save({'config': ckpt['config'], 'model': model.state_dict()}, output_path)
    
    size_mb = os.path.getsize(output_path) / 1024**2
    orig_mb = os.path.getsize(input_path) / 1024**2
    print(f"\nExport complete:")
    print(f"  Original: {orig_mb:.1f} MB")
    print(f"  INT8:     {size_mb:.1f} MB (compression {orig_mb/size_mb:.1f}x)")
    print(f"  Saved to: {output_path}")
    
    return size_mb


def test_inference(model_path: str, prompt_tokens: list = None, max_new: int = 30):
    """
    Loads8 modemodelsumeasuresand gandngenerationsspeed
    """
    print("\n" + "="*60)
    print("BitStateLM INT8 Inference Test")
    print("="*60)
    
    # ЗамерAM додоззагрузки
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1024**2
    print(f"RAM before load: {ram_before:.1f} MB")
    
    # Load model
    print(f"\nLoading: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = BitStateLM(ckpt['config'])
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    
    # Measure RAM after loading
    ram_after = process.memory_info().rss / 1024**2
    ram_model = ram_after - ram_before
    print(f"RAM after load: {ram_after:.1f} MB")
    print(f"Model RAM usage: {ram_model:.1f} MB")
    
    # Check that weights are INT8
    int8_buffers = 0
    for name, buf in model.named_buffers():
        if '_w_i8' in name:
            int8_buffers += 1
    print(f"INT8 buffers: {int8_buffers}")
    
    # Generation
    if prompt_tokens is None:
        prompt_tokens = [1, 2, 3, 4, 5]  # dummy prompt
    
    prompt = torch.tensor([prompt_tokens], dtype=torch.long)
    print(f"\nGenerating {max_new} tokens from prompt {prompt_tokens}...")
    
    with torch.no_grad():
        # Warm-up
        _ = model.generate(prompt[:, :2], max_new=2)
        
        # Real generation with timing
        t0 = time.time()
        generated = model.generate(prompt, max_new=max_new, temperature=1.0, top_k=50)
        t1 = time.time()
    
    elapsed = t1 - t0
    tokens_per_sec = max_new / elapsed if elapsed > 0 else 0
    
    print(f"\nGenerated: {generated[0].tolist()}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {tokens_per_sec:.2f} tok/s")
    
    # Финальный замер RAM
    ram_final = process.memory_info().rss / 1024**2
    print(f"RAM final: {ram_final:.1f} MB")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  File size: {os.path.getsize(model_path)/1024**2:.1f} MB")
    print(f"  Model RAM: {ram_model:.1f} MB")
    print(f"  Generate speed: {tokens_per_sec:.2f} tok/s")
    print("="*60)
    
    return tokens_per_sec


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='checkpoints/bitstate_best.pt')
    parser.add_argument('--output', default='bitstate_int8.pt')
    parser.add_argument('--test', action='store_true', help='Test inference after export')
    args = parser.parse_args()
    
    # Export if checkpoint exists
    if os.path.exists(args.input):
        export_checkpoint(args.input, args.output)
    else:
        print(f"Checkpoint not found: {args.input}")
        print("Creating test model from scratch...")
        # Create test model
        cfg = BitStateConfig(vocab_size=32011, n_layer=4, n_embd=256, n_head=4, 
                         head_size=64, ff_mult=2, use_bitnet=True, tie_weights=True)
        model = BitStateLM(cfg)
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({'config': cfg, 'model': model.state_dict(), 'step': 0}, args.input)
        export_checkpoint(args.input, args.output)
    
    # Test inference
    if args.test or os.path.exists(args.output):
        test_inference(args.output)
