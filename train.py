"""
BitStateLM Training Pipeline

Supports:
  - Knowledge distillation from teacher model (same tokenizer required)
  - Chunked sequential data loading (avoids random I/O thrashing)
  - Gradient accumulation for large effective batch sizes
  - Curriculum learning: short sequences → long sequences
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
import numpy as np

from model import BitStateLM, BitStateConfig


# --- Training Config ---

@dataclass
class TrainConfig:
    data_path:     str   = 'data/tokens.bin'
    val_split:     float = 0.005

    # Distillation: teacher and student MUST use the same tokenizer.
    # If teacher_path = None: train without distillation (CE loss only).
    # If set: student uses teacher's tokenizer.
    # BitStateConfig.vocab_size must match len(teacher_tokenizer).
    teacher_path:  Optional[str] = None
    alpha:         float = 0.7    # soft loss weight
    temperature:   float = 4.0

    # Gradient accumulation:
    # Naive wkv_train with B=128, T=1024 creates ~68 GB VRAM tensors (T×T matrix × 32 layers).
    # Chunked scan + gradient checkpointing reduced peak, but B=128 is still heavy.
    # Solution: batch_size=4, grad_accumulation=32 → effective batch = 128.
    # With ≥80 GB VRAM and Triton kernel: can use batch_size=128, accum=1.
    batch_size:    int   = 4
    grad_accum:    int   = 32     # effective batch = batch_size × grad_accum
    seq_len:       int   = 1024
    lr:            float = 3e-4
    lr_min:        float = 3e-5
    weight_decay:  float = 0.1
    grad_clip:     float = 1.0
    warmup_steps:  int   = 2000
    max_steps:     int   = 400_000   # ~52B tokens (was 100k = 1.6B tokens)

    device:        str   = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype:         str   = 'bfloat16'
    # torch.compile on Windows CPU requires MSVC (cl.exe).
    # Without compiler raises RuntimeError. Enable only on CUDA/Linux.
    compile_model: bool  = torch.cuda.is_available()
    log_interval:  int   = 100
    eval_interval: int   = 2000
    save_interval: int   = 10_000
    out_dir:       str   = 'checkpoints'
    num_workers:   int   = 4

    # Curriculum: short sequences → long sequences (stabilizes BitNet training)
    seq_warmup:       bool = True
    seq_warmup_steps: int  = 20_000
    seq_warmup_min:   int  = 128


# --- Dataset ---

class ChunkedTokenDataset(Dataset):
    """
    Dataset with chunked sequential reading.

    Problem: np.memmap + random indices causes disk thrashing.
    With 100GB file, each random slice misses page cache,
    triggering syscall read(). GPU waits for data.

    Solution: split file into chunk_size chunks, shuffle chunk indices
    (torch.randperm), but read sequentially within each chunk.
    This provides:
      - randomness for training (different text parts each epoch)
      - sequential disk access (OS read-ahead works)
      - DataLoader prefetch covers latency of next chunk

    chunk_size: must be >> seq_len, otherwise no benefit.
    Recommended: chunk_size = seq_len * 1024 (≈ 1M tokens per chunk).
    """
    def __init__(
        self,
        path: str,
        seq_len: int,
        split: str = 'train',
        val_frac: float = 0.005,
        chunk_size: Optional[int] = None,
    ):
        data     = np.memmap(path, dtype=np.uint16, mode='r')
        n        = len(data)
        cut      = int(n * (1 - val_frac))
        raw      = data[:cut] if split == 'train' else data[cut:]

        self.seq_len    = seq_len
        self.chunk_size = chunk_size or seq_len * 1024  # default: 1M tokens per chunk

        # Split into chunks; drop incomplete last chunk
        n_chunks = len(raw) // self.chunk_size
        self.raw = raw[:n_chunks * self.chunk_size]

        # Each "example" = one (x, y) slice of length seq_len
        # Grouped by chunks for sequential reading
        self.seqs_per_chunk = self.chunk_size // seq_len
        self.n_chunks       = n_chunks

    def __len__(self):
        return self.n_chunks * self.seqs_per_chunk

    def __getitem__(self, idx: int):
        chunk_id = idx // self.seqs_per_chunk
        local_id = idx  % self.seqs_per_chunk

        start = chunk_id * self.chunk_size + local_id * self.seq_len
        # astype creates copy in RAM; needed for non-memmap tensor
        tokens = torch.from_numpy(
            self.raw[start : start + self.seq_len + 1].astype(np.int64)
        )
        return tokens[:-1], tokens[1:]


def make_loader(dataset: ChunkedTokenDataset, batch_size: int, num_workers: int) -> DataLoader:
    """
    DataLoader with chunk-level shuffling (not byte-level).
    shuffle=True shuffles __getitem__ indices. With ChunkedTokenDataset
    this means shuffling chunks, while reading sequentially within each chunk.
    """
    # prefetch_factor only when num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )


# --- Distillation ---

class DistillSetup:
    """
    Knowledge distillation setup: teacher and student share tokenizer.

    Why same tokenizer is required:
    Different models (Phi-3.5, Llama, etc.) have different BPE vocabularies.
    Token with index 145 means different things in different models.
    KL-divergence between their distributions = information noise.

    Correct approach:
    1. Take teacher's tokenizer (Phi-3.5: LlamaTokenizer, vocab=32064)
    2. Set student vocab_size = teacher vocab_size
    3. Both forward passes on same tokens → logits are comparable
    4. KL-divergence is now meaningful

    Alternative if vocabularies MUST differ:
    Feature distillation through intermediate representations,
    not through logits. But this is significantly harder and worse in practice.
    """
    def __init__(self, teacher_path: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading teacher: {teacher_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        self.teacher   = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.vocab_size = len(self.tokenizer)
        n_params = sum(p.numel() for p in self.teacher.parameters())
        print(f"Teacher loaded: {n_params/1e6:.0f}M params, vocab={self.vocab_size}")
        print(f"IMPORTANT: student vocab_size must be = {self.vocab_size}")

    @torch.no_grad()
    def get_teacher_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher(x).logits   # (B, T, teacher_vocab_size)


class DistillLoss(nn.Module):
    def __init__(self, alpha: float, temperature: float):
        super().__init__()
        self.alpha = alpha
        self.T     = temperature

    def forward(self, student_logits, teacher_logits, labels):
        """
        student_logits and teacher_logits must have the same vocab_size.
        This is guaranteed by both using the same tokenizer.
        """
        B, T, V = student_logits.shape
        assert teacher_logits.size(-1) == V, (
            f"Vocab size mismatch: student {V} vs teacher {teacher_logits.size(-1)}. "
            "Ensure BitStateConfig.vocab_size matches the teacher's vocab."
        )

        ce = F.cross_entropy(student_logits.reshape(-1, V), labels.reshape(-1), ignore_index=-1)

        s_log  = F.log_softmax(student_logits.reshape(-1, V) / self.T, dim=-1)
        t_prob = F.softmax(teacher_logits.reshape(-1, V) / self.T, dim=-1)
        kl     = F.kl_div(s_log, t_prob, reduction='batchmean') * (self.T ** 2)

        return self.alpha * kl + (1 - self.alpha) * ce, ce.detach(), kl.detach()


# --- Trainer ---

class Trainer:
    def __init__(self, model_cfg: BitStateConfig, train_cfg: TrainConfig):
        self.tcfg = train_cfg
        self.mcfg = model_cfg
        os.makedirs(train_cfg.out_dir, exist_ok=True)

        # When using distillation, vocab_size MUST match
        self.distill = None
        if train_cfg.teacher_path:
            self.distill = DistillSetup(train_cfg.teacher_path, train_cfg.device)
            if model_cfg.vocab_size != self.distill.vocab_size:
                raise ValueError(
                    f"Model vocab_size ({model_cfg.vocab_size}) ≠ teacher vocab "
                    f"({self.distill.vocab_size}). "
                    f"Set BitStateConfig(vocab_size={self.distill.vocab_size})."
                )
            self.criterion = DistillLoss(train_cfg.alpha, train_cfg.temperature)

        self.model = BitStateLM(model_cfg).to(train_cfg.device)
        if train_cfg.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

        # Separate parameters: weight decay only for weight matrices
        decay_p   = [p for n, p in self.model.named_parameters() if p.dim() >= 2 and 'emb' not in n]
        nodecay_p = [p for n, p in self.model.named_parameters() if p.dim() < 2 or 'emb' in n]
        self.opt = AdamW(
            [{'params': decay_p, 'weight_decay': train_cfg.weight_decay},
             {'params': nodecay_p, 'weight_decay': 0.0}],
            lr=train_cfg.lr, betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),
        )
        self.sched  = CosineAnnealingLR(self.opt, train_cfg.max_steps, eta_min=train_cfg.lr_min)
        # GradScaler only needed for float16 (bfloat16 has no underflow).
        # Warning: dtype='float16' + torch.compile in PyTorch < 2.3 may conflict.
        # If you see "Expected all tensors to be on the same device" with compile_model=True,
        # either disable compile_model or switch to dtype='bfloat16'.
        assert train_cfg.dtype in ('float16', 'bfloat16', 'float32'), \
            f"Unknown dtype: {train_cfg.dtype}"
        self.scaler = GradScaler(enabled=(train_cfg.dtype == 'float16'))
        self.ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[train_cfg.dtype]

        # Chunked dataset + DataLoader
        self.train_ds = ChunkedTokenDataset(train_cfg.data_path, train_cfg.seq_len, 'train', train_cfg.val_split)
        self.val_ds   = ChunkedTokenDataset(train_cfg.data_path, train_cfg.seq_len, 'val',   train_cfg.val_split)
        self.train_loader = make_loader(self.train_ds, train_cfg.batch_size, train_cfg.num_workers)
        self.val_loader   = make_loader(self.val_ds,   train_cfg.batch_size, train_cfg.num_workers)
        self._train_iter  = iter(self.train_loader)

        self.step          = 0
        self.best_val_loss = float('inf')

        # Log actual training volume
        total_tokens = train_cfg.batch_size * train_cfg.seq_len * train_cfg.max_steps
        print(f"Training volume: {total_tokens/1e9:.1f}B tokens")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.0f}M")
        print(f"Chinchilla optimal: ~{sum(p.numel() for p in self.model.parameters()) * 20 / 1e9:.0f}B tokens")

    def _get_batch(self):
        try:
            x, y = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            x, y = next(self._train_iter)
        return x.to(self.tcfg.device), y.to(self.tcfg.device)

    def _warmup_lr(self, step):
        if step < self.tcfg.warmup_steps:
            return step / self.tcfg.warmup_steps
        return 1.0

    def _curriculum_seqlen(self):
        if not self.tcfg.seq_warmup or self.step > self.tcfg.seq_warmup_steps:
            return self.tcfg.seq_len
        frac = self.step / self.tcfg.seq_warmup_steps
        return max(self.tcfg.seq_warmup_min, int(self.tcfg.seq_warmup_min + (self.tcfg.seq_len - self.tcfg.seq_warmup_min) * frac))

    def train_step(self) -> dict:
        """
        One training step with gradient accumulation.
        Parameters are updated once per grad_accum micro-steps,
        simulating effective batch = batch_size × grad_accum.
        """
        device_type = self.tcfg.device.split(':')[0]   # 'cuda' or 'cpu'
        total_loss_acc = 0.0
        ce_acc         = 0.0
        kl_acc         = 0.0

        # Curriculum: compute current seq_len for this step
        cur_seq = self._curriculum_seqlen()

        for micro_step in range(self.tcfg.grad_accum):
            x, y = self._get_batch()
            # Truncate to cur_seq. DataLoader always gives full seq_len,
            # but on early steps we use only first cur_seq tokens.
            if cur_seq < x.size(1):
                x = x[:, :cur_seq]
                y = y[:, :cur_seq]
            is_last = (micro_step == self.tcfg.grad_accum - 1)

            with autocast(device_type=device_type, dtype=self.ptdtype):
                logits, _, ce_loss = self.model(x, targets=y)

                if self.distill is not None:
                    t_logits = self.distill.get_teacher_logits(x)
                    loss, ce, kl = self.criterion(logits, t_logits, y)
                else:
                    loss, ce, kl = ce_loss, ce_loss.detach(), torch.tensor(0.0)

                # Normalize loss by grad_accum so gradient scale doesn't grow
                loss = loss / self.tcfg.grad_accum

            self.scaler.scale(loss).backward()
            total_loss_acc += loss.item()
            ce_acc         += ce.item() / self.tcfg.grad_accum
            kl_acc         += kl.item() / self.tcfg.grad_accum

        # Update parameters only after full accumulation
        for g in self.opt.param_groups:
            g['lr'] = self.tcfg.lr * self._warmup_lr(self.step)

        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        if self.step >= self.tcfg.warmup_steps:
            self.sched.step()

        return {'loss': total_loss_acc, 'ce': ce_acc, 'kl': kl_acc,
                'lr': self.opt.param_groups[0]['lr']}

    @torch.no_grad()
    def eval(self, n_batches=20) -> float:
        self.model.eval()
        val_iter = iter(self.val_loader)
        losses = []
        for _ in range(n_batches):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break
            x, y = x.to(self.tcfg.device), y.to(self.tcfg.device)
            with autocast(device_type=self.tcfg.device.split(':')[0], dtype=self.ptdtype):
                _, _, loss = self.model(x, targets=y)
            losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))

    def run(self):
        self.model.train()
        t0 = time.time()
        while self.step < self.tcfg.max_steps:
            m = self.train_step()
            self.step += 1
            if self.step % self.tcfg.log_interval == 0:
                elapsed = time.time() - t0
                print(f"step {self.step:6d} | loss {m['loss']:.4f} | ce {m['ce']:.4f} | "
                      f"kl {m['kl']:.4f} | lr {m['lr']:.2e} | {elapsed:.0f}s")
                t0 = time.time()
            if self.step % self.tcfg.eval_interval == 0:
                vl = self.eval()
                print(f"  ↳ val_loss={vl:.4f}  ppl={math.exp(vl):.2f}")
                if vl < self.best_val_loss:
                    self.best_val_loss = vl
                    self._save('best')
            if self.step % self.tcfg.save_interval == 0:
                self._save(f'step_{self.step}')
        self._save('final')

    def _save(self, tag):
        path = os.path.join(self.tcfg.out_dir, f'bitstate_{tag}.pt')
        torch.save({'config': self.mcfg, 'step': self.step,
                    'model': self.model.state_dict(), 'opt': self.opt.state_dict()}, path)
        print(f"  ✓ {path}")


# --- Main ---
if __name__ == '__main__':
    # Auto-detect mode: no CUDA → teacher=None and fewer layers for CPU
    use_cuda = torch.cuda.is_available()
    
    # Phi-3.5-mini uses LlamaTokenizer with 32064 tokens.
    # For CPU mode: teacher=None (else OOM). For CUDA: can enable distillation.
    teacher = 'microsoft/Phi-3.5-mini-instruct' if use_cuda else None
    
    # CPU-friendly config (fewer layers for quick testing)
    if use_cuda:
        model_cfg = BitStateConfig(
            vocab_size  = 32011,   # actual vocab after TinyStories tokenization
            n_layer     = 32,
            n_embd      = 512,
            n_head      = 8,
            head_size   = 64,
            ff_mult     = 4,
            use_bitnet  = True,
            tie_weights = True,
        )
        train_cfg = TrainConfig(
            data_path     = 'data/tokens.bin',
            teacher_path  = teacher,
            alpha         = 0.7,
            temperature   = 4.0,
            batch_size    = 4,
            grad_accum    = 32,
            seq_len       = 1024,
            max_steps     = 400_000,
            lr            = 3e-4,
            num_workers   = 4,
        )
    else:
        # CPU mode: minimal config for testing
        print("[INFO] CPU mode detected: using small config (4 layers, 256 embd)")
        model_cfg = BitStateConfig(
            vocab_size  = 32011,  # actual vocab after TinyStories tokenization
            n_layer     = 4,
            n_embd      = 256,
            n_head      = 4,
            head_size   = 64,
            ff_mult     = 2,
            use_bitnet  = True,
            tie_weights = True,
        )
        train_cfg = TrainConfig(
            data_path     = 'data/tokens.bin',
            teacher_path  = None,  # No distillation on CPU
            batch_size    = 2,
            grad_accum    = 1,
            seq_len       = 128,
            max_steps     = 10_000,
            lr            = 3e-4,
            num_workers   = 0,  # CPU: 0 workers for compatibility
            dtype         = 'float32',  # CPU: bfloat16 not always available
        )

    Trainer(model_cfg, train_cfg).run()