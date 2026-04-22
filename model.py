"""
BitStateLM: sub-1GB Language Model with BitNet 1.58-bit weights and RWKV-style attention.

Key features:
  - 1.58-bit ternary weights (-1, 0, +1) via native STE (torch.compile compatible)
  - Chunked WKV computation with gradient checkpointing for training
  - O(1) memory inference path for autoregressive generation
  - MatMul-free INT8 inference mode
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# --- Config ---

@dataclass
class BitStateConfig:
    vocab_size:  int   = 32000
    n_layer:     int   = 32
    n_embd:      int   = 512
    n_head:      int   = 8
    head_size:   int   = 64      # must equal n_embd // n_head
    ff_mult:     int   = 4
    max_seq_len: int   = 2048
    dropout:     float = 0.1
    use_bitnet:  bool  = True
    tie_weights: bool  = True

# Backward compatibility with old checkpoints
PicoConfig = BitStateConfig


# --- BitNet 1.58-bit ---

# --- Native STE quantization ---
# Native STE via detach trick: y = (round(x) - x).detach() + x
# Forward: rounded values; Backward: gradient = 1 everywhere (STE).
# Pure tensor ops allow torch.compile to fuse into single Triton kernel.

def _weight_quant_ste(w):
    # Detach scale to avoid dense Jacobian and gradient leakage
    scale = w.detach().abs().mean().clamp(min=1e-8)
    w_norm = w / scale
    w_clip = w_norm.clamp(-1.0, 1.0)
    # Native STE: gradient flows with value 1 everywhere
    w_q = (w_clip.round() - w_norm).detach() + w_norm
    return w_q * scale


def _act_quant_ste(x):
    # Fast L1-based clipping (O(N)) instead of slow quantile (O(N log N))
    clip_val = x.detach().abs().mean(dim=-1, keepdim=True).clamp(min=1e-8) * 2.5
    scale = clip_val / 127.0
    # Scale first to utilize all 255 INT8 levels
    x_scaled = x / scale
    x_clip = x_scaled.clamp(-127.0, 127.0)
    # Native STE for activations
    x_q = (x_clip.round() - x_scaled).detach() + x_scaled
    return x_q * scale


class BitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with 1.58-bit ternary weights and INT8 activations.

    Training mode:
      weight: FP32 parameter, quantized via STE in forward
      input:  normalized via LN, then quantized to INT8 via STE
      → gradients flow through both STEs in backprop

    Inference mode (after to_inference_mode()):
      weight: INT8 buffer (not a parameter!). FP32 copy is deleted from memory
      input:  quantized to INT8 similarly
      → INT8×INT8 matmul, dequantized via two scalar scales
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_f  = in_features
        self.out_f = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # elementwise_affine=True: learnable γ and β are critical for BitNet.
        # Before {-1,0,+1} quantization, the model must scale channels.
        # γ allows pushing important channels past the round() threshold → +1 or -1,
        # and suppressing insignificant ones → 0. Without γ/β the network loses
        # this routing mechanism and hits a loss plateau.
        self.ln     = nn.LayerNorm(in_features, elementwise_affine=True)
        self._is_inference = False
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_inference:
            return self._infer_forward(x)
        return self._train_forward(x)

    def _train_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normed = self.ln(x)
        x_q = _act_quant_ste(x_normed)    # INT8 activations (native STE, compile-safe)
        w_q = _weight_quant_ste(self.weight)  # Ternary weights (native STE, compile-safe)
        return F.linear(x_q, w_q)

    def _infer_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        INT8 inference: weights stored as INT8 buffer, no FP32 copy in memory.

        Note on strides: `.t()` returns non-contiguous view. `torch._int_mm` and cublas
        require C-contiguous tensors. Weights are pre-transposed and stored contiguous
        in `_w_i8_t` (prepared in to_inference_mode).
        """
        x_n = self.ln(x)
        # O(N) outlier-robust scale: abs_mean × 2.5 ≈ 99th percentile for
        # normalized activations after LN. No sorting, Triton-fusable.
        clip_val = x_n.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8) * 2.5
        a_scale  = clip_val / 127.0
        x_clipped = x_n.clamp(-clip_val, clip_val)
        x_i8 = (x_clipped / a_scale).round().clamp(-128, 127).to(torch.int8)
        flat = x_i8.reshape(-1, self.in_f)  # contiguous by construction

        if flat.is_cuda and hasattr(torch, "_int_mm"):
            try:
                # _w_i8_t: pre-transposed + contiguous, prepared in to_inference_mode
                out = torch._int_mm(flat, self._w_i8_t).reshape(*x.shape[:-1], self.out_f).float()
            except RuntimeError:
                # Fallback: cast to float (slower but always works)
                out = F.linear(flat.float(), self._w_i8.float()).reshape(*x.shape[:-1], self.out_f)
        else:
            out = F.linear(flat.float(), self._w_i8.float()).reshape(*x.shape[:-1], self.out_f)

        return out * (self._w_scale * a_scale)

    @torch.no_grad()
    def to_inference_mode(self):
        """
        Convert layer to INT8 inference mode.
        Critical: `del self.weight` removes FP32 parameter from memory,
        freeing 4× memory compared to keeping FP32. Without this, INT8 buffer
        would simply be added to FP32, doubling memory usage.
        """
        w_scale = self.weight.abs().mean().clamp(min=1e-8)
        w_i8    = (self.weight / w_scale).round().clamp(-1, 1).to(torch.int8)

        self.register_buffer('_w_i8',    w_i8)
        # Pre-transpose + contiguous: torch._int_mm requires contiguous.
        # Done once here, not on every forward pass (.t() = non-contiguous view).
        self.register_buffer('_w_i8_t',  w_i8.t().contiguous())
        self.register_buffer('_w_scale', w_scale.clone())

        del self.weight
        self._parameters.pop('weight', None)
        self._is_inference = True

    def extra_repr(self):
        mode = 'INT8-inference' if self._is_inference else 'FP32→ternary-training'
        return f'{self.in_f}→{self.out_f} [{mode}]'


def _linear(use_bitnet: bool, in_f: int, out_f: int) -> nn.Module:
    return BitLinear(in_f, out_f) if use_bitnet else nn.Linear(in_f, out_f, bias=False)


# --- WKV (parallel prefix-scan, no Python loop) ---

def _wkv_chunk_fn(
    r_c: torch.Tensor,      # (B, H, C, S)
    k_c: torch.Tensor,
    v_c: torch.Tensor,
    state: torch.Tensor,    # (B, H, S, S); state from previous chunk
    w: torch.Tensor,        # (H, S)
    u_exp: torch.Tensor,    # (H, S)
    log_w: torch.Tensor,    # (H, S)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute one WKV chunk. Called directly or via gradient checkpointing.
    With checkpointing, intermediate tensors (M, A_intra, A_cross, kv_c)
    are not saved in memory during forward; recomputed in backward.

    Returns: (y_c, new_state)
      y_c:       (B, H, C, S)
      new_state: (B, H, S, S)
    """
    B, H, C, S = r_c.shape
    device, dtype = r_c.device, r_c.dtype

    # kv_c[b,h,t,s1,s2] = k[t,s1] * v[t,s2]
    kv_c = k_c.unsqueeze(-1) * v_c.unsqueeze(-2)   # (B, H, C, S, S)

    # Causal intra-chunk decay matrix (C×C, H, S)
    t_idx = torch.arange(C, device=device, dtype=dtype)
    diff  = (t_idx.unsqueeze(0) - t_idx.unsqueeze(1)).clamp(min=0)  # (C, C)
    causal = (t_idx.unsqueeze(0) >= t_idx.unsqueeze(1)).to(dtype)   # (C, C)
    log_M  = diff.unsqueeze(-1).unsqueeze(-1) * log_w.unsqueeze(0).unsqueeze(0)
    M      = causal.unsqueeze(-1).unsqueeze(-1) * torch.exp(log_M)  # (C,C,H,S)

    # Intra-chunk accumulation: A[t] = Σ_{i≤t} w^(t-i) ⊙ kv[i]
    A_intra = torch.einsum('tihS,bhiSo->bhtSo', M, kv_c)  # (B,H,C,S,S)

    # Cross-chunk: contribution from previous chunk state
    # At local time t: cross = w^(t+1) ⊙ state
    t_p1 = (t_idx + 1).unsqueeze(-1).unsqueeze(-1)         # (C,1,1)
    decay_cross = torch.exp(t_p1 * log_w.unsqueeze(0))     # (C, H, S)
    # decay_cross is (C,H,S): permute to (H,C,S) first for correct broadcast
    A_cross = decay_cross.permute(1,0,2).unsqueeze(0).unsqueeze(-1) * state.unsqueeze(2)  # (B,H,C,S,S)

    # Output with u-bonus for current token
    A_u = A_intra + A_cross + u_exp.unsqueeze(0).unsqueeze(2).unsqueeze(-1) * kv_c
    y_c = torch.einsum('bhts,bhtso->bhto', r_c, A_u)       # (B, H, C, S)

    # Update state for next chunk:
    # new_state = w^C ⊙ state + Σ_{i=0}^{C-1} w^(C-1-i) ⊙ kv[i]
    decay_to_end = torch.exp(
        (C - 1 - t_idx).unsqueeze(-1).unsqueeze(-1) * log_w.unsqueeze(0)
    )  # (C, H, S)
    w_C       = torch.exp(C * log_w)                        # (H, S)
    new_state = (w_C.unsqueeze(0).unsqueeze(-1) * state +
                 torch.einsum('chs,bhcso->bhso', decay_to_end, kv_c))

    return y_c, new_state


def wkv_train(
    r: torch.Tensor,          # (B, H, T, S)
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,  # (H, S)
    u: torch.Tensor,          # (H, S)
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunked WKV for training with gradient checkpointing.

    Recurrence:
        state_t = w ⊙ state_{t-1} + k_t ⊗ v_t
        y_t     = r_t · (state_t + u ⊙ k_t ⊗ v_t)

    Naive implementation builds M matrix (T×T×H×S) and tensor A (B×H×T×S×S)
    For B=128, T=1024 this is ~68 GB VRAM.

    This implementation processes T in chunks of chunk_size (default 64):
      - Peak memory per chunk: O(C²×H×S + B×H×C×S²)
        Example (B=4, C=64, H=8, S=64): ~52 MB per chunk vs 2+ GB naive
      - Gradient checkpointing recomputes chunk activations in backward:
        intermediates (M, A_intra, A_cross, kv_c) don't stay in memory
      - State (B×H×S²) passed between chunks as regular tensor

    Total VRAM: O(T/C × B×H×S × C + B×H×S²) = O(T×B×H×S + B×H×S²)
    vs. O(T²×H×S + B×H×T×S²) for naive approach.
    """
    from torch.utils.checkpoint import checkpoint as grad_ckpt

    B, H, T, S = r.shape
    device, dtype = r.device, r.dtype

    w     = torch.exp(-torch.exp(log_decay))        # (H, S), ∈ (0,1)
    u_exp = torch.exp(u)                            # (H, S)
    log_w = torch.log(w.clamp(min=1e-38))           # (H, S)

    y_chunks: list = []
    # Store each state as separate node in autograd graph.
    # If we reused one variable `state = grad_ckpt(...)`, Python would lose
    # reference to previous tensor; backward through chunk chain impossible
    # (gradient only flows through last chunk).
    # Solution: accumulate in list, always pass states[-1].
    states: list = [torch.zeros(B, H, S, S, device=device, dtype=dtype)]

    for t0 in range(0, T, chunk_size):
        t1  = min(t0 + chunk_size, T)
        r_c = r[:, :, t0:t1, :]
        k_c = k[:, :, t0:t1, :]
        v_c = v[:, :, t0:t1, :]

        if r_c.requires_grad:
            # Gradient checkpointing: chunk intermediate tensors not stored.
            # Only inputs (r_c, k_c, v_c, state) and outputs (y_c, new_state) saved.
            y_c, new_state = grad_ckpt(
                _wkv_chunk_fn, r_c, k_c, v_c, states[-1], w, u_exp, log_w,
                use_reentrant=False,
            )
        else:
            y_c, new_state = _wkv_chunk_fn(r_c, k_c, v_c, states[-1], w, u_exp, log_w)

        y_chunks.append(y_c)
        states.append(new_state)

    return torch.cat(y_chunks, dim=2), states[-1]   # (B, H, T, S), (B, H, S, S)


def wkv_infer_step(
    r: torch.Tensor,          # (B, H, 1, S)
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,  # (H, S)
    u: torch.Tensor,
    state: torch.Tensor,      # (B, H, S, S)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single step of autoregressive inference (T=1). O(H×S²) memory, O(1) state.
    Uses explicit state. No computation graph through time.
    """
    B, H, _, S = r.shape
    w     = torch.exp(-torch.exp(log_decay))   # (H, S)
    u_exp = torch.exp(u)

    k0 = k[:, :, 0, :]    # (B, H, S)
    v0 = v[:, :, 0, :]
    r0 = r[:, :, 0, :]

    kv    = k0.unsqueeze(-1) * v0.unsqueeze(-2)                             # (B,H,S,S)
    # state: (B,H,S,S), u_exp: (H,S), kv: (B,H,S,S)
    u_kv  = u_exp.unsqueeze(0).unsqueeze(-1) * kv                           # (B,H,S,S)
    y     = torch.einsum('bhs,bhso->bho', r0, state + u_kv)
    new_s = w.unsqueeze(0).unsqueeze(-1) * state + kv

    return y.unsqueeze(2), new_s   # (B,H,1,S), (B,H,S,S)


# --- TimeMix ---

class TimeMix(nn.Module):
    def __init__(self, cfg: BitStateConfig, layer_id: int):
        super().__init__()
        D, H, S = cfg.n_embd, cfg.n_head, cfg.head_size
        self.H, self.S = H, S

        self.mu_r = nn.Parameter(torch.zeros(1, 1, D))
        self.mu_k = nn.Parameter(torch.zeros(1, 1, D))
        self.mu_v = nn.Parameter(torch.zeros(1, 1, D))
        self.mu_g = nn.Parameter(torch.zeros(1, 1, D))

        self.log_decay = nn.Parameter(torch.zeros(H, S))
        self.u         = nn.Parameter(torch.zeros(H, S))

        self.r_proj = _linear(cfg.use_bitnet, D, D)
        self.k_proj = _linear(cfg.use_bitnet, D, D)
        self.v_proj = _linear(cfg.use_bitnet, D, D)
        self.g_proj = _linear(cfg.use_bitnet, D, D)
        self.o_proj = _linear(cfg.use_bitnet, D, D)
        self.gn     = nn.GroupNorm(H, D, eps=1e-5)

        self._init_params(layer_id, cfg.n_layer)

    def _init_params(self, lid, n_layer):
        with torch.no_grad():
            for p in [self.mu_r, self.mu_k, self.mu_v, self.mu_g]:
                nn.init.uniform_(p, -1e-4, 1e-4)
            decay = torch.arange(self.S, dtype=torch.float32) / max(self.S - 1, 1)
            # Earlier layers: faster decay (local); deeper: slower (global)
            base_decay = -4.0 + 3.0 * (lid / max(n_layer - 1, 1))
            self.log_decay.data = (base_decay + 2.0 * decay).unsqueeze(0).expand(self.H, -1).clone()
            nn.init.constant_(self.u, math.log(0.3))

    def forward(self, x, state=None):
        B, T, C = x.shape
        H, S = self.H, self.S

        prev = state[0] if state is not None else torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        x_prev = torch.cat([prev, x[:, :-1, :]], dim=1)
        dx = x_prev - x

        r = self.r_proj(x + dx * self.mu_r).view(B, T, H, S).permute(0, 2, 1, 3)
        k = self.k_proj(x + dx * self.mu_k).view(B, T, H, S).permute(0, 2, 1, 3)
        v = self.v_proj(x + dx * self.mu_v).view(B, T, H, S).permute(0, 2, 1, 3)
        g = F.silu(self.g_proj(x + dx * self.mu_g))

        wkv_state = state[1] if state is not None else torch.zeros(B, H, S, S, device=x.device, dtype=x.dtype)

        if T == 1:
            # Autoregressive inference: O(1) memory, uses explicit state
            y, new_wkv = wkv_infer_step(r, k, v, self.log_decay, self.u, wkv_state)
        else:
            # Training: parallel prefix-scan, no Python loop.
            # wkv_train returns (y, final_state) for KV cache preservation.
            y, new_wkv = wkv_train(r, k, v, self.log_decay, self.u, chunk_size=64)

        y = y.permute(0, 2, 1, 3).contiguous().view(B * T, C)
        y = self.gn(y).view(B, T, C)
        out = self.o_proj(y * g)

        new_state = (x[:, -1:].detach(), new_wkv.detach())
        return out, new_state


# --- ChannelMix ---

class ChannelMix(nn.Module):
    def __init__(self, cfg: BitStateConfig):
        super().__init__()
        D, hidden = cfg.n_embd, cfg.n_embd * cfg.ff_mult
        self.mu_k = nn.Parameter(torch.zeros(1, 1, D))
        self.mu_r = nn.Parameter(torch.zeros(1, 1, D))
        self.k    = _linear(cfg.use_bitnet, D, hidden)
        self.r    = _linear(cfg.use_bitnet, D, D)
        self.v    = _linear(cfg.use_bitnet, hidden, D)

    def forward(self, x, state=None):
        B, T, C = x.shape
        prev = state if state is not None else torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        x_prev = torch.cat([prev, x[:, :-1, :]], dim=1)
        dx = x_prev - x
        k_act = torch.relu(self.k(x + dx * self.mu_k)) ** 2
        rgate = torch.sigmoid(self.r(x + dx * self.mu_r))
        return rgate * self.v(k_act), x[:, -1:].detach()


# --- Block + Model ---

class BitStateBlock(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.tm  = TimeMix(cfg, layer_id)
        self.cm  = ChannelMix(cfg)

    def forward(self, x, state=None):
        tm_s = state.get('tm') if state else None
        cm_s = state.get('cm') if state else None
        dx, ntm = self.tm(self.ln1(x), tm_s)
        x = x + dx
        dx, ncm = self.cm(self.ln2(x), cm_s)
        return x + dx, {'tm': ntm, 'cm': ncm}


class BitStateLM(nn.Module):
    def __init__(self, cfg: BitStateConfig):
        super().__init__()
        self.cfg    = cfg
        self.emb    = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop   = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([BitStateBlock(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f   = nn.LayerNorm(cfg.n_embd)
        self.head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.emb.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('o_proj.weight') or name.endswith('v.weight'):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, BitLinear):
            if hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None: nn.init.ones_(m.weight)
            if m.bias   is not None: nn.init.zeros_(m.bias)

    def forward(self, idx, states=None, targets=None):
        x = self.drop(self.emb(idx))
        new_states = []
        for i, block in enumerate(self.blocks):
            x, ns = block(x, states[i] if states else None)
            new_states.append(ns)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, new_states, loss

    def to_inference_mode(self):
        """
        Convert all BitLinear layers to INT8 inference mode.
        FP32 parameters are deleted, only INT8 buffers remain in memory.
        Call once before deployment.
        """
        converted = sum(1 for m in self.modules() if isinstance(m, BitLinear) and not m._is_inference)
        for m in self.modules():
            if isinstance(m, BitLinear) and not m._is_inference:
                m.to_inference_mode()
        return self

    @torch.inference_mode()
    def generate(self, prompt, max_new=200, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.1):
        states = None
        logits, states, _ = self(prompt, states)
        generated, seen = [], {}
        for _ in range(max_new):
            nl = logits[:, -1, :].clone().float()
            # Replace any inf/nan before proceeding
            nl = torch.nan_to_num(nl, nan=0.0, posinf=1e4, neginf=-1e4)
            nl = nl / max(temperature, 1e-8)
            if repetition_penalty != 1.0:
                for tid, cnt in seen.items():
                    nl[:, tid] /= repetition_penalty ** min(cnt, 3)
            if top_k > 0:
                k_val = min(top_k, nl.size(-1))
                kth = torch.topk(nl, k_val).values[:, -1, None]
                nl  = nl.masked_fill(nl < kth, -1e9)
            if top_p < 1.0:
                sl, si = torch.sort(nl, descending=True)
                probs  = F.softmax(sl, dim=-1)
                cp     = torch.cumsum(probs, dim=-1)
                remove = (cp - probs) > top_p
                nl.scatter_(1, si, remove.float() * -1e9)
            probs = F.softmax(nl, dim=-1)
            # Safety: if all probs are 0 (shouldn't happen), fallback to uniform
            if probs.sum() < 1e-8:
                probs = torch.ones_like(probs) / probs.size(-1)
            tok = torch.multinomial(probs, 1)
            generated.append(tok)
            seen[tok.item()] = seen.get(tok.item(), 0) + 1
            logits, states, _ = self(tok, states)
        return torch.cat(generated, dim=1)

    def memory_breakdown(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        no_emb = sum(p.numel() for n, p in self.named_parameters() if 'emb' not in n)
        n_lay  = self.cfg.n_layer
        D      = self.cfg.n_embd
        return {
            'total_params':         total,
            'non_emb_params':       no_emb,
            'emb_fp32_mb':          self.emb.weight.numel() * 4 / 1024**2,
            'weights_int8_mb':      no_emb * 1   / 1024**2,
            'weights_ternary_mb':   no_emb * 1.58 / 8 / 1024**2,
            'rwkv_state_mb_fixed':  n_lay * D * D * 4 / 1024**2,
        }
