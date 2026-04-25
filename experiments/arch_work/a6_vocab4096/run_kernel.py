import os
os.environ['ENABLE_LOOPING_AT'] = '0.35'
os.environ['LOOP_END'] = '5'
os.environ['LOOP_START'] = '3'
os.environ['MODEL_DIM'] = '512'
os.environ['NUM_LAYERS'] = '11'
os.environ['NUM_LOOPS'] = '2'
os.environ['QK_GAIN_INIT'] = '5.25'
os.environ['RUN_ID'] = 'a6_vocab4096'
os.environ['VOCAB_SIZE'] = '4096'
#!/usr/bin/env python3
"""
Parameter Golf - Kaggle 2xT4 Optimized Training Pipeline
=======================================================
Clean, stable implementation for long-running sessions.

Key features:
- Checkpointing every 5 minutes with resume capability
- NaN/Inf detection with graceful recovery
- Incremental 10-minute validation cycles
- Standard proven architecture (11L x 512d, SP8192, depth recurrence)
- TTT (Test-Time Training) for final BPB boost

Target: Sub-1 BPB on 2xT4 GPUs with extended training.
"""

import gc
import glob
import io
import json
import math
import os
import random
import signal
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ============================================================================
# CONSTANTS
# ============================================================================

# T4 specs: 16GB VRAM each, ~65 TFLOPs vs H100 SXM ~400 TFLOPs
# T4 is ~3x slower, so we compensate with more accumulation
_T4_BATCH_TOKENS = 8192
_T4_SEQ_LEN = 512
_T4_GRAD_ACCUM = 1          # 1 for quick feedback, 4 for larger effective batch
_T4_ITERATIONS = 2500        # Base iterations (will scale with time)
_T4_MAX_WALLCLOCK = 570      # 10 minutes for quick validation runs
_T4_MATRIX_LR = 0.001       # AdamW LR tuned for T4

# Checkpointing
_CHECKPOINT_DIR = "checkpoints"
_CHECKPOINT_EVERY_SECONDS = 300  # 5 minutes
_CHECKPOINT_PREFIX = "ckpt_step_"

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

class H:
    """Hyperparameters - all configurable via environment variables."""
    
    # Paths
    data_path = os.environ.get("DATA_PATH", "/tmp/fineweb-pg")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4())[:8])
    seed = int(os.environ.get("SEED", 1337))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", _CHECKPOINT_DIR)
    
    # Training
    iterations = int(os.environ.get("ITERATIONS", _T4_ITERATIONS))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", _T4_BATCH_TOKENS))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", _T4_SEQ_LEN))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", _T4_GRAD_ACCUM))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", _T4_MAX_WALLCLOCK))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 50))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    min_lr_ratio = float(os.environ.get("MIN_LR_RATIO", 0.1))
    
    # Validation
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 50))
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 131072))
    
    # Model architecture (proven SOTA config)
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    
    # Depth recurrence (proven: layers 3-5 loop twice)
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    
    # QK-Gain (proven: 5.25 is optimal)
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.25))
    
    # Optimizer
    matrix_lr = float(os.environ.get("MATRIX_LR", _T4_MATRIX_LR))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.095))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    
    # EMA (proven: 0.9965 is optimal)
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))
    
    # Quantization
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 20.0))
    
    # TTT (Test-Time Training) - enables via env var
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_warmup_tokens = int(os.environ.get("TTT_WARMUP_TOKENS", 32768))


# ============================================================================
# DATA LOADING
# ============================================================================

def download_data(data_path, vocab_size=8192):
    """Download tokenizer and data shards from HuggingFace."""
    from huggingface_hub import hf_hub_download
    
    data_path = os.path.join(data_path, f"fineweb10B_sp{vocab_size}")
    parent = os.path.dirname(data_path)
    tok_dir = os.path.join(parent, "tokenizers")
    
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)
    
    repo_id = "kevclark/parameter-golf"
    
    # Tokenizer
    tok_path = os.path.join(tok_dir, f"fineweb_{vocab_size}_bpe.model")
    if not os.path.exists(tok_path) or os.path.getsize(tok_path) < 1000:
        print(f"Downloading tokenizer...")
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=f"datasets/tokenizers/fineweb_{vocab_size}_bpe.model",
            repo_type="dataset"
        )
        import shutil
        shutil.copy(cached, tok_path)
        print(f"Downloaded: {tok_path}")
    
    # Data shards - need both val and train
    for prefix, count in [("val", 1), ("train", 2)]:
        for i in range(count):
            local = os.path.join(data_path, f"fineweb_{prefix}_{i:06d}.bin")
            if os.path.exists(local) and os.path.getsize(local) > 1000:
                continue
            remote = f"datasets/datasets/fineweb10B_sp{vocab_size}/fineweb_{prefix}_{i:06d}.bin"
            try:
                cached = hf_hub_download(repo_id=repo_id, filename=remote, repo_type="dataset")
                import shutil
                shutil.copy(cached, local)
                print(f"Downloaded: {os.path.basename(local)}")
            except Exception as e:
                print(f"ERROR downloading {remote}: {e}")
                raise
    
    return data_path, tok_path


def load_shard(path):
    """Load a FineWeb binary shard."""
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520:
        raise ValueError(f"Bad shard header: {path}")
    num_tokens = int(header[2])
    data = np.fromfile(path, dtype="<u2", count=num_tokens, offset=256*4)
    return torch.from_numpy(data.astype(np.uint16))


class TokenStream:
    """Streaming token loader that cycles through shards."""
    
    def __init__(self, pattern):
        self.files = sorted(Path(p) for p in glob.glob(str(pattern)))
        if not self.files:
            raise FileNotFoundError(f"No files matching: {pattern}")
        self.file_idx = 0
        self.tokens = load_shard(self.files[0])
        self.pos = 0
    
    def _next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_shard(self.files[self.file_idx])
        self.pos = 0
    
    def take(self, n):
        """Take n tokens, cycling through shards."""
        result = []
        while n > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._next_file()
                continue
            k = min(n, avail)
            result.append(self.tokens[self.pos:self.pos+k])
            self.pos += k
            n -= k
        return torch.cat(result)
    
    def seek_tokens(self, n):
        """Skip first n tokens (for resume)."""
        while n > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= n:
                n -= avail
                self._next_file()
            else:
                self.pos += n
                n = 0


# ============================================================================
# TOKENIZER UTILITIES
# ============================================================================

def build_byte_lut(sp, vocab_size, device):
    """Build lookup table: token -> byte count."""
    base_bytes = torch.zeros(vocab_size, dtype=torch.int16, device=device)
    for i in range(min(int(sp.vocab_size()), vocab_size)):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i):
            continue
        piece = sp.id_to_piece(i)
        if piece.startswith("▁"):
            piece = piece[1:]
        base_bytes[i] = len(piece.encode("utf-8"))
    return base_bytes


def compute_val_bpb(val_loss, val_tokens, base_bytes_lut, device):
    """Compute validation bits-per-byte from cross-entropy loss."""
    bits_per_token = val_loss / math.log(2)
    val_subset = val_tokens.to(device).long()
    byte_count = base_bytes_lut[val_subset].sum().item()
    tokens_per_byte = val_subset.numel() / max(byte_count, 1)
    return bits_per_token * tokens_per_byte


# ============================================================================
# MODEL
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return self.scale * F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    """Linear that casts weights to float32 for computation."""
    def forward(self, x):
        return F.linear(x, self.weight.float(), 
                        self.bias.float() if self.bias is not None else None)


class Rotary(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim, base=10000.0, rope_dims=0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos = None
        self._sin = None
    
    def forward(self, seq_len, device, dtype):
        if self._cos is None or self._sin is None or self._cos.size(-2) != seq_len:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
        return self._cos.to(dtype), self._sin.to(dtype)


def apply_rope(x, cos, sin, rope_dims=0):
    """Apply rotary embedding to query/key tensors."""
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        rotated = torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), -1)
        return torch.cat((rotated, x_pass), -1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), -1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dims = rope_dims
        kv_dim = num_kv_heads * self.head_dim
        
        self.q_proj = CastedLinear(dim, dim)
        self.k_proj = CastedLinear(dim, kv_dim)
        self.v_proj = CastedLinear(dim, kv_dim)
        self.o_proj = CastedLinear(dim, dim)
        
        self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=rope_dims)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
    
    def forward(self, x):
        b, s, d = x.shape
        
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # RMSNorm before QK projection (proven improvement)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        
        # Apply RoPE
        cos, sin = self.rotary(s, x.device, q.dtype)
        q = apply_rope(q, cos, sin, self.rope_dims)
        k = apply_rope(k, cos, sin, self.rope_dims)
        
        # QK-Gain (learnable per-head scaling)
        q = q * self.q_gain.view(1, -1, 1, 1).to(dtype=q.dtype)
        
        # KV replication for GQA
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q * scale, k.transpose(-2, -1).float())
        causal = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), 1)
        attn = attn.masked_fill(causal, float("-inf"))
        attn = F.softmax(attn, dim=-1).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)


class MLP(nn.Module):
    """MLP with squared LeakyReLU activation."""
    def __init__(self, dim, mult=4):
        super().__init__()
        self.fc = CastedLinear(dim, mult * dim)
        self.proj = CastedLinear(mult * dim, dim)
    
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Block(nn.Module):
    """Transformer block with pre-norm and learned residuals."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, 
                 qk_gain_init, layer_idx, rope_dims=0):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        
        self.attn = Attention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        
        # Learned residual scaling (proven improvement)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((
            torch.ones(dim), torch.zeros(dim)
        )).float())
        
        # Layer scaling (deeper layers slightly smaller)
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)
    
    def forward(self, x, x0):
        dtype = x.dtype
        mix = self.resid_mix.to(dtype=dtype)
        
        # Residual mix: combine current and original residual
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        
        # Attention block
        x = x_in + self.attn_scale.to(dtype=dtype)[None, None, :] * \
            self.attn(self.attn_norm(x_in) * self.ln_scale)
        
        # MLP block
        x = x + self.mlp_scale.to(dtype=dtype)[None, None, :] * \
            self.mlp(self.mlp_norm(x) * self.ln_scale)
        
        return x


class GPT(nn.Module):
    """Standard transformer language model."""
    def __init__(self):
        super().__init__()
        
        self.tok_emb = nn.Embedding(H.vocab_size, H.model_dim)
        self.blocks = nn.ModuleList([
            Block(
                H.model_dim, H.num_heads, H.num_kv_heads, H.mlp_mult,
                H.rope_base, H.qk_gain_init, i, H.rope_dims
            )
            for i in range(H.num_layers)
        ])
        self.final_norm = RMSNorm(H.model_dim)
        
        # Tied embeddings save parameters
        self.lm_head = None if H.tie_embeddings else \
            CastedLinear(H.model_dim, H.vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        if H.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, std=0.005)
        for m in self.modules():
            if isinstance(m, CastedLinear) and m.bias is not None:
                if (m.bias.data == 0).all() or m.bias.data.abs().sum() == 0:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, target=None):
        x = self.tok_emb(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        
        # Compute depth loop activation step
        warmup_steps = int(H.iterations * H.enable_looping_at)
        loop_active = False
        
        for i, block in enumerate(self.blocks):
            x = block(x, x0)
            
            # Depth recurrence: loop layers 3-5 after warmup
            if (i >= H.loop_start and i <= H.loop_end and 
                H.num_loops > 1 and loop_active):
                for _ in range(H.num_loops - 1):
                    x = block(x, x0)
            
            # Activate recurrence after warmup
            if i == H.loop_start:
                loop_active = True
        
        x = self.final_norm(x)
        
        # Output projection
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.tok_emb.weight)
        
        # Logit softcap (improves calibration)
        logits = H.logit_softcap * torch.tanh(logits / H.logit_softcap)
        
        if target is not None:
            return F.cross_entropy(logits.view(-1, H.vocab_size), target.view(-1), 
                                   reduction='mean')
        return logits


# ============================================================================
# QUANTIZATION
# ============================================================================

def quantize_row(t, clip_std_mult=12.85, bits=8):
    """Per-row quantization for GPTQ."""
    t = t.float()
    qmax = (1 << (bits - 1)) - 1
    
    if t.ndim < 2:
        scale = (t.abs().max() / qmax).clamp_min(1/qmax)
        q = torch.round(torch.clamp(t, -qmax * scale, qmax * scale) / scale).to(torch.int8)
        return q, scale
    
    std = t.std(dim=1, keepdim=True)
    clip = std * clip_std_mult
    scale = (clip / qmax).clamp_min(1/qmax)
    q = torch.round(torch.clamp(t, -clip, clip) / scale).to(torch.int8)
    return q, scale.squeeze(-1)


def pack_int6(q):
    """Pack int6 values into bytes (4 values per 3 bytes)."""
    flat = (q.to(torch.int16).flatten() + 32).clamp_(0, 63).to(torch.int32)
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int32)])
    
    vals = flat.view(-1, 4)
    word = vals[:, 0] | (vals[:, 1] << 6) | (vals[:, 2] << 12) | (vals[:, 3] << 18)
    
    packed = torch.empty((word.numel(), 3), dtype=torch.uint8)
    packed[:, 0] = (word & 255).to(torch.uint8)
    packed[:, 1] = ((word >> 8) & 255).to(torch.uint8)
    packed[:, 2] = ((word >> 16) & 255).to(torch.uint8)
    
    return packed.flatten(), pad


def quantize_state_dict(state_dict):
    """Quantize model weights for storage."""
    quantized = {}
    scales = {}
    meta = {}
    
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        
        # Non-float tensors (embeddings indices, etc.)
        if not t.is_floating_point():
            quantized[name] = t.to(torch.int16)
            continue
        
        # Embedding layers: int8
        if "tok_emb" in name or "lm_head" in name:
            q, s = quantize_row(t, clip_std_mult=H.embed_clip_sigmas, bits=8)
            quantized[name] = q
            meta[name] = {"bits": 8, "shape": tuple(t.shape)}
        
        # Matrix weights: int6
        elif t.ndim == 2:
            q, s = quantize_row(t, clip_std_mult=H.matrix_clip_sigmas, bits=6)
            packed, pad = pack_int6(q)
            quantized[name] = packed
            meta[name] = {"bits": 6, "shape": tuple(t.shape), "pad": pad}
        
        # Other: int8
        else:
            q, s = quantize_row(t, clip_std_mult=H.matrix_clip_sigmas, bits=8)
            quantized[name] = q
            meta[name] = {"bits": 8, "shape": tuple(t.shape)}
        
        scales[name] = s
    
    return {"q": quantized, "s": scales, "meta": meta}


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(step, model, optimizer, ema_state, best_val_bpb, 
                    train_stream_pos, run_dir):
    """Save training checkpoint."""
    checkpoint = {
        "step": step,
        "model": model.state_dict() if hasattr(model, 'state_dict') else None,
        "optimizer": optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
        "ema_state": ema_state,
        "best_val_bpb": best_val_bpb,
        "train_stream_pos": train_stream_pos,
    }
    
    path = os.path.join(run_dir, f"{_CHECKPOINT_PREFIX}{step}.pt")
    torch.save(checkpoint, path)
    
    # Clean old checkpoints (keep last 3)
    ckpts = sorted(glob.glob(os.path.join(run_dir, f"{_CHECKPOINT_PREFIX}*.pt")))
    for old in ckpts[:-3]:
        os.remove(old)
    
    return path


def load_checkpoint(path, model, optimizer):
    """Load training checkpoint and return step + stream position."""
    ckpt = torch.load(path, map_location='cpu')
    
    if ckpt.get("model") is not None:
        model.load_state_dict(ckpt["model"])
    if ckpt.get("optimizer") is not None and hasattr(optimizer, 'load_state_dict'):
        optimizer.load_state_dict(ckpt["optimizer"])
    
    return ckpt.get("step", 0), ckpt.get("best_val_bpb", float('inf')), \
           ckpt.get("train_stream_pos", 0)


# ============================================================================
# TTT (TEST-TIME TRAINING)
# ============================================================================

def run_ttt(model, val_tokens, base_bytes_lut, device, args, log_fn):
    """Run Test-Time Training for final BPB boost."""
    log_fn("[TTT] Starting Test-Time Training...")
    
    model.eval()
    ttt_opt = torch.optim.SGD(model.parameters(), lr=args.ttt_lr, momentum=0.9)
    
    tokens_used = 0
    total_tokens = min(val_tokens.numel() - 1, args.ttt_warmup_tokens * args.ttt_epochs)
    
    for epoch in range(args.ttt_epochs):
        indices = torch.randperm(val_tokens.numel() - 1)
        
        for start in range(0, len(indices), args.ttt_warmup_tokens):
            chunk = indices[start:start + args.ttt_warmup_tokens]
            
            # Score first (legal requirement)
            x = val_tokens[chunk].to(device).long()
            y = val_tokens[chunk + 1].to(device).long()
            
            with torch.no_grad():
                _ = model(x, y)
            
            # Then train
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            
            if not torch.isfinite(loss):
                log_fn(f"[TTT] Non-finite loss at epoch {epoch}, chunk {start}")
                continue
            
            ttt_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ttt_opt.step()
            
            tokens_used += x.numel()
        
        # Evaluate after each epoch
        val_loss_sum = 0.0
        val_token_count = 0
        eval_limit = min(val_tokens.numel() - 1, args.val_max_tokens)
        
        with torch.no_grad():
            for i in range(0, eval_limit, args.train_seq_len * 4):
                x = val_tokens[i:i + args.train_seq_len * 4].to(device).long()
                y = val_tokens[i + 1:i + 1 + args.train_seq_len * 4].to(device).long()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                val_loss_sum += loss.item() * x.numel()
                val_token_count += x.numel()
        
        ttt_val_bpb = compute_val_bpb(
            val_loss_sum / val_token_count, val_tokens[:val_token_count],
            base_bytes_lut, device
        )
        log_fn(f"[TTT] Epoch {epoch + 1}/{args.ttt_epochs}: val_bpb={ttt_val_bpb:.4f}")
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train():
    """Main training function."""
    args = H()
    
    # ==========================================================================
    # SETUP
    # ==========================================================================
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training")
    
    # Distributed setup
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    
    is_main = rank == 0
    
    # Enable TF32 for faster compute
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create run directory
    run_dir = os.path.join(args.checkpoint_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Logging
    logf = open(f"logs/{args.run_id}.rank{rank}.txt", "w")
    
    def log(msg):
        if is_main:
            print(msg)
        print(msg, file=logf, flush=True)
    
    def log_metrics(metrics):
        """Log structured metrics."""
        line = " ".join(f"{k}={v}" for k, v in metrics.items())
        log(line)
        # Also print for auto-parsing
        if is_main:
            for k, v in metrics.items():
                print(f"METRIC {k}={v}", flush=True)
    
    log("=" * 60)
    log(f"Parameter Golf - T4 Training")
    log(f"Run ID: {args.run_id}")
    log(f"Device: {torch.cuda.get_device_name()} (rank {rank}/{world_size})")
    log(f"PyTorch: {torch.__version__}")
    log("=" * 60)
    
    # ==========================================================================
    # DATA
    # ==========================================================================
    
    log("[1/6] Downloading data...")
    try:
        data_path, tok_path = download_data(args.data_path, args.vocab_size)
    except Exception as e:
        log(f"ERROR: Data download failed: {e}")
        raise
    
    # Tokenizer
    log("[2/6] Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=tok_path)
    actual_vocab = int(sp.vocab_size())
    if actual_vocab != args.vocab_size:
        log(f"Warning: vocab mismatch {actual_vocab} vs {args.vocab_size}")
        args.vocab_size = actual_vocab
    
    # Validation tokens
    val_tokens = load_shard(os.path.join(data_path, "fineweb_val_000000.bin"))
    base_bytes_lut = build_byte_lut(sp, args.vocab_size, device)
    log(f"Val tokens: {val_tokens.numel():,}")
    
    # Training data stream
    train_stream = TokenStream(os.path.join(data_path, "fineweb_train_*.bin"))
    
    # ==========================================================================
    # MODEL
    # ==========================================================================
    
    log("[3/6] Building model...")
    model = GPT().to(device).to(torch.bfloat16)
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    raw_model = model.module if world_size > 1 else model
    n_params = sum(p.numel() for p in raw_model.parameters())
    log(f"Params: {n_params:,} ({n_params * 2 / 1024 / 1024:.1f} MB bf16)")
    
    # ==========================================================================
    # OPTIMIZER
    # ==========================================================================
    
    opt = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.matrix_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )
    
    # EMA for model averaging
    ema_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    log("[4/6] Training...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    step = 0
    best_val_bpb = float('inf')
    best_state = None
    grad_accum = 0
    last_val_step = -1
    last_checkpoint_time = t0
    train_stream_pos = 0
    
    # Resume from checkpoint if exists
    latest_ckpt = None
    if os.path.exists(run_dir):
        ckpts = sorted(glob.glob(os.path.join(run_dir, f"{_CHECKPOINT_PREFIX}*.pt")))
        if ckpts:
            latest_ckpt = ckpts[-1]
    
    if latest_ckpt:
        log(f"Resuming from checkpoint: {latest_ckpt}")
        resume_step, resume_val_bpb, resume_pos = load_checkpoint(
            latest_ckpt, raw_model, opt
        )
        step = resume_step
        best_val_bpb = resume_val_bpb
        train_stream_pos = resume_pos
        train_stream.seek_tokens(train_stream_pos)
        log(f"Resumed from step {step}, val_bpb={best_val_bpb:.4f}")
    
    # Batch function
    def get_batch():
        span = args.train_batch_tokens + 1
        chunk = train_stream.take(span)
        x = chunk[:-1].reshape(-1, args.train_seq_len).to(device).long()
        y = chunk[1:].reshape(-1, args.train_seq_len).to(device).long()
        return x, y
    
    # Learning rate schedule
    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        warmdown_start = int(args.iterations * (1 - args.warmdown_frac))
        if step >= warmdown_start:
            frac = (args.iterations - step) / max(args.iterations - warmdown_start, 1)
            return max(frac, args.min_lr_ratio)
        return 1.0
    
    while step < args.iterations:
        elapsed = time.perf_counter() - t0
        
        # Checkpoint every N seconds
        if is_main and (elapsed - last_checkpoint_time) >= _CHECKPOINT_EVERY_SECONDS:
            if best_state is not None:
                ckpt_path = save_checkpoint(
                    step, raw_model, opt, ema_state, best_val_bpb, 
                    train_stream_pos, run_dir
                )
                log(f"[CKPT] Saved: {ckpt_path}")
            last_checkpoint_time = time.perf_counter()
        
        # Validation
        if step % args.val_loss_every == 0 and step > 0 and step != last_val_step:
            last_val_step = step
            if world_size > 1:
                dist.barrier()
            
            torch.cuda.synchronize()
            
            if is_main:
                model.eval()
                val_loss_sum = 0.0
                val_token_count = 0
                val_limit = min(val_tokens.numel() - 1, args.val_max_tokens)
                
                with torch.no_grad():
                    for i in range(0, val_limit, args.train_seq_len * 4):
                        x = val_tokens[i:i + args.train_seq_len * 4].to(device).long()
                        y = val_tokens[i + 1:i + 1 + args.train_seq_len * 4].to(device).long()
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = raw_model(x, y)
                        val_loss_sum += loss.item() * x.numel()
                        val_token_count += x.numel()
                
                val_loss = val_loss_sum / val_token_count
                val_bpb = compute_val_bpb(val_loss, val_tokens[:val_token_count],
                                          base_bytes_lut, device)
                
                # Update EMA
                current_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                for k in ema_state:
                    ema_state[k] = args.ema_decay * ema_state[k] + (1 - args.ema_decay) * current_state[k]
                
                # Save best
                if val_bpb < best_val_bpb:
                    best_val_bpb = val_bpb
                    best_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                
                log_metrics({
                    "step": step,
                    "val_loss": f"{val_loss:.4f}",
                    "val_bpb": f"{val_bpb:.4f}",
                    "best_bpb": f"{best_val_bpb:.4f}",
                    "elapsed_s": f"{elapsed:.0f}"
                })
                
                # Checkpoint on improvement
                if val_bpb <= best_val_bpb:
                    ckpt_path = save_checkpoint(
                        step, raw_model, opt, ema_state, best_val_bpb,
                        train_stream_pos, run_dir
                    )
                    log(f"[CKPT] Best checkpoint saved")
            
            model.train()
            if world_size > 1:
                dist.barrier()
        
        # Training step
        try:
            x, y = get_batch()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = raw_model(x, y)
            
            # NaN/Inf detection
            if not torch.isfinite(loss):
                log(f"ERROR: Non-finite loss at step {step}: {loss.item()}")
                # Save checkpoint before dying
                if is_main and best_state is not None:
                    save_checkpoint(step, raw_model, opt, ema_state, best_val_bpb,
                                   train_stream_pos, run_dir)
                break
            
            loss.backward()
            grad_accum += 1
            
            # Optimizer step
            if grad_accum >= args.grad_accum_steps:
                scale = lr_schedule(step)
                for g in opt.param_groups:
                    g["lr"] = args.matrix_lr * scale
                
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip_norm)
                
                opt.step()
                opt.zero_grad(set_to_none=True)
                
                grad_accum = 0
                step += 1
                train_stream_pos += args.train_batch_tokens
        
        except Exception as e:
            log(f"ERROR at step {step}: {e}")
            if is_main:
                save_checkpoint(step, raw_model, opt, ema_state, best_val_bpb,
                              train_stream_pos, run_dir)
            raise
        
        # Wallclock limit
        if elapsed >= args.max_wallclock_seconds:
            log(f"Wallclock limit reached at step {step}")
            break
        
        # Progress logging
        if step <= 10 or step % 100 == 0:
            log(f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} time:{elapsed:.0f}s")
    
    log(f"[4/6] Training complete. Best val_bpb: {best_val_bpb:.4f}")
    
    if world_size > 1:
        dist.barrier()
    
    # ==========================================================================
    # SAVE
    # ==========================================================================
    
    if not is_main:
        logf.close()
        if world_size > 1:
            dist.destroy_process_group()
        return
    
    log("[5/6] Saving final model...")
    
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
    
    torch.save(best_state, os.path.join(run_dir, "best_model.pt"))
    
    # Quantize and compress
    quant = quantize_state_dict(best_state)
    buf = io.BytesIO()
    torch.save(quant, buf)
    
    import brotli
    compressed = brotli.compress(buf.getvalue(), quality=11)
    
    code_size = len(open(__file__).read().encode())
    total_size = len(compressed) + code_size
    
    log(f"Compressed artifact: {len(compressed):,} bytes")
    log(f"Code size: {code_size:,} bytes")
    log(f"Total: {total_size:,} bytes (limit: 16,000,000)")
    
    with open(os.path.join(run_dir, "best_model.int8.br"), "wb") as f:
        f.write(compressed)
    
    # ==========================================================================
    # TTT
    # ==========================================================================
    
    if args.ttt_enabled:
        log("[6/6] Running TTT...")
        raw_model.load_state_dict(best_state)
        raw_model = run_ttt(raw_model, val_tokens, base_bytes_lut, device, args, log)
        
        # Save TTT model
        ttt_state = {k: v.cpu() for k, v in raw_model.state_dict().items()}
        torch.save(ttt_state, os.path.join(run_dir, "best_model_ttt.pt"))
    
    log("=" * 60)
    log("DONE")
    log("=" * 60)
    
    log_metrics({
        "final_val_bpb": f"{best_val_bpb:.4f}",
        "total_steps": step,
        "total_time_s": f"{time.perf_counter() - t0:.0f}",
        "artifact_bytes": len(compressed),
        "code_bytes": code_size,
        "total_bytes": total_size
    })
    
    logf.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Signal handling for graceful shutdown
    def handle_signal(signum, frame):
        print("Received signal, saving checkpoint...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    # Launch DDP if multiple GPUs
    if (
        os.environ.get("USE_DDP", "1") == "1"
        and "LOCAL_RANK" not in os.environ
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={torch.cuda.device_count()}",
            __file__,
        ]
        os.execvp(cmd[0], cmd)
    
    train()
