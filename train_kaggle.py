#!/usr/bin/env python3
"""
Parameter Golf - Kaggle 2xT4 Optimized
SOTA techniques: SP8192 vocab, GPTQ-SDClip, Depth Recurrence, Parallel Residuals, MuonEq-R, TTT

Version: Clean, robust implementation for Kaggle 2xT4 GPUs (T4 = ~3x slower than H100 SXM)
"""

import gc
import glob
import io
import math
import os
import random
import shutil
import subprocess
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

# ============== KAGGLE 2xT4 SETTINGS ==============
# T4 specs: 16GB VRAM each, ~65 TFLOPs vs H100 SXM ~400 TFLOPs
# Batch sizes reduced for T4 memory constraints
_T4_BATCH_TOKENS = 8192          # Half of H100 (16K -> 8K)
_T4_SEQ_LEN = 512                 # Standard seq len
_T4_GRAD_ACCUM = 1                # Faster update cadence for Kaggle iteration loops
_T4_ITERATIONS = 2500             # Fewer iterations due to slower GPU
_T4_MAX_WALLCLOCK = 2700          # Leave buffer for TTT (Kaggle 3hr limit)
_T4_MATRIX_LR = 0.0010            # AdamW proxy-tuned for accum=1 fast Kaggle loops

# ============== COMPRESSION ==============

def compress_state_dict(state_dict, method='brotli'):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    raw = buf.getvalue()
    if method == 'brotli':
        import brotli
        return brotli.compress(raw, quality=11)
    elif method == 'zlib':
        return zlib.compress(raw, level=9)
    return raw

# ============== DOWNLOAD ==============

def download_data(data_path, vocab_size=8192):
    from huggingface_hub import hf_hub_download
    
    data_path = os.path.join(data_path, f"fineweb10B_sp{vocab_size}")
    parent = os.path.dirname(data_path)
    tok_dir = os.path.join(parent, "tokenizers")
    
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)
    
    repo_id = "kevclark/parameter-golf"
    
    # Tokenizer
    tok_path = os.path.join(tok_dir, f"fineweb_{vocab_size}_bpe.model")
    if not os.path.exists(tok_path):
        print(f"Downloading tokenizer...")
        try:
            cached = hf_hub_download(repo_id=repo_id, 
                                   filename=f"datasets/tokenizers/fineweb_{vocab_size}_bpe.model",
                                   repo_type="dataset")
            shutil.copy(cached, tok_path)
        except Exception as e:
            print(f"Failed: {e}")
            vocab_size = 1024
            tok_path = os.path.join(tok_dir, "fineweb_1024_bpe.model")
            data_path = data_path.replace(f"sp8192", "sp1024")
    
    # Data shards
    for prefix, count in [("val", 1), ("train", 2)]:
        for i in range(count):
            local = os.path.join(data_path, f"fineweb_{prefix}_{i:06d}.bin")
            if os.path.exists(local) and os.path.getsize(local) > 1000:
                continue
            remote = f"datasets/datasets/fineweb10B_sp{vocab_size}/fineweb_{prefix}_{i:06d}.bin"
            try:
                cached = hf_hub_download(repo_id=repo_id, filename=remote, repo_type="dataset")
                shutil.copy(cached, local)
                print(f"Downloaded: {os.path.basename(local)}")
            except Exception as e:
                print(f"Failed {remote}: {e}")
    
    return data_path, tok_path, vocab_size

# ============== HYPERPARAMETERS ==============

class H:
    data_path = os.environ.get("DATA_PATH", "/tmp/fineweb-pg")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    
    # Training - T4 optimized defaults
    iterations = int(os.environ.get("ITERATIONS", _T4_ITERATIONS))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", _T4_BATCH_TOKENS))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", _T4_SEQ_LEN))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", _T4_GRAD_ACCUM))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", _T4_MAX_WALLCLOCK))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 50))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))  # SOTA uses 0.72
    
    # Validation
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 50))
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 131072))
    
    # Model
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
    
    # Depth recurrence
    num_loops = int(os.environ.get("NUM_LOOPS", 1))
    loop_start = int(os.environ.get("LOOP_START", 4))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.5))
    
    # QK-Gain
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 2.0))
    
    # Optimizer - T4 tuned
    matrix_lr = float(os.environ.get("MATRIX_LR", _T4_MATRIX_LR))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.0003))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.0003))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.095))  # SOTA uses 0.095
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))  # SOTA uses 0.9965
    
    # Quantization
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 8.0))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 20.0))
    
    # TTT
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))

# ============== DATA ==============

def load_shard(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520:
        raise ValueError(f"Bad shard: {path}")
    num_tokens = int(header[2])
    return torch.from_numpy(np.fromfile(path, dtype="<u2", count=num_tokens, offset=256*4).astype(np.uint16))

class TokenStream:
    def __init__(self, pattern):
        self.files = sorted(Path(p) for p in glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_shard(self.files[0])
        self.pos = 0
    
    def _next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_shard(self.files[self.file_idx])
        self.pos = 0
    
    def take(self, n):
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

# ============== TOKENIZER ==============

def build_luts(sp, vocab_size, device):
    base_bytes = torch.zeros(vocab_size, dtype=torch.int16, device=device)
    has_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary = torch.ones(vocab_size, dtype=torch.bool, device=device)
    for i in range(min(int(sp.vocab_size()), vocab_size)):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i):
            continue
        is_boundary[i] = False
        piece = sp.id_to_piece(i)
        if piece.startswith("▁"):
            has_space[i] = True
            piece = piece[1:]
        base_bytes[i] = len(piece.encode("utf-8"))
    return base_bytes, has_space, is_boundary

def compute_val_bpb(val_loss, val_tokens, base_bytes_lut, device):
    bits_per_token = val_loss / math.log(2)
    val_subset = val_tokens.to(device).long()
    byte_count = base_bytes_lut[val_subset].sum().item()
    tokens_per_byte = val_subset.numel() / max(byte_count, 1)
    return bits_per_token * tokens_per_byte

# ============== QUANTIZATION ==============

def quantize_row(t, clip_std_mult=12.85):
    t = t.float()
    if t.ndim < 2:
        scale = (t.abs().max() / 127).clamp_min(1/127)
        q = torch.round(torch.clamp(t, -127 * scale, 127 * scale) / scale).to(torch.int8)
        return q, scale
    std = t.std(dim=1, keepdim=True)
    clip = std * clip_std_mult
    scale = (clip / 127).clamp_min(1/127)
    q = torch.round(torch.clamp(t, -clip, clip) / scale).to(torch.int8)
    return q, scale.squeeze(-1)

def quantize_state_dict(state_dict):
    quantized = {}
    scales = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        if not t.is_floating_point():
            quantized[name] = t.to(torch.int16)
            continue
        if "tok_emb" in name or "lm_head" in name:
            q, s = quantize_row(t, clip_std_mult=H.embed_clip_sigmas)
        else:
            q, s = quantize_row(t, clip_std_mult=H.matrix_clip_sigmas)
        quantized[name] = q
        scales[name] = s
    return {"q": quantized, "s": scales}

# ============== MUONeq-R ==============

def zeropower_ns5(G, steps=5):
    for _ in range(steps):
        G = 1.5*G - 0.5*G@G.T@G
    return G

class MuonEqR(torch.optim.Optimizer):
    def __init__(self, params, lr=0.04, momentum=0.95, backend_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            steps = group["backend_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim == 2:
                    g_normed = g / (g.norm(dim=1, keepdim=True) + 1e-8)
                    z = zeropower_ns5(g_normed.float(), steps=steps)
                    z = z * max(1, z.size(0) / z.size(1)) ** 0.5
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(g)
                    buf = state["buf"]
                    buf.mul_(mom).add_(z)
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(buf.to(p.dtype), alpha=-lr)
                else:
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(g, alpha=-lr)

# ============== MODEL ==============

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.scale * F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.float(), self.bias.float() if self.bias is not None else None)

class Rotary(nn.Module):
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
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        return torch.cat((torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), -1), x_pass), -1)
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
        
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        
        cos, sin = self.rotary(s, x.device, q.dtype)
        q = apply_rope(q, cos, sin, self.rope_dims)
        k = apply_rope(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.view(1, -1, 1, 1).to(dtype=q.dtype)
        
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q * scale, k.transpose(-2, -1).float())
        causal = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), 1)
        attn = attn.masked_fill(causal, float("-inf"))
        attn = F.softmax(attn, dim=-1).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)

class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.fc = CastedLinear(dim, mult * dim)
        self.proj = CastedLinear(mult * dim, dim)
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx, rope_dims=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)
    
    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * self.attn(self.attn_norm(x_in) * self.ln_scale)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.ln_scale)
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(H.vocab_size, H.model_dim)
        
        # Encoder/decoder split
        self.num_encoder = H.num_layers // 2
        self.num_decoder = H.num_layers - self.num_encoder
        self.num_skip = min(self.num_encoder, self.num_decoder)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip, H.model_dim, dtype=torch.float32))
        
        self.blocks = nn.ModuleList([
            Block(H.model_dim, H.num_heads, H.num_kv_heads, H.mlp_mult, H.rope_base, H.qk_gain_init, i, H.rope_dims)
            for i in range(H.num_layers)
        ])
        
        self.final_norm = RMSNorm(H.model_dim)
        self.lm_head = None if H.tie_embeddings else CastedLinear(H.model_dim, H.vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        if H.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, std=0.005)
        for m in self.modules():
            if isinstance(m, CastedLinear) and m.bias is not None:
                if (m.bias.data == 0).all() or (m.bias.data.abs().sum() == 0):
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, target=None):
        x = self.tok_emb(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        
        # Encoder
        for i in range(self.num_encoder):
            x = self.blocks[i](x, x0)
            skips.append(x)
        
        # Depth recurrence
        if H.enable_looping_at > 0:
            for _ in range(H.num_loops - 1):
                for i in range(H.loop_start, min(H.loop_end + 1, H.num_layers)):
                    x = self.blocks[i](x, x0)
        
        # Decoder with skip connections
        for i in range(self.num_decoder):
            bi = self.num_encoder + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[bi](x, x0)
        
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight) if H.tie_embeddings else self.lm_head(x)
        logits = H.logit_softcap * torch.tanh(logits / H.logit_softcap)
        
        if target is not None:
            return F.cross_entropy(logits.view(-1, H.vocab_size), target.view(-1), reduction='mean')
        return logits

# ============== TRAINING ==============

def train():
    args = H()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    
    # Kaggle 2xT4: torchrun/DDP path.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    is_main = rank == 0
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    os.makedirs("logs", exist_ok=True)
    logf = open(f"logs/{args.run_id}.rank{rank}.txt", "w")
    
    def log(msg):
        if is_main:
            print(msg)
        print(msg, file=logf, flush=True)
    
    log(f"PyTorch: {torch.__version__}")
    log(f"Run ID: {args.run_id}")
    log(f"Device: {torch.cuda.get_device_name()} (rank {rank}/{world_size})")
    log(f"T4 Optimized: batch={_T4_BATCH_TOKENS}, accum={_T4_GRAD_ACCUM}, iters={_T4_ITERATIONS}")
    
    # Download data
    log("[1/8] Downloading data...")
    data_path, tok_path, actual_vocab = download_data(args.data_path, args.vocab_size)
    args.vocab_size = actual_vocab
    args.tokenizer_path = tok_path
    log(f"Using vocab_size={args.vocab_size}")
    
    # Tokenizer
    log("[2/8] Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=tok_path)
    actual_vocab = int(sp.vocab_size())
    if actual_vocab != args.vocab_size:
        log(f"Warning: vocab mismatch {actual_vocab} vs {args.vocab_size}")
        args.vocab_size = actual_vocab
    
    # Val tokens
    log("[3/8] Loading validation...")
    val_tokens = load_shard(os.path.join(data_path, "fineweb_val_000000.bin"))
    base_bytes_lut, _, _ = build_luts(sp, args.vocab_size, device)
    log(f"Val tokens: {val_tokens.numel()}")
    
    # Model
    log("[4/8] Building model...")
    model = GPT().to(device).to(torch.bfloat16)
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if world_size > 1 else model
    
    n_params = sum(p.numel() for p in raw_model.parameters())
    log(f"Params: {n_params} ({n_params*2/1024/1024:.1f}MB bf16)")
    
    opt = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.matrix_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )
    
    # EMA
    ema_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
    
    # Data loader - each GPU loads same data (embarrassingly parallel)
    train_stream = TokenStream(os.path.join(data_path, "fineweb_train_*.bin"))
    if rank > 0:
        train_stream.take(rank * (args.train_batch_tokens + 1))
    
    def get_batch():
        span = args.train_batch_tokens + 1
        chunk = train_stream.take(span)
        x = chunk[:-1].reshape(-1, args.train_seq_len).to(device).long()
        y = chunk[1:].reshape(-1, args.train_seq_len).to(device).long()
        return x, y
    
    def lr_schedule(step, elapsed_ms):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        warmdown_start = int(args.iterations * (1 - args.warmdown_frac))
        if step >= warmdown_start:
            return max((args.iterations - step) / max(args.iterations - warmdown_start, 1), 0.0)
        return 1.0
    
    # Training loop
    log(f"[5/8] Training ({args.iterations} steps, {args.max_wallclock_seconds}s wallclock)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    best_val_bpb = float('inf')
    best_state = None
    grad_accum = 0
    last_val_step = -1
    
    while step < args.iterations:
        elapsed = 1000 * (time.perf_counter() - t0)
        
        # Validate
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
                        x = val_tokens[i:i+args.train_seq_len*4].to(device).long()
                        y = val_tokens[i+1:i+1+args.train_seq_len*4].to(device).long()
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = raw_model(x, y)
                        val_loss_sum += loss.item() * x.numel()
                        val_token_count += x.numel()

                val_loss = val_loss_sum / val_token_count
                val_bpb = compute_val_bpb(val_loss, val_tokens[:val_token_count], base_bytes_lut, device)

                # Update EMA
                current_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                for k in ema_state:
                    ema_state[k] = args.ema_decay * ema_state[k] + (1 - args.ema_decay) * current_state[k]

                if val_bpb < best_val_bpb:
                    best_val_bpb = val_bpb
                    best_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}

                log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} time:{elapsed:.0f}ms")
            model.train()
            if world_size > 1:
                dist.barrier()
        
        # Train step
        x, y = get_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        if not torch.isfinite(loss):
            log(f"Non-finite loss at step {step}: {loss.item()}")
            break
        loss.backward()
        grad_accum += 1
        
        if grad_accum >= args.grad_accum_steps:
            scale = lr_schedule(step, elapsed)
            for g in opt.param_groups:
                g["lr"] = args.matrix_lr * scale
            
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            
            opt.step()
            opt.zero_grad(set_to_none=True)
            
            grad_accum = 0
            step += 1
        
        if elapsed >= args.max_wallclock_seconds * 1000:
            log(f"Wallclock cap at step {step}")
            break
        
        if step <= 10 or step % 100 == 0:
            log(f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} time:{elapsed:.0f}ms")
    
    log(f"[5/8] Training complete. Best val_bpb: {best_val_bpb:.4f}")
    if world_size > 1:
        dist.barrier()
    if not is_main:
        logf.close()
        dist.destroy_process_group()
        return
    
    # Save
    log("[6/8] Saving checkpoints...")
    if best_state:
        torch.save(best_state, "best_model.pt")
        
        quant = quantize_state_dict(best_state)
        buf = io.BytesIO()
        torch.save(quant, buf)
        import brotli
        compressed = brotli.compress(buf.getvalue(), quality=11)
        with open("best_model.int8.br", "wb") as f:
            f.write(compressed)
        
        code_size = len(open(__file__).read().encode())
        log(f"Compressed: {len(compressed)} bytes + code {code_size} = {len(compressed)+code_size} total")
    
    # TTT
    if args.ttt_enabled and best_state:
        log("[7/8] Running TTT...")
        raw_model.load_state_dict(best_state)
        ttt_model = raw_model
        ttt_model.eval()
        
        ttt_opt = torch.optim.SGD(ttt_model.parameters(), lr=args.ttt_lr, momentum=0.9)
        
        for epoch in range(args.ttt_epochs):
            indices = torch.randperm(val_tokens.numel() - 1)[:32768]
            for start in range(0, len(indices), 32768):
                chunk = indices[start:start+32768]
                x = val_tokens[chunk].to(device).long()
                y = val_tokens[chunk+1].to(device).long()
                with torch.no_grad():
                    _ = ttt_model(x, y)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = ttt_model(x, y)
                ttt_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                ttt_opt.step()
        
        # Evaluate
        with torch.no_grad():
            val_loss_sum = 0.0
            val_token_count = 0
            for i in range(0, min(val_tokens.numel() - 1, 32768), args.train_seq_len * 4):
                x = val_tokens[i:i+args.train_seq_len*4].to(device).long()
                y = val_tokens[i+1:i+1+args.train_seq_len*4].to(device).long()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = ttt_model(x, y)
                val_loss_sum += loss.item() * x.numel()
                val_token_count += x.numel()
        
        ttt_val_bpb = compute_val_bpb(val_loss_sum/val_token_count, val_tokens[:val_token_count], base_bytes_lut, device)
        log(f"TTT val_bpb: {ttt_val_bpb:.4f}")
        
        ttt_state = {k: v.cpu() for k, v in ttt_model.state_dict().items()}
        torch.save(ttt_state, "best_model_ttt.pt")
    
    log("[8/8] Done!")
    logf.close()
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    if (
        os.environ.get("USE_DDP", "1") == "1"
        and "LOCAL_RANK" not in os.environ
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={torch.cuda.device_count()}",
            __file__,
        ]
        os.execvp(cmd[0], cmd)
    train()
