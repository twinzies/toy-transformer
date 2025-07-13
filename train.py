from models import Pinky, PinkyConfig, CustomBlock
from modules import FeedForwardGeGLU, FeedForwardSwiGLU, LayerNorm, RMSNorm, CausalSelfAttention, HiPAttention, FlashAttention3, MultiHeadLatentAttention
import torch
import torch.nn as nn
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Pinky Transformer')

# Model architecture
parser.add_argument('--norm', type=str, default='RMSNorm', choices=['LayerNorm', 'RMSNorm'],
                    help='Normalization layer to use')
parser.add_argument('--attention', type=str, default='HiPAttention', 
                    choices=['CausalSelfAttention', 'HiPAttention', 'FlashAttention3', 'MultiHeadLatentAttention'],
                    help='Attention mechanism to use')
parser.add_argument('--feedforward', type=str, default='FeedForwardGeGLU',
                    choices=['FeedForwardGeGLU', 'FeedForwardSwiGLU'],
                    help='Feedforward network to use')

# Training parameters
parser.add_argument('--max_iters', type=int, default=2500, help='Maximum training iterations')
parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

# Model size
parser.add_argument('--block_size', type=int, default=128, help='Block size (sequence length)')
parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads')
parser.add_argument('--n_layer', type=int, default=6, help='Number of layers')

# Model presets for easy scaling
parser.add_argument('--model_preset', type=str, default=None, 
                    choices=['tiny', 'small', 'medium', 'large'],
                    help='Use predefined model configurations')

# Attention-specific parameters
parser.add_argument('--mla_latent_dim', type=int, default=None, 
                    help='Latent dimension for MLA (default: n_embd//4)')
parser.add_argument('--hip_chunk_size', type=int, default=32,
                    help='Chunk size for HiPAttention')
parser.add_argument('--hip_top_k_chunks', type=int, default=8,
                    help='Top-k chunks for HiPAttention')

# Analysis options
parser.add_argument('--track_memory', action='store_true',
                    help='Track memory usage during training')
parser.add_argument('--save_curves', action='store_true',
                    help='Save training curves to file')
parser.add_argument('--benchmark_inference', action='store_true',
                    help='Benchmark inference speed after training')

args = parser.parse_args()

# Apply model presets if specified
if args.model_preset == 'tiny':
    args.n_layer, args.n_embd, args.n_head = 4, 256, 4
    args.block_size = 64
elif args.model_preset == 'small':
    args.n_layer, args.n_embd, args.n_head = 6, 384, 6
    args.block_size = 128
elif args.model_preset == 'medium':
    args.n_layer, args.n_embd, args.n_head = 12, 768, 12
    args.block_size = 256
elif args.model_preset == 'large':
    args.n_layer, args.n_embd, args.n_head = 24, 1024, 16
    args.block_size = 512

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'mps'

# Component mapping
NORM_MAP = {
    'LayerNorm': LayerNorm,
    'RMSNorm': RMSNorm
}

ATTENTION_MAP = {
    'CausalSelfAttention': CausalSelfAttention,
    'HiPAttention': HiPAttention,
    'FlashAttention3': FlashAttention3,
    'MultiHeadLatentAttention': MultiHeadLatentAttention
}

FEEDFORWARD_MAP = {
    'FeedForwardGeGLU': FeedForwardGeGLU,
    'FeedForwardSwiGLU': FeedForwardSwiGLU
}

# Get selected components
Norm = NORM_MAP[args.norm]
FeedForward = FEEDFORWARD_MAP[args.feedforward]

# Create attention class with parameters
def create_attention_class(config):
    if args.attention == 'HiPAttention':
        return lambda cfg: HiPAttention(cfg, chunk_size=args.hip_chunk_size, top_k_chunks=args.hip_top_k_chunks)
    elif args.attention == 'MultiHeadLatentAttention':
        latent_dim = args.mla_latent_dim if args.mla_latent_dim else config.n_embd // 4
        return lambda cfg: MultiHeadLatentAttention(cfg, latent_dim=latent_dim)
    else:
        return ATTENTION_MAP[args.attention]

# Load and process data
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Simple character-level tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Split data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Create model with matching config
config = PinkyConfig(
    block_size=args.block_size,
    vocab_size=vocab_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    dropout=args.dropout,
    bias=True
)

# Create a wrapper class for CustomBlock
class MyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        Attention = create_attention_class(config)
        self.block = CustomBlock(Norm, Attention, FeedForward, config)
    
    def forward(self, x):
        return self.block(x)

print(f"Using custom block with: {args.norm} + {args.attention} + {args.feedforward}")
if args.attention == 'MultiHeadLatentAttention':
    latent_dim = args.mla_latent_dim if args.mla_latent_dim else config.n_embd // 4
    print(f"MLA latent dimension: {latent_dim}")
elif args.attention == 'HiPAttention':
    print(f"HiP chunk_size: {args.hip_chunk_size}, top_k_chunks: {args.hip_top_k_chunks}")

model = Pinky(config, Block=MyBlock)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Memory tracking setup
if args.track_memory:
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_usage = []
    
def get_memory_usage():
    if args.track_memory:
        mem_mb = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
            return mem_mb, gpu_mem_mb
        return mem_mb, 0
    return 0, 0

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([data[i:i+args.block_size] for i in ix])
    y = torch.stack([data[i+1:i+args.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# Training curves tracking
if args.save_curves:
    training_log = {
        'iterations': [],
        'train_loss': [],
        'val_loss': [],
        'time_per_iter': [],
        'memory_usage': [],
        'gpu_memory': []
    }

for iter in range(args.max_iters):
    t0 = time.time()

    # every once in a while evaluate the loss on train and val sets
    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss()
        dt = time.time() - t0
        mem_cpu, mem_gpu = get_memory_usage()
        
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {dt/60:.2f}min")
        if args.track_memory:
            print(f"  Memory - CPU: {mem_cpu:.1f}MB, GPU: {mem_gpu:.1f}MB")
        
        # Save to training log
        if args.save_curves:
            training_log['iterations'].append(iter)
            training_log['train_loss'].append(losses['train'].item())
            training_log['val_loss'].append(losses['val'].item())
            training_log['time_per_iter'].append(dt / max(1, iter - (0 if iter == 0 else training_log['iterations'][-2] if len(training_log['iterations']) > 1 else 0)))
            training_log['memory_usage'].append(mem_cpu)
            training_log['gpu_memory'].append(mem_gpu)

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save training curves if requested
if args.save_curves:
    import json
    config_str = f"{args.norm}_{args.attention}_{args.feedforward}"
    if args.model_preset:
        config_str = f"{args.model_preset}_{config_str}"
    
    log_filename = f"training_log_{config_str}.json"
    with open(log_filename, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training curves saved to {log_filename}")

# Benchmark inference speed if requested
if args.benchmark_inference:
    print("\n=== Inference Benchmarking ===")
    model.eval()
    
    sequence_lengths = [64, 128, 256, 512]
    if args.block_size < max(sequence_lengths):
        sequence_lengths = [s for s in sequence_lengths if s <= args.block_size]
    
    for seq_len in sequence_lengths:
        if seq_len > args.block_size:
            continue
            
        # Create test input
        test_input = torch.randint(0, vocab_size, (1, seq_len), device=device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(test_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        num_runs = 50
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(test_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        tokens_per_sec = seq_len / avg_time
        
        print(f"Seq len {seq_len:3d}: {avg_time*1000:.2f}ms/forward, {tokens_per_sec:.0f} tokens/sec")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))