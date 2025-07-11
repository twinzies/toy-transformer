from models import Pinky, PinkyConfig, CustomBlock
from modules import FeedForwardGeGLU, FeedForwardSwiGLU, LayerNorm, RMSNorm, CausalSelfAttention, HiPAttention, FlashAttention3
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
                    choices=['CausalSelfAttention', 'HiPAttention', 'FlashAttention3'],
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

args = parser.parse_args()

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
    'FlashAttention3': FlashAttention3
}

FEEDFORWARD_MAP = {
    'FeedForwardGeGLU': FeedForwardGeGLU,
    'FeedForwardSwiGLU': FeedForwardSwiGLU
}

# Get selected components
Norm = NORM_MAP[args.norm]
Attention = ATTENTION_MAP[args.attention]
FeedForward = FEEDFORWARD_MAP[args.feedforward]

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
        self.block = CustomBlock(Norm, Attention, FeedForward, config)
    
    def forward(self, x):
        return self.block(x)

print(f"Using custom block with: {args.norm} + {args.attention} + {args.feedforward}")

model = Pinky(config, Block=MyBlock)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

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

for iter in range(args.max_iters):
    t0 = time.time()

    # every once in a while evaluate the loss on train and val sets
    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss()
        dt = time.time() - t0
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {dt/60:.2f}min")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))