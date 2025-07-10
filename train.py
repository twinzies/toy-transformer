from models import Pinky, PinkyConfig, CustomBlock
from modules import FeedForwardGeGLU, FeedForwardSwiGLU, LayerNorm, RMSNorm, CausalSelfAttention, HiPAttention, FlashAttention3
import torch
import torch.nn as nn
import time

device = 'cuda' if torch.cuda.is_available() else 'mps'
max_iters = 2500
eval_interval = 500
learning_rate = 3e-4
batch_size = 64
block_size = 128
n_embd = 384
n_head = 6
n_layer = 6

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
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=0.2,
    bias=True
)

# Create a wrapper class for CustomBlock
class MyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = CustomBlock(RMSNorm, HiPAttention, FeedForwardGeGLU, config)
    
    def forward(self, x):
        return self.block(x)

print(f"Using custom block with: RMSNorm + HiPAttention + FeedForwardGeGLU")

model = Pinky(config, Block=MyBlock)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    t0 = time.time()

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Print time per step
    dt = time.time() - t0
    print(f"step {iter}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))