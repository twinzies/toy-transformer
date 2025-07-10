import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """Standard Layer Normalization implementation."""
    def __init__(self, size: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None
        self.eps = eps
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
    
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    RMS normalization is a simpler alternative to LayerNorm that only normalizes
    the variance without centering the mean.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

class CausalSelfAttention(nn.Module):
    """Causal Self-Attention Mechanism.
    
    Standard multi-head self-attention with causal masking for autoregressive modeling.
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        ))
        self.n_head = config.n_head
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    
        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class HiPAttention(nn.Module):
    """
    Hierarchically Pruned (HiP) Attention.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "d_model must be divisible by num_heads"

        self.d_model = config.n_embd
        self.num_heads = config.n_head
        self.d_head = self.d_model // self.num_heads
        self.chunk_size = 32
        self.top_k_chunks = 8

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, mask=None):
        query = x
        key = x
        value = x
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape

        # Project and split heads
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_kv, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_kv, self.num_heads, self.d_head).transpose(1, 2)

        # The core of HiP Attention is the hierarchical pruning.
        # This is done for each query in the sequence.
        
        outputs = []
        for i in range(seq_len_q):
            q_i = q[:, :, i, :].unsqueeze(2) # (batch_size, num_heads, 1, d_head)
            
            # Get the sparse indices for the current query
            sparse_indices = self.hierarchical_pruning(q_i, k)
            
            # Gather the selected keys and values
            k_sparse = torch.gather(k, 2, sparse_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_head))
            v_sparse = torch.gather(v, 2, sparse_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_head))

            # Standard scaled dot-product attention on the sparse keys/values
            scores = torch.matmul(q_i, k_sparse.transpose(-2, -1)) / math.sqrt(self.d_head)
            
            # Apply mask if provided
            if mask is not None:
                # The mask needs to be adapted for the sparse keys
                sparse_mask = torch.gather(mask, 1, sparse_indices.squeeze(0).squeeze(0))
                scores = scores.masked_fill(sparse_mask == 0, -1e9)

            attn = F.softmax(scores, dim=-1)
            context = torch.matmul(attn, v_sparse)
            outputs.append(context)

        # Concatenate the outputs for all queries
        output = torch.cat(outputs, dim=2)
        
        # Reshape and project back to d_model
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        return self.out_proj(output)

    def hierarchical_pruning(self, q_i, k):
        """
        Performs hierarchical pruning to find the most relevant keys for a given query.

        Args:
            q_i (torch.Tensor): The current query tensor. Shape: (batch_size, num_heads, 1, d_head)
            k (torch.Tensor): The key tensor. Shape: (batch_size, num_heads, seq_len_kv, d_head)

        Returns:
            torch.Tensor: The indices of the selected keys.
        """
        batch_size, num_heads, seq_len_kv, d_head = k.shape
        
        # Start with all indices
        current_indices = torch.arange(seq_len_kv).view(1, 1, -1).expand(batch_size, num_heads, -1).to(k.device)
        
        current_chunk_size = self.chunk_size
        
        while current_chunk_size > 1:
            num_chunks = math.ceil(current_indices.size(2) / current_chunk_size)
            
            # Pad indices to be divisible by chunk size
            padding_size = num_chunks * current_chunk_size - current_indices.size(2)
            padded_indices = F.pad(current_indices, (0, padding_size), value=-1)
            
            # Reshape into chunks
            chunks = padded_indices.view(batch_size, num_heads, num_chunks, current_chunk_size)
            
            # Select representative index (center of each chunk)
            rep_indices = chunks[:, :, :, current_chunk_size // 2]
            
            # Mask out padding
            rep_indices = rep_indices.masked_fill(rep_indices == -1, 0)
            
            # Gather representative keys
            rep_keys = torch.gather(k, 2, rep_indices.unsqueeze(-1).expand(-1, -1, -1, d_head))
            
            # Calculate importance scores
            scores = torch.matmul(q_i, rep_keys.transpose(-2, -1)).squeeze(2)
            
            # Select top-k chunks
            _, top_k_chunk_indices = torch.topk(scores, min(self.top_k_chunks, num_chunks), dim=-1)
            
            # Gather the indices from the top-k chunks
            selected_chunks = torch.gather(chunks, 2, top_k_chunk_indices.unsqueeze(-1).expand(-1, -1, -1, current_chunk_size))
            
            # Flatten and remove padding
            current_indices = selected_chunks.reshape(batch_size, num_heads, -1)
            # Remove padding by masking
            mask = current_indices != -1
            valid_counts = mask.sum(dim=-1, keepdim=True)
            max_valid = valid_counts.max().item()
            current_indices = current_indices[:, :, :max_valid]
            current_indices = current_indices.masked_fill(~mask[:, :, :max_valid], 0)
            
            # Reduce chunk size for the next level of pruning
            current_chunk_size = max(1, current_chunk_size // 2)

        return current_indices
    
class FlashAttention3(nn.Module):
    """
    A from-scratch implementation that simulates the logic of FlashAttention.

    Disclaimer: This implementation is for educational purposes to demonstrate the
    tiling and online softmax algorithm of FlashAttention. It is written in pure
    PyTorch and will NOT achieve the performance of the official CUDA-level

    implementation. For performance, always use the official `flash-attn` package.
    """
    def __init__(self, config, block_size=64, causal=True):
        """
        Initializes the FlashAttention3 module.

        Args:
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of attention heads.
            block_size (int): The block size for tiling. A key hyperparameter.
            causal (bool): Whether to apply causal masking for auto-regressive tasks.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "d_model must be divisible by num_heads"

        self.d_model = config.n_embd
        self.num_heads = config.n_head
        self.d_head = self.d_model // self.num_heads
        self.block_size = block_size
        self.causal = causal
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        """
        Forward pass for FlashAttention.

        Args:
            query (torch.Tensor): The query tensor. Shape: (batch_size, seq_len_q, d_model)
            key (torch.Tensor): The key tensor. Shape: (batch_size, seq_len_kv, d_model)
            value (torch.Tensor): The value tensor. Shape: (batch_size, seq_len_kv, d_model)

        Returns:
            torch.Tensor: The output of the attention mechanism.
        """
        query = x
        key = x
        value = x
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape

        # Project and split heads
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_kv, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_kv, self.num_heads, self.d_head).transpose(1, 2)

        # Initialize the output tensor and the online softmax statistics
        # O is the output, l is the running sum of exponentials for softmax
        # m is the running maximum for stable softmax
        O = torch.zeros_like(q)
        l = torch.zeros(batch_size, self.num_heads, seq_len_q, 1, device=q.device)
        m = torch.full((batch_size, self.num_heads, seq_len_q, 1), -float('inf'), device=q.device)

        # --- Tiling Loops ---
        # The core of the FlashAttention algorithm.
        # We iterate through the key/value tensors in blocks.
        num_blocks_kv = math.ceil(seq_len_kv / self.block_size)
        for j in range(num_blocks_kv):
            # Get the j-th block of K and V
            start_j = j * self.block_size
            end_j = min(start_j + self.block_size, seq_len_kv)
            k_j = k[:, :, start_j:end_j, :]
            v_j = v[:, :, start_j:end_j, :]

            # We then iterate through the query tensor in blocks.
            num_blocks_q = math.ceil(seq_len_q / self.block_size)
            for i in range(num_blocks_q):
                # Get the i-th block of Q, O, l, and m
                start_i = i * self.block_size
                end_i = min(start_i + self.block_size, seq_len_q)
                q_i = q[:, :, start_i:end_i, :]
                O_i = O[:, :, start_i:end_i, :]
                l_i = l[:, :, start_i:end_i, :]
                m_i = m[:, :, start_i:end_i, :]

                # --- Core Attention Calculation for the Block ---
                # This is where the "online softmax" update happens.

                # 1. Calculate scores for the current block
                S_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) / self.scale

                # Apply causal mask if needed
                if self.causal:
                    # Create a mask for the current block
                    mask_offset = start_j - start_i
                    causal_mask = torch.triu(torch.ones(end_i - start_i, end_j - start_j, device=q.device), diagonal=mask_offset + 1).bool()
                    S_ij.masked_fill_(causal_mask, -float('inf'))

                # 2. Get the new max for the stable softmax
                m_ij_new = torch.max(S_ij, dim=-1, keepdim=True)[0]
                m_i_new = torch.maximum(m_i, m_ij_new)

                # 3. Calculate the new probabilities (P_ij) with the updated max
                P_ij = torch.exp(S_ij - m_i_new)

                # 4. Update the running sum of exponentials (l_i)
                # We rescale the old sum by the change in the max value
                l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.sum(P_ij, dim=-1, keepdim=True)

                # 5. Update the output (O_i)
                # We rescale the old output by the change in the max value
                O_i_rescaled = O_i * (l_i / l_i_new) * torch.exp(m_i - m_i_new)
                O_i_new = O_i_rescaled + torch.matmul(P_ij, v_j)

                # --- Update the global tensors with the new block values ---
                O[:, :, start_i:end_i, :] = O_i_new
                l[:, :, start_i:end_i, :] = l_i_new
                m[:, :, start_i:end_i, :] = m_i_new

        # Reshape and project back to d_model
        output = O.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        return self.out_proj(output)

class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit.
    
    The formula is: SwiGLU(x, W, V) = Swish(xW) * xV
    Swish(x) is x * sigmoid(x), which is equivalent to F.silu(x) in PyTorch.
    """
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        """
        Initializes the SwiGLU module.

        Args:
            dim_in (int): Input dimension.
            dim_out (int): Output dimension.
            bias (bool): Whether to use a bias in the linear projection.
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim_in).

        Returns:
            torch.Tensor: Output tensor of shape (..., dim_out).
        """
        # Project the input
        x_proj = self.proj(x)
        
        # Split the projected tensor into two halves along the last dimension
        x1, x2 = x_proj.chunk(2, dim=-1)
        
        return F.silu(x1) * x2

class GeGLU(nn.Module):
    """
    GELU Gated Linear Unit.
    
    Another variant from "GLU Variants Improve Transformer".
    
    The formula is: GeGLU(x, W, V) = GELU(xW) * xV
    """
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        """
        Initializes the GeGLU module.

        Args:
            dim_in (int): Input dimension.
            dim_out (int): Output dimension.
            bias (bool): Whether to use a bias in the linear projection.
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Single linear layer for combined projections
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GeGLU.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim_in).

        Returns:
            torch.Tensor: Output tensor of shape (..., dim_out).
        """
        # Project the input
        x_proj = self.proj(x)
        
        # Split into two halves
        x1, x2 = x_proj.chunk(2, dim=-1)
        
        # Apply GELU activation to the first half and multiply by the gate
        return F.gelu(x1) * x2

class FeedForwardSwiGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.swiglu    = SwiGLU(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class FeedForwardGeGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.geglue    = GeGLU(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.geglue(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x