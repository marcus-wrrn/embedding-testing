import torch
import torch.nn as nn
from torch.nn import functional as F
from base_models import MLP
class AttentionHead(nn.Module):
    """
    Represents a single head of the self-attention mechanism.
    
    Args:
    - head_size (int): Dimension of the key, query, and value vectors.
    - n_embd (int): Input embedding dimension, expected size of input tensor's last dimension.
    - dropout (float): Dropout probability for the attention weights.
    - block_size (int, optional): Size of the attention block (sequence length). Required if forward_only is True.
    - forward_only (bool, default=False): If True, the attention mechanism will only attend to 
        previous or same positions, simulating a decoder's behavior in autoregressive tasks.
    
    Usage:
    ```python
    attention = AttentionHead(head_size=64, n_embd=512, dropout=0.1, block_size=128, forward_only=True)
    ```
    """

    def __init__(self, head_size, n_embd, dropout=0.1, block_size=None, forward_only=False):
  
        super().__init__()
        # No bias for the layers (only focusing on the values)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # If model is a decoder, then apply tril mask, since it is forward only
        if forward_only:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout=0.1, block_size=None, forward_only=False):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, n_embd, dropout, block_size, forward_only) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = MLP(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x