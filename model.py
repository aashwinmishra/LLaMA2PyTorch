import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass 
class ModelArgs:
  """
  Hyperparameter definitions for the class.
  """
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: Optional[int] = None
  vocab_size: int = -1 #placeholder
  multiple_of: int = 256
  ffn_dim_multiplier: Optional[float] = None 
  norm_eps: float = 1e-5
  max_batch_size: int = 32
  max_seq_length: int = 2048
  device: str = None


def precompute_theta_pos_frequencies(head_dim: int, 
                                     seq_len: int, 
                                     device: str, 
                                     theta: float=10000.0)->torch.tensor:
  theta = torch.tensor([10000**(-2*(i-1)/head_dim) for i in range(head_dim//2)]).to(device)
  m = torch.arange(seq_len, device=device)
  freqs = torch.outer(m, theta).float()
  return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_embeddings(x: torch.Tensor, 
                            freqs_complex: torch.Tensor, 
                            device: str)->torch.tensor:
  x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
  freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
  x_rotated = x_complex * freqs_complex
  x_out = torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x).to(device)
  return x_out


class RMSNorm(nn.Module):
  def __init__(self, emb_dim: int, eps: float=1e-6):
    super().__init__()
    self.eps = eps 
    self.weight = nn.Parameter(torch.ones(emb_dim))

  def forward(self, x):
    return self.weight * x / (torch.square(x).mean(dim=-1, keepdim=True).sqrt() + self.eps)


class SelfAttention(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    self.n_heads_q = args.n_heads 
    self.n_rep = self.n_heads_q // self.n_kv_heads
    self.head_dim = args.dim // args.n_heads
    self.wq = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
    self.wv = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)
    self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_length, self.n_kv_heads, self.head_dim)
    self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_length, self.n_kv_heads, self.head_dim)

  def forward(self, x, start_pos, freqs_complex):
    batch_size, seq_len, _ = x.shape 

    q = self.wq(x).view(batch_size, seq_len, self.n_heads_q, self.head_dim)     #[batch_size, seq_len, self.n_heads, self.head_dim]
    k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)    #[batch_size, seq_len, self.n_kv_heads, self.head_dim]
    v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)    #[batch_size, seq_len, self.n_kv_heads, self.head_dim]

    q = apply_rotary_embeddings(q, freqs_complex, x.device)
    k = apply_rotary_embeddings(k, freqs_complex, x.device)

    self.cache_k[:batch_size, start_pos:start_pos + seq_len] = k 
    self.cache_v[:batch_size, start_pos:start_pos + seq_len] = v 

    k = self.cache_k[:batch_size, :start_pos + seq_len]                         #[batch_size, seq_len+pos, n_kv_heads, head_dim]
    v = self.cache_v[:batch_size, :start_pos + seq_len]                         #[batch_size, seq_len+pos, n_kv_heads, head_dim]

    k = k.repeat_interleave(self.n_rep, dim=2)                                  #[batch_size, seq_len+pos, n_heads, head_dim]
    v = v.repeat_interleave(self.n_rep, dim=2)                                  #[batch_size, seq_len+pos, n_heads, head_dim]

    q = q.transpose(1, 2)                                                       #[batch_size, n_heads, seq_len, head_dim]
    k = k.permute(0, 2, 3, 1)                                                   #[batch_size, n_heads, head_dim, seq_len+pos]
    v = v.transpose(1, 2)                                                       #[batch_size, n_heads, seq_len+pos, head_dim]

    scores = torch.matmul(q, k) / math.sqrt(self.head_dim)                      #[batch_size, n_heads, seq_len, seq_len+pos]
    attn = F.softmax(scores, dim=-1)                                            #[batch_size, n_heads, seq_len, seq_len+pos]

    out = torch.matmul(attn, v)                                                 #[batch_size, n_heads, seq_len, head_dim]
    out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)        #[batch_size, seq_len, dim]
    out = self.wo(out)                                                          #[batch_size, seq_len, dim]
    return out


class FeedForward(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    hidden_dim = int(args.dim * 8/3) 
    if args.ffn_dim_multiplier is not None:
      hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
    hidden_dim = math.ceil(hidden_dim / args.multiple_of) * args.multiple_of

    self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
    self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

  def forward(self, x):
    swish = F.silu(self.w1(x))
    x_V = self.w3(x)
    x = swish * x_V 
    return self.w2(x)


class EncoderBlock(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.n_heads = args.n_heads
    self.n_kv_heads = args.n_kv_heads
    self.dim = args.dim 
    self.head_dim = args.dim // args.n_heads
    self.attention = SelfAttention(args)
    self.feed_forward = FeedForward(args)
    self.attention_norm = RMSNorm(emb_dim=args.dim, eps=args.norm_eps)
    self.ffn_norm = RMSNorm(emb_dim=args.dim, eps=args.norm_eps)

  def forward(self, x, start_pos, freqs_complex):
    h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out 


class Transformer(nn.Module):
  def __init__(self, args: ModelArgs) -> None:
    super().__init__()
    self.args = args
    self.vocab_size = args.vocab_size
    self.n_layers = args.n_layers
    self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
    self.layers = nn.ModuleList()
    for _ in range(args.n_layers):
      self.layers.append(EncoderBlock(args))
    self.norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
    self.freqs_complex = precompute_theta_pos_frequencies(args.dim // args.n_heads, args.max_seq_length * 2, device=args.device)

  def forward(self, tokens: torch.Tensor, start_pos: int):
    #Input Shape:                         [batch_size, sequence_length]
    batch_size, seq_len = tokens.shape
    h = self.tok_embeddings(tokens)       #[batch_size, sequence_length, d_model]
    freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]
    for layer in self.layers:
      h = layer(h, start_pos, freqs_complex)
    h = self.norm(h)
    return self.output(h).float()

