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

