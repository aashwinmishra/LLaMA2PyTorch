import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
from sentencepiece import SentencePieceProcessor
from pathlib import Path
from tqdm import tqdm 
from typing import List, Dict, Tuple, Optional
from model import ModelArgs, Transformer


class LLaMA:
  def __init__(self, 
               model: Transformer, 
               tokenizer: SentencePieceProcessor, 
               model_args: ModelArgs):
    self.model = model 
    self.tokenizer = tokenizer 
    self.args = model_args 

  @staticmethod
  def build(checkpoints_dir: str, 
            tokenizer_path: str, 
            load_model: bool, 
            max_seq_len: int, 
            max_batch_size: int, 
            device: str):
    if load_model:
      checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
      assert len(checkpoints) > 0 
      chk_path = checkpoints[0]
      print(f"Loading Checkpoint {chk_path}")
      checkpoint = torch.load(chk_path, map_location="cpu")

    with open(Path(checkpoints_dir) / "params.json", "r") as f:
      params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_length=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        **params
    )

    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    model_args.vocab_size = tokenizer.vocab_size()

    if device == 'cuda':
      torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
      torch.set_default_tensor_type(torch.BFloat16Tensor)

    model = Transformer(model_args).to(device)

    if load_model:
      del checkpoint["rope.freqs"]
      model.load_state_dict(checkpoint, strict=True)

    return LLaMA(model, tokenizer, model_args)
                                    
