from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import torch, random, os, numpy as np
DEFAULT_THINK_PROMPT = "Let's think step by step."
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def build_prompt(prompt: str) -> str:
    return f"{prompt}\n\n{DEFAULT_THINK_PROMPT}\n"
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
