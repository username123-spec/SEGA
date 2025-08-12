from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
@dataclass
class HFModel:
    model: AutoModelForCausalLM
    tokenizer: Any
def load_causal_lm(model_name_or_path: str, device: Optional[str] = None) -> HFModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if device and not torch.cuda.is_available():
        model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    return HFModel(model=model, tokenizer=tokenizer)
def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    labels_exp = labels.unsqueeze(-1)
    selected = torch.gather(logp, dim=-1, index=labels_exp).squeeze(-1)
    mask = (labels != ignore_index).float()
    seq_logprob = (selected * mask).sum(dim=-1)
    return seq_logprob
