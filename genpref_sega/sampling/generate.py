from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import torch
from transformers import GenerationConfig
from ..utils.common import build_prompt
@dataclass
class SampledCOT:
    text: str
    tokens: List[int]
    scores: List[torch.Tensor]
def sample_cots(hf, prompt: str, k: int = 4, max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 0.95) -> List[SampledCOT]:
    tokenizer = hf.tokenizer; model = hf.model; device = next(model.parameters()).device
    input_text = build_prompt(prompt)
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    out = model.generate(
        **inputs, do_sample=True, temperature=temperature, top_p=top_p,
        num_return_sequences=k, max_new_tokens=max_new_tokens,
        output_scores=True, return_dict_in_generate=True
    )
    seqs = out.sequences
    scores = out.scores
    results: List[SampledCOT] = []
    prompt_len = inputs["input_ids"].shape[1]
    for i in range(k):
        gen_tokens = seqs[i, prompt_len:].tolist()
        per_step_logits = [scores_t[i] for scores_t in scores]
        gen_text = tokenizer.decode(seqs[i, prompt_len:], skip_special_tokens=True)
        results.append(SampledCOT(text=gen_text, tokens=gen_tokens, scores=per_step_logits))
    return results
