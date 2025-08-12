from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch, math
@dataclass
class EntropyScoreConfig:
    lambda_fork: float = 0.4
    top_m: float = 0.1
    final_window: int = 3
def token_entropy(logits: torch.Tensor) -> float:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    ent = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
    return float(ent)
def sequence_entropies(per_step_logits: List[torch.Tensor]) -> List[float]:
    return [token_entropy(log) for log in per_step_logits]
def entropy_guided_score(entropies: List[float], cfg: EntropyScoreConfig) -> float:
    n = len(entropies)
    if n == 0: return -1e9
    m_final = min(cfg.final_window, n)
    H_final = sum(entropies[-m_final:]) / m_final
    rest = entropies[:-m_final] if n > m_final else []
    if not rest:
        avg_top = 0.0
    else:
        if cfg.top_m < 1.0: k = max(1, int(math.ceil(len(rest) * cfg.top_m)))
        else: k = int(cfg.top_m)
        k = max(1, min(k, len(rest)))
        top_vals = sorted(rest, reverse=True)[:k]
        avg_top = sum(top_vals) / len(top_vals)
    return -H_final + cfg.lambda_fork * avg_top
