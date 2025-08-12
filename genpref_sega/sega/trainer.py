from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Literal
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from ..models.hf import sequence_logprobs
from ..scoring.entropy import EntropyScoreConfig, sequence_entropies, entropy_guided_score
from ..sampling.generate import sample_cots
RewardMapping = Literal["identity", "softmax"]
@dataclass
class SEGAConfig:
    k: int = 4
    lambda_fork: float = 0.4
    top_m: float = 0.1
    final_window: int = 3
    reward_mapping: RewardMapping = "softmax"
    beta: float = 1.0
    gen_max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    lr: float = 5e-6
    grad_clip: float = 1.0
def map_scores_to_rewards(scores: torch.Tensor, mapping: RewardMapping = "softmax") -> torch.Tensor:
    if mapping == "identity": return scores
    return torch.nn.functional.softmax(scores, dim=-1)
def sega_step(hf, batch_prompts: List[str], cfg: SEGAConfig) -> Tuple[torch.Tensor, Dict[str, Any]]:
    device = next(hf.model.parameters()).device
    tokenizer = hf.tokenizer; model = hf.model
    all_loss = torch.tensor(0.0, device=device); debug_scores = []
    for prompt in batch_prompts:
        samples = sample_cots(hf, prompt, k=cfg.k, max_new_tokens=cfg.gen_max_new_tokens, temperature=cfg.temperature, top_p=cfg.top_p)
        S_list = []
        for s in samples:
            ents = sequence_entropies(s.scores)
            S = entropy_guided_score(ents, EntropyScoreConfig(lambda_fork=cfg.lambda_fork, top_m=cfg.top_m, final_window=cfg.final_window))
            S_list.append(S)
        S_t = torch.tensor(S_list, device=device, dtype=torch.float32)
        r_t = map_scores_to_rewards(S_t, mapping=cfg.reward_mapping)
        r_bar = r_t.mean(); A_t = r_t - r_bar
        texts = [s.text for s in samples]
        full_inputs = [prompt + "\n\n" + t for t in texts]
        enc = tokenizer(full_inputs, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True).to(device)
        outputs = model(**enc, labels=None)
        logits = outputs.logits
        logp_list = []
        for i, s in enumerate(samples):
            gen_ids = tokenizer(s.text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)[0]
            gen_len = gen_ids.shape[0]
            row_logits = logits[i, -gen_len:, :]
            row_labels = gen_ids
            row_logprob = sequence_logprobs(row_logits.unsqueeze(0), row_labels.unsqueeze(0)).squeeze(0)
            logp_list.append(row_logprob)
        logp_t = torch.stack(logp_list, dim=0)
        loss_q = -(A_t.detach() * logp_t).sum() / cfg.k
        all_loss = all_loss + loss_q
        debug_scores.append({"S": [float(x) for x in S_list], "r": [float(x) for x in r_t.tolist()], "A": [float(x) for x in A_t.tolist()], "avg_logp": float(logp_t.mean().item())})
    all_loss = all_loss / max(1, len(batch_prompts))
    return all_loss, {"groups": debug_scores}
class SEGATrainer:
    def __init__(self, hf, sega_cfg: SEGAConfig):
        self.hf = hf; self.cfg = sega_cfg; self.opt = AdamW(self.hf.model.parameters(), lr=sega_cfg.lr)
    def train_epoch(self, prompts: List[str], batch_size: int = 1) -> Dict[str, Any]:
        model = self.hf.model; model.train()
        total_loss = 0.0
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            loss, metrics = sega_step(self.hf, batch, self.cfg)
            self.opt.zero_grad(set_to_none=True); loss.backward()
            clip_grad_norm_(model.parameters(), self.cfg.grad_clip); self.opt.step()
            total_loss += float(loss.item())
        return {"loss": total_loss / max(1, (len(prompts)+batch_size-1)//batch_size)}
