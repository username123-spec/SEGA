import argparse, os
from genpref_sega.models.hf import load_causal_lm
from genpref_sega.data.jsonl import load_pairs
from genpref_sega.utils.common import set_seed
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
class PairDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]
def collate_fn(batch, tokenizer):
    texts = [b["prompt"] + "\n\n" + b["chosen"] for b in batch]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    labels = enc["input_ids"].clone()
    return {**enc, "labels": labels}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); set_seed(args.seed)
    hf = load_causal_lm(args.model_name_or_path); model, tokenizer = hf.model, hf.tokenizer
    data = load_pairs(args.train_jsonl); ds = PairDataset(data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    optim = AdamW(model.parameters(), lr=args.lr); model.train()
    for ep in range(args.epochs):
        pbar = tqdm(dl, desc=f"SFT epoch {ep+1}")
        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch); loss = out.loss
            optim.zero_grad(set_to_none=True); loss.backward(); clip_grad_norm_(model.parameters(), 1.0); optim.step()
            pbar.set_postfix(loss=float(loss.item()))
        model.save_pretrained(args.out_dir); tokenizer.save_pretrained(args.out_dir)
if __name__ == "__main__": main()
