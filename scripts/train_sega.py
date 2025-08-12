import argparse, os
from genpref_sega.models.hf import load_causal_lm
from genpref_sega.data.jsonl import load_pairs
from genpref_sega.sega.trainer import SEGATrainer, SEGAConfig
from genpref_sega.utils.common import set_seed
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lambda_fork", type=float, default=0.4)
    ap.add_argument("--top_m", type=float, default=0.1)
    ap.add_argument("--final_window", type=int, default=3)
    ap.add_argument("--reward_mapping", type=str, choices=["identity","softmax"], default="softmax")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gen_max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); set_seed(args.seed)
    hf = load_causal_lm(args.model_name_or_path)
    data = load_pairs(args.train_jsonl); prompts = [r["prompt"] for r in data]
    cfg = SEGAConfig(k=args.k, lambda_fork=args.lambda_fork, top_m=args.top_m, final_window=args.final_window,
                     reward_mapping=args.reward_mapping, beta=args.beta, gen_max_new_tokens=args.gen_max_new_tokens,
                     temperature=args.temperature, top_p=args.top_p, lr=args.lr)
    trainer = SEGATrainer(hf, cfg)
    for ep in range(args.epochs):
        metrics = trainer.train_epoch(prompts, batch_size=args.batch_size)
        print({"epoch": ep+1, **metrics})
        hf.model.save_pretrained(args.out_dir); hf.tokenizer.save_pretrained(args.out_dir)
if __name__ == "__main__": main()
