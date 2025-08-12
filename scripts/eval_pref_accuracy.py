import argparse, os, json
from genpref_sega.models.hf import load_causal_lm, sequence_logprobs
def logprob_of_completion(hf, prompt: str, completion: str) -> float:
    tokenizer = hf.tokenizer; model = hf.model
    full = prompt + "\n\n" + completion
    enc = tokenizer(full, return_tensors="pt").to(model.device)
    out = model(**enc); logits = out.logits
    gen_ids = tokenizer(completion, return_tensors="pt").input_ids.to(model.device)[0]
    gen_len = gen_ids.shape[0]
    row_logits = logits[0, -gen_len:, :]; row_labels = gen_ids
    logp = sequence_logprobs(row_logits.unsqueeze(0), row_labels.unsqueeze(0)).item()
    return float(logp)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()
    hf = load_causal_lm(args.model_name_or_path)
    items = [json.loads(l) for l in open(args.eval_jsonl, "r", encoding="utf-8")]
    correct = 0
    for r in items:
        lp1 = logprob_of_completion(hf, r["prompt"], r["chosen"])
        lp2 = logprob_of_completion(hf, r["prompt"], r["rejected"])
        r1 = args.beta * lp1; r2 = args.beta * lp2
        if r1 >= r2: correct += 1
    acc = correct / max(1, len(items))
    print(json.dumps({"accuracy": acc, "n": len(items)}, indent=2))
if __name__ == "__main__": main()
