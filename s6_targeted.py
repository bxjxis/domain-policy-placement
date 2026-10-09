"""Targeted check on the 25 review samples (all blocked as S6 by the default guard earlier).

For each sample, run Llama Guard 3 with the default arm and the policy arm:
  input mode  (prompt only)               -> compared with annotator A's input decision
  output mode (prompt + stored response)  -> secondary; responses are old Qwen outputs, truncated
Reuses the prompt builder and parsers from placement_harness.py.
Usage (cluster): python s6_targeted.py
"""
import csv, json, sys
from pathlib import Path
from placement_harness import (GUARD, GUARD_REV, build_guard_prompt, load_policy,
                               parse_verdict, verdict_step, enc)

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "medical_policy_review_25.csv", encoding="utf-8-sig", newline="")))
ptxt, sha = load_policy()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(GUARD, revision=GUARD_REV)
model = AutoModelForCausalLM.from_pretrained(GUARD, revision=GUARD_REV, dtype=torch.bfloat16, device_map="cuda").eval()
iu, is_ = tok.encode("unsafe", add_special_tokens=False)[0], tok.encode("safe", add_special_tokens=False)[0]

def classify(text):
    inp = enc(tok, text, "cuda"); plen = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id,
                             output_scores=True, return_dict_in_generate=True)
    gen = out.sequences[0][plen:].tolist()
    raw = tok.decode(gen, skip_special_tokens=True)
    v, cats = parse_verdict(raw)
    k = verdict_step([tok.decode([i]) for i in gen])
    p = None
    if k is not None:
        pr = torch.softmax(out.scores[k][0].float(), -1)
        pu, ps = pr[iu].item(), pr[is_].item()
        p = pu / (pu + ps) if pu + ps > 0 else None
    return dict(verdict=v, cats=cats, raw=raw, p=p)

res = []
for r in rows:
    d = dict(idx=int(r["idx"]), A_input=r["A_input_decision"].strip())
    for arm in ("default", "policy"):
        d[f"in_{arm}"] = classify(build_guard_prompt(r["prompt"], None, arm, ptxt))
        d[f"out_{arm}"] = classify(build_guard_prompt(r["prompt"], r["response"], arm, ptxt))
    res.append(d)

out = HERE / "s6_targeted_results.json"
json.dump(dict(policy_sha256=sha, guard=GUARD, guard_revision=GUARD_REV, results=res),
          open(out, "w"), ensure_ascii=False, indent=1)

fmt = lambda g: f"{g['verdict'][:6]:6s}{','.join(g['cats']):5s} p={g['p']:.2f}" if g["p"] is not None else g["verdict"]
print(f"policy sha {sha[:12]}\nidx  A      | IN default            IN policy             | OUT default           OUT policy")
for d in res:
    print(f"{d['idx']:3d}  {d['A_input']:6s} | {fmt(d['in_default']):21s} {fmt(d['in_policy']):21s} | "
          f"{fmt(d['out_default']):21s} {fmt(d['out_policy']):21s}")
for arm in ("default", "policy"):
    pred = {d["idx"]: ("block" if d[f"in_{arm}"]["verdict"] != "safe" else "allow") for d in res}
    agree = sum(pred[d["idx"]] == d["A_input"] for d in res)
    fb = sum(pred[d["idx"]] == "block" and d["A_input"] == "allow" for d in res)
    fa = sum(pred[d["idx"]] == "allow" and d["A_input"] == "block" for d in res)
    print(f"INPUT {arm:7s}: agree with A {agree}/25 | blocks A-allowed {fb} | allows A-blocked {fa}")
print("saved", out)
