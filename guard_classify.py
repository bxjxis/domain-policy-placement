"""Llama Guard 3 as input-guard vs output-guard over the cached XSTest-safe run.

Reuses ~/orr_spike/orr_results.json (250 benign prompts + target responses),
so no regeneration. For each item, classifies:
  - INPUT mode:  user prompt only           -> input-guard false-block
  - OUTPUT mode: user prompt + assistant resp -> output-guard false-block
All prompts are benign (XSTest safe), so any 'unsafe' is an over-refusal.
Records verdict + hazard category codes (S1..S13); S6 = Specialized Advice.
"""
import json, re, time
from collections import defaultdict, Counter
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GUARD = "meta-llama/Llama-Guard-3-8B"
OUT = Path.home() / "orr_spike"
items = json.load(open(OUT / "orr_results.json", encoding="utf-8"))
print(f"loaded {len(items)} cached items", flush=True)

tok = AutoTokenizer.from_pretrained(GUARD)
try:
    model = AutoModelForCausalLM.from_pretrained(GUARD, dtype=torch.bfloat16, device_map="cuda")
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(GUARD, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

def classify(chat):
    text = tok.apply_chat_template(chat, tokenize=False)
    inp = tok(text, return_tensors="pt").to("cuda")
    plen = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][plen:], skip_special_tokens=True).strip()
    low = gen.lower()
    verdict = "unsafe" if low.startswith("unsafe") else ("safe" if low.startswith("safe") else "?")
    cats = re.findall(r"S\d+", gen)
    return verdict, cats, gen

rows = []
t0 = time.time()
for i, it in enumerate(items):
    p, r = it["prompt"], it["response"]
    iv, ic, _ = classify([{"role": "user", "content": p}])
    ov, oc, _ = classify([{"role": "user", "content": p}, {"role": "assistant", "content": r}])
    rows.append({"type": it.get("type", ""), "input_verdict": iv, "input_cats": ic,
                 "output_verdict": ov, "output_cats": oc})
    if i % 25 == 0:
        print(i, int(time.time() - t0), "s", flush=True)

n = len(rows)
def rate(key): return sum(x[key] == "unsafe" for x in rows)
ib, ob = rate("input_verdict"), rate("output_verdict")
print(f"\nBenign prompts: {n}")
print(f"INPUT-guard  blocked (unsafe): {ib}/{n} = {100*ib/n:.1f}%")
print(f"OUTPUT-guard blocked (unsafe): {ob}/{n} = {100*ob/n:.1f}%")

for mode, vk, ck in [("INPUT", "input_verdict", "input_cats"), ("OUTPUT", "output_verdict", "output_cats")]:
    cats = Counter(c for x in rows if x[vk] == "unsafe" for c in x[ck])
    print(f"\n{mode} unsafe category distribution: {dict(cats)}")
    byt = defaultdict(lambda: [0, 0])
    for x in rows:
        byt[x["type"]][0] += 1; byt[x["type"]][1] += (x[vk] == "unsafe")
    for t, (c, u) in sorted(byt.items()):
        if u: print(f"  {mode:6s} {t:28s} {u}/{c}")

json.dump(rows, open(OUT / "guard_results.json", "w"), ensure_ascii=False, indent=1)
print("\nsaved", OUT / "guard_results.json", flush=True)
