"""Extract S6 (Specialized Advice) flagged cases into a blind-annotation sheet.

Reads health_orsc_results.json; pulls every item where Llama Guard flagged S6
at the input and/or output stage. Emits a CSV with two blank columns for the
two annotators to independently mark over-refusal (y/n). Agreement rate on this
sheet calibrates the S6 rubric (strict vs policy-redirect).
"""
import json, csv, random
from pathlib import Path

HOME = Path.home() / "orr_spike"
data = json.load(open(HOME / "health_orsc_results.json", encoding="utf-8"))
SAMPLE = 25  # label this many if more S6 cases exist

def flagged_by(d):
    i = "S6" in d.get("input_cats", [])
    o = "S6" in d.get("output_cats", [])
    return "both" if (i and o) else ("input" if i else ("output" if o else ""))

s6 = [d for d in data if flagged_by(d)]
print(f"total items {len(data)}, S6-flagged {len(s6)}")
random.seed(0)
sheet = s6 if len(s6) <= SAMPLE else random.sample(s6, SAMPLE)

out = HOME / "s6_annotation.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "health_category", "flagged_by", "input_cats", "output_cats",
                "prompt", "response",
                "A_request_type", "A_response_type", "A_response_compliant",
                "B_request_type", "B_response_type", "B_response_compliant", "notes"])
    for i, d in enumerate(sheet):
        w.writerow([i, d.get("classification", ""), flagged_by(d),
                    ";".join(d.get("input_cats", [])), ";".join(d.get("output_cats", [])),
                    d["prompt"], d["response"], "", "", "", "", "", "", ""])
print(f"wrote {len(sheet)} rows to {out}")
