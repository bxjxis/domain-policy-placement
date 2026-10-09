# Where Should Domain Safety Policies Live?

A small-scale trustworthy AI research project on policy placement in safety pipelines for financial and medical assistants.

**Research question:** When the same domain policy is supplied to the answering model, the input guard, or the output guard, which placement gives users more compliant, useful answers while limiting harmful assistance?

**Status:** Research in progress. This repository contains pilot code for defense stacking, analysis tools, an annotation rubric, and a draft medical policy. The policy-placement experiment described below is planned; final results are not yet available.

## Motivation

A safety pipeline can reject a benign request at several stages. An input guard may block the question, the answering model may refuse, or an output guard may block the answer. Improving one stage may have little effect on what the user ultimately receives.

For example, explaining how to identify fraudulent financial statements is allowed under our intended deployment policy, while providing operational help to fabricate them is prohibited. We study whether making this boundary explicit at different stages improves the complete pipeline.

```text
User request -> Input guard -> Answering model -> Output guard -> Delivered answer
```

## Planned experiment

We use one frozen policy per domain, with allow, deny, and exception clauses. The substantive policy is shared across stages; task instructions and the guard-template rendering are recorded separately.

| Switch | Default setting (0) | Policy setting (1) |
| --- | --- | --- |
| M: answering model | Baseline prompt | Domain policy in the system prompt |
| I: input guard | Default safety policy | Domain-adapted safety policy |
| O: output guard | Default safety policy | Domain-adapted safety policy |

The three switches produce eight configurations. **Both guards remain present in all eight configurations.** Two additional references use the answering model alone, with and without the domain policy. Self-reminders are disabled in this experiment.

Planned models:

- Answering models: `Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`.
- Input and output guard: `meta-llama/Llama-Guard-3-8B`.

Under a stateless pipeline in which guards only block or pass, we can reuse two model responses and six guard decisions per request to compose the configurations. This reuse does not apply to pipelines that rewrite requests, retry generation, or otherwise alter subsequent stages.

## Data and evaluation

Medical data supports the initial pilot. The main financial evaluation will use benign/harmful request pairs with similar topics and wording but different intent. Each benign request will include a reason it is allowed and a minimum acceptable answer.

Planned data sources include:

- [Health-ORSC-Bench](https://github.com/ZhihaoZhang97/Health-ORSC-Bench): medical boundary requests and harmful seeds.
- [FinSafetyBench](https://github.com/sustech-nlp/FinSafetyBench): financial harmful requests in English and Chinese.
- [FinSafeGuard](https://huggingface.co/datasets/domyn/FinSafeGuard): candidate financial conversations for supplementary component evaluation.

Source labels will be reviewed against our deployment policy. Development and test examples will be separated by source/template family, keeping related variants together. Policies and test annotations will be frozen before the final comparison. Upstream datasets are not bundled here; their licenses and attribution requirements apply separately.

Primary outcomes, reported for every configuration:

1. **Benign usefulness:** the proportion of benign requests that receive a compliant answer meeting the minimum requirements.
2. **End-to-end harmful assistance:** the proportion of harmful requests that receive harmful help after all pipeline stages.

Additional analysis will attribute refusals and blocks to stages, examine paired boundary behavior, and report inference cost. A safe explanation in response to a harmful request can count as safe handling. Blocking a harmful answer to a benign request is a correct block.

Final judgments will distinguish refusal, policy compliance, and answer adequacy, using a frozen LLM-judge rubric with human auditing. Keyword refusal counts are preliminary mechanism measurements. Comparisons will use paired statistics and confidence intervals; absence of a significant safety difference will not be treated as proof of unchanged safety.

## Expected contribution

The intended contribution is a controlled comparison of policy placement, an annotated financial evaluation set, and an explanation of how individual stages affect final outcomes. Custom-policy guards, opposite-intent financial pairs, and per-layer attribution have prior work; we do not claim to introduce those ideas. Conclusions will be limited to the tested models, policies, and datasets.

Relevant prior work includes [FinGuard](https://arxiv.org/abs/2605.29427), [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), and prior work on [defense stacking and refusal attribution](https://arxiv.org/abs/2608.28327). The precise overlap with related policy-placement studies remains under review.

## Repository guide

| File | Purpose |
| --- | --- |
| `combo_harness.py` | Existing pilot harness for input guard / self-reminder / output guard combinations; these switches differ from the planned M/I/O experiment |
| `summarize.py` | Reconstructs and summarizes saved pilot results without rerunning inference |
| `summarizer_usage.md` | Usage and interpretation of the existing summarizer |
| `test_summarize.py` | Synthetic-data tests for result reconstruction and summarization |
| `s6_rubric.md` | Existing medical annotation rubric; answer-adequacy assessment will be added for the main experiment |
| `policies/medical.md` | Draft medical deployment policy and guard-category mapping |
| `health_orsc.py`, `guard_classify.py`, `spike_orr.py`, `extract_s6.py` | Earlier pilot and inspection scripts |
| `project_plan.md` | Earlier planning document; some sections describe previous scope and should be read with that distinction |

The placement harness, finance policy, judge script, and final evaluation set are pending. This README describes the current research direction.

## Using the existing pilot tools

The inference scripts require access to the relevant model weights, a suitable PyTorch/Transformers environment, and sufficient GPU memory. Some scripts use the local path `~/orr_spike`; review dataset and output paths before running. A pinned inference environment will be recorded for the main experiment.

To summarize existing result files:

```powershell
python summarize.py Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct --data-dir "PATH_TO_RESULTS" --json-out summary.json --csv-out summary.csv
```

To run the existing summarizer checks:

```powershell
python -m unittest test_summarize.py
```

See [summarizer_usage.md](summarizer_usage.md) for expected files and field definitions. No final placement scores or deployment recommendations are claimed at this stage.
