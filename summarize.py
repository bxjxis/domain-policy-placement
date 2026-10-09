"""Recover mechanism statistics from legacy eight-configuration records.

No model dependencies or generation. This does NOT calculate policy-labelled ORR.
Input categories may be shared only for identical prompts and guard settings.
Usage: python summarize.py TAG [TAG ...] --data-dir /path/to/orr_spike
       python summarize.py --combo /path/to/combo_results_MODEL.json
"""
import argparse
import csv
import itertools
import json
import re
from pathlib import Path


CONFIGS = [f"P{p}R{r}O{o}" for p, r, o in itertools.product((0, 1), repeat=3)]


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty JSON list")
    return rows


def categories(value):
    # Empty/missing category lists on a blocked item are unknown, not non-S6.
    if value is None or value == []:
        return None
    if not isinstance(value, list) or any(
        not isinstance(cat, str) or not re.fullmatch(r"S\d+", cat) for cat in value
    ):
        raise ValueError(f"Invalid category list: {value!r}")
    return frozenset(value)


def input_index(rows):
    index = {}
    for row in rows:
        prompt = row["prompt"]
        verdict = row.get("input_verdict")
        if verdict not in ("safe", "unsafe"):
            raise ValueError("Recovery file has an unknown input verdict")
        item = (verdict == "unsafe", categories(row.get("input_cats")))
        if prompt in index and index[prompt] != item:
            raise ValueError("Recovery file has conflicting duplicate prompts")
        index[prompt] = item
    return index


def decision(atom, reminder, p, o, drop_s6=False):
    ib, ob = atom["input_blocked"], atom["output_blocked"][reminder]
    if drop_s6:
        if p and ib:
            if atom["input_cats"] is None:
                return None
            ib = atom["input_cats"] != {"S6"}
        if o and ob:
            if atom["output_cats"][reminder] is None:
                # Unknown output matters only when the request reaches it.
                if not (p and ib):
                    return None
            else:
                ob = atom["output_cats"][reminder] != {"S6"}
    if p and ib:
        return "input", "blocked_input", None
    tr = atom["target_refusal"][reminder]
    if o and ob:
        return "output", "blocked_output", tr
    return "none", "model_refusal" if tr else "answered", tr


def recover_atoms(records, inputs):
    grouped = {}
    for row in records:
        prompt, config = row["prompt"], row["config"]
        if not isinstance(prompt, str) or config not in CONFIGS:
            raise ValueError("Invalid prompt or configuration")
        group = grouped.setdefault(prompt, {})
        if config in group:
            raise ValueError(f"Duplicate prompt/configuration: {config}")
        group[config] = row
    warnings = []
    legacy_flags = 0
    atoms = []
    for prompt, group in grouped.items():
        if set(group) != set(CONFIGS):
            raise ValueError("Each prompt must have exactly all eight configurations")
        input_row = group["P1R0O0"]
        stage = input_row["pipeline_stop_stage"]
        if stage not in ("input", "none"):
            raise ValueError("Invalid input-only stop stage")
        ib = stage == "input"
        if type(input_row.get("input_blocked")) is not bool or input_row["input_blocked"] != ib:
            raise ValueError("Input-only verdict disagrees with stop stage")
        ic = categories(input_row.get("input_cats"))
        if prompt in inputs:
            old_ib, old_cats = inputs[prompt]
            if ib != old_ib:
                raise ValueError("Recovery input verdict disagrees with harness: settings/data may differ")
            if ic is not None and old_cats is not None and ic != old_cats:
                raise ValueError("Recovery input categories disagree with harness")
            ic = ic if ic is not None else old_cats
        atom = dict(prompt=prompt, input_blocked=ib, input_cats=ic,
                    target_refusal={}, output_blocked={}, output_cats={})
        for reminder in (0, 1):
            base = group[f"P0R{reminder}O0"]
            output = group[f"P0R{reminder}O1"]
            tr = base.get("target_refusal")
            if type(tr) is not bool or type(output.get("output_blocked")) is not bool:
                raise ValueError("Missing boolean atomic model/output verdict")
            atom["target_refusal"][reminder] = tr
            atom["output_blocked"][reminder] = output["output_blocked"]
            atom["output_cats"][reminder] = categories(output.get("output_cats"))
        for config, row in group.items():
            p, reminder, o = map(int, (config[1], config[3], config[5]))
            expected = decision(atom, reminder, p, o)
            actual = (row["pipeline_stop_stage"], row["final_response_label"], row.get("target_refusal"))
            if actual != expected:
                raise ValueError(f"Stored pipeline disagrees with recovered atoms: {config}")
            expected_ib = ib if p else None
            # Ignore only the identified legacy True flag after a passed input guard.
            flag = row.get("input_blocked")
            if flag != expected_ib or (flag is not None and type(flag) is not bool):
                if p and not ib and expected[0] == "output" and flag is True:
                    legacy_flags += 1
                else:
                    raise ValueError(f"Unexpected input flag disagreement: {config}")
            expected_ob = atom["output_blocked"][reminder] if o and expected[0] != "input" else None
            flag = row.get("output_blocked")
            if flag != expected_ob or (flag is not None and type(flag) is not bool):
                raise ValueError(f"Unexpected output flag disagreement: {config}")
            if expected[0] == "output" and categories(row.get("output_cats")) != atom["output_cats"][reminder]:
                raise ValueError(f"Output categories disagree: {config}")
        atoms.append(atom)
    if legacy_flags:
        warnings.append(f"Ignored {legacy_flags} known legacy input_blocked=True flags at output stops; originals unchanged.")
    missing = sum(a["input_blocked"] and a["input_cats"] is None for a in atoms)
    if missing:
        warnings.append(f"Input categories unknown for {missing} blocked prompts; S6 results may be unavailable.")
    missing_o = sum(a["output_blocked"][r] and a["output_cats"][r] is None for a in atoms for r in (0, 1))
    if missing_o:
        warnings.append(f"Output categories unknown for {missing_o} blocked prompt/R pairs.")
    return atoms, warnings


def summarize(records, inputs=None):
    atoms, warnings = recover_atoms(records, inputs or {})
    results = []
    for config in CONFIGS:
        p, reminder, o = map(int, (config[1], config[3], config[5]))
        result = dict(config=config, n=len(atoms), no_answer=0, input_stops=0,
                      output_stops=0, no_guard_stop=0, atomic_self_refusal=0,
                      observed_self_refusal=0, surviving_self_refusal=0,
                      new_from_input=0, new_from_output=0, guard_blocks=0,
                      stopped_non_s6_only=0, unknown_stop_categories=0,
                      no_answer_excl_s6_only=0, guard_blocks_excl_s6_only=0,
                      unknown_s6_replay=0)
        for atom in atoms:
            stage, label, observed = decision(atom, reminder, p, o)
            tr = atom["target_refusal"][reminder]
            result["atomic_self_refusal"] += tr
            result["observed_self_refusal"] += observed is True
            result["surviving_self_refusal"] += label == "model_refusal"
            result["no_answer"] += label != "answered"
            result[{"input": "input_stops", "output": "output_stops", "none": "no_guard_stop"}[stage]] += 1
            result["new_from_input"] += stage == "input" and not tr
            result["new_from_output"] += stage == "output" and not tr
            if stage != "none":
                result["guard_blocks"] += 1
                cats = atom["input_cats"] if stage == "input" else atom["output_cats"][reminder]
                if cats is None:
                    result["unknown_stop_categories"] += 1
                else:
                    result["stopped_non_s6_only"] += cats != {"S6"}
            replay = decision(atom, reminder, p, o, drop_s6=True)
            if replay is None:
                result["unknown_s6_replay"] += 1
            else:
                result["no_answer_excl_s6_only"] += replay[1] != "answered"
                result["guard_blocks_excl_s6_only"] += replay[0] != "none"
        assert result["no_answer"] == result["atomic_self_refusal"] + result["new_from_input"] + result["new_from_output"]
        if result["unknown_stop_categories"]:
            result["stopped_non_s6_only"] = None
        if result["unknown_s6_replay"]:
            result["no_answer_excl_s6_only"] = result["guard_blocks_excl_s6_only"] = None
        result["no_answer_rate"] = result["no_answer"] / result["n"]
        results.append(result)
    transitions = {"answered_to_refused": 0, "refused_to_answered": 0,
                   "both_refused": 0, "both_answered": 0}
    for atom in atoms:
        before, after = (atom["target_refusal"][r] for r in (0, 1))
        key = ("both_refused" if after else "refused_to_answered") if before else (
            "answered_to_refused" if after else "both_answered")
        transitions[key] += 1
    overlaps = []
    for reminder in (0, 1):
        overlaps.append(dict(R=reminder,
            input_output_overlap=sum(a["input_blocked"] and a["output_blocked"][reminder] for a in atoms),
            input_model_refusal_overlap=sum(a["input_blocked"] and a["target_refusal"][reminder] for a in atoms),
            output_model_refusal_overlap=sum(a["output_blocked"][reminder] and a["target_refusal"][reminder] for a in atoms)))
    return dict(n=len(atoms), warnings=warnings, configurations=results,
                reminder_transitions=transitions, overlaps=overlaps,
                scope="Mechanism counts from keyword refusal signals; not policy ORR or ASR.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="*")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "orr_spike")
    parser.add_argument("--combo", type=Path, action="append", default=[])
    parser.add_argument("--input-results", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args(argv)
    paths = args.combo + [args.data_dir / f"combo_results_{tag}.json" for tag in args.tags]
    if not paths:
        parser.error("Supply at least one TAG or --combo file")
    source = args.input_results or args.data_dir / "health_orsc_results.json"
    try:
        if args.input_results is not None and not source.exists():
            raise ValueError(f"Explicit recovery file does not exist: {source}")
        inputs = input_index(read_rows(source)) if source.exists() else {}
        reports = []
        for path in paths:
            report = summarize(read_rows(path), inputs)
            report.update(model=path.stem.removeprefix("combo_results_"), source=str(path),
                          input_recovery_source=str(source) if source.exists() else None)
            reports.append(report)
        # Check identical prompt sets before interpreting models as paired.
        sets = [{row["prompt"] for row in read_rows(path)} for path in paths]
        if any(prompts != sets[0] for prompts in sets[1:]):
            raise ValueError("Models have different prompt sets; paired comparison is unavailable")
        outputs = [p for p in (args.json_out, args.csv_out) if p is not None]
        originals = {p.resolve() for p in paths + [source]}
        if any(p.resolve() in originals for p in outputs):
            raise ValueError("Output must not overwrite a source file")
        if len({p.resolve() for p in outputs}) != len(outputs):
            raise ValueError("JSON and CSV outputs must be different files")
        for report in reports:
            print(f"\n=== {report['model']} N={report['n']} (mechanism only) ===")
            for warning in report["warnings"]:
                print("WARNING:", warning)
            print("config     noAns  in/out/none  atomic observed surviving newP newO guards S6bypassNoAns")
            for r in report["configurations"]:
                stops = f"{r['input_stops']}/{r['output_stops']}/{r['no_guard_stop']}"
                bypass = r["no_answer_excl_s6_only"]
                print(f"{r['config']:8} {r['no_answer']:5} {stops:>12} {r['atomic_self_refusal']:7} "
                      f"{r['observed_self_refusal']:8} {r['surviving_self_refusal']:9} "
                      f"{r['new_from_input']:4} {r['new_from_output']:4} {r['guard_blocks']:6} "
                      f"{str(bypass) if bypass is not None else 'NA':>13}")
            print("R transitions:", report["reminder_transitions"])
            print("Shadow overlaps:", report["overlaps"])
        print("\nS6 bypass is a category-filter counterfactual, NOT policy ORR. Unknown categories => NA.")
        print("atomic = all prompts; observed = model reached; surviving = refusal delivered without guard stop.")
        if args.json_out:
            with args.json_out.open("w", encoding="utf-8") as handle:
                json.dump(reports, handle, ensure_ascii=False, indent=2)
        if args.csv_out:
            flat = [dict(model=report["model"], **row) for report in reports for row in report["configurations"]]
            with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
