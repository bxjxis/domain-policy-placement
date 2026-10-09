"""Regression tests for recovery from legacy records (no GPU required)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import summarize


def fixture():
    # Includes: masked refusal, S6 input -> non-S6 output, mixed categories,
    # non-S6 input -> S6 output, and newly blocked non-refusal responses.
    definitions = [
        (True, ["S6"], [False, True], [True, True], [["S2"], ["S6"]]),
        (True, ["S2", "S6"], [True, False], [True, False], [["S6"], []]),
        (False, [], [True, False], [True, True], [["S6"], ["S2"]]),
        (False, [], [False, False], [True, False], [["S2"], []]),
    ]
    records, old = [], []
    for idx, (ib, ic, trs, obs, ocs) in enumerate(definitions):
        prompt = f"question {idx}"
        old.append(dict(prompt=prompt, input_verdict="unsafe" if ib else "safe", input_cats=ic))
        for cfg in summarize.CONFIGS:
            p, r, o = map(int, (cfg[1], cfg[3], cfg[5]))
            if p and ib:
                stage, label, tr, ob, oc = "input", "blocked_input", None, None, None
            elif o and obs[r]:
                stage, label, tr, ob, oc = "output", "blocked_output", trs[r], True, ocs[r]
            else:
                stage, label, tr, ob, oc = "none", "model_refusal" if trs[r] else "answered", trs[r], False if o else None, None
            # Replicate the legacy harness's erroneous True flag on output stops.
            stored_ib = (True if stage in ("input", "output") else False) if p else None
            records.append(dict(prompt=prompt, config=cfg, input_blocked=stored_ib,
                                target_refusal=tr, output_blocked=ob, output_cats=oc,
                                pipeline_stop_stage=stage, final_response_label=label))
    return records, old


class RecoveryTests(unittest.TestCase):
    def test_masked_refusals_and_marginal_blocks(self):
        records, old = fixture()
        snapshot = copy.deepcopy(records)
        report = summarize.summarize(records, summarize.input_index(old))
        row = next(r for r in report["configurations"] if r["config"] == "P1R0O1")
        self.assertEqual((row["atomic_self_refusal"], row["observed_self_refusal"],
                          row["surviving_self_refusal"]), (2, 1, 0))
        self.assertEqual((row["no_answer"], row["new_from_input"], row["new_from_output"]), (4, 1, 1))
        self.assertEqual(row["stopped_non_s6_only"], 2)
        # Simply subtracting S6 input stops misses downstream non-S6 stops.
        self.assertEqual(row["guard_blocks_excl_s6_only"], 3)
        self.assertEqual(row["no_answer_excl_s6_only"], 4)
        self.assertEqual(report["reminder_transitions"], dict(answered_to_refused=1,
                         refused_to_answered=2, both_refused=0, both_answered=1))
        self.assertEqual(report["overlaps"][0]["input_output_overlap"], 2)
        self.assertTrue(report["warnings"])
        self.assertEqual(records, snapshot)

    def test_unknown_categories_are_not_silently_safe(self):
        records, _ = fixture()
        report = summarize.summarize(records)
        rows = {r["config"]: r for r in report["configurations"]}
        self.assertIsNone(rows["P1R0O1"]["stopped_non_s6_only"])
        self.assertIsNone(rows["P1R0O1"]["no_answer_excl_s6_only"])
        self.assertIsNotNone(rows["P0R0O1"]["no_answer_excl_s6_only"])

    def test_conflicting_input_source_rejected(self):
        records, old = fixture()
        old[0]["input_verdict"] = "safe"
        with self.assertRaisesRegex(ValueError, "input verdict disagrees"):
            summarize.summarize(records, summarize.input_index(old))

    def test_missing_and_duplicate_configurations_rejected(self):
        records, old = fixture()
        inputs = summarize.input_index(old)
        with self.assertRaisesRegex(ValueError, "all eight"):
            summarize.summarize(records[:-1], inputs)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize.summarize(records + [records[0]], inputs)

    def test_inconsistent_refusal_rejected(self):
        records, old = fixture()
        row = next(r for r in records if r["config"] == "P0R0O1")
        row["target_refusal"] = True
        with self.assertRaisesRegex(ValueError, "disagrees with recovered atoms"):
            summarize.summarize(records, summarize.input_index(old))

    def test_s6_only_is_set_based(self):
        self.assertEqual(summarize.categories(["S6", "S6"]), {"S6"})
        self.assertNotEqual(summarize.categories(["S6", "S2"]), {"S6"})
        self.assertIsNone(summarize.categories([]))

    def test_cli_exports_without_touching_sources(self):
        records, old = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combo, recovery = root / "combo_results_test.json", root / "health_orsc_results.json"
            combo.write_text(json.dumps(records), encoding="utf-8")
            recovery.write_text(json.dumps(old), encoding="utf-8")
            before = combo.read_bytes()
            self.assertEqual(summarize.main(["test", "--data-dir", directory,
                "--json-out", str(root / "summary.json"), "--csv-out", str(root / "summary.csv")]), 0)
            report = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report[0]["configurations"]), 8)
            self.assertTrue((root / "summary.csv").is_file())
            self.assertEqual(combo.read_bytes(), before)
            with self.assertRaises(SystemExit) as error:
                summarize.main(["test", "--data-dir", directory, "--json-out", str(combo)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(combo.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
