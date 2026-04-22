import importlib.util
import unittest
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_ar_ssd_baseline.py"
    spec = importlib.util.spec_from_file_location("eval_ar_ssd_baseline", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvalArSSDBaselineTests(unittest.TestCase):
    def test_cycle_summary_computes_prompt_group_stats(self):
        mod = _load_module()
        prompt_records = [
            {"group_key": "alpaca:0", "dataset_name": "alpaca", "prompt_index": 0},
            {"group_key": "gsm:1", "dataset_name": "gsm", "prompt_index": 1},
        ]
        rows = [
            {
                "seq_id": 0,
                "batch_size": 2,
                "cycle_idx": 0,
                "accepted_len": 3,
                "recovery_token": 10,
                "cache_hit": True,
                "draft_service_ms": 5.0,
                "post_verify_feedback_ms": 1.0,
                "target_verify_ms": 7.0,
                "total_cycle_ms": 15.0,
                "tokens_committed_this_cycle": 3,
            },
            {
                "seq_id": 1,
                "batch_size": 2,
                "cycle_idx": 0,
                "accepted_len": 2,
                "recovery_token": 11,
                "cache_hit": False,
                "draft_service_ms": 5.0,
                "post_verify_feedback_ms": 1.0,
                "target_verify_ms": 7.0,
                "total_cycle_ms": 15.0,
                "tokens_committed_this_cycle": 2,
            },
        ]
        enriched = mod._enrich_cycle_rows(rows, prompt_records)
        summary = mod._summarize_cycle_rows(enriched)

        self.assertAlmostEqual(summary["cache_hit_mean"], 0.5)
        self.assertAlmostEqual(summary["miss_rate"], 0.5)
        self.assertAlmostEqual(summary["draft_service_ms"], 5.0)
        self.assertAlmostEqual(summary["target_verify_ms"], 7.0)
        self.assertEqual(set(summary["prompt_group_metrics"].keys()), {"alpaca:0", "gsm:1"})
        self.assertIn("throughput_tok_s", summary["prompt_group_summary"])
        self.assertIn("accepted_suffix_mean", summary["prompt_group_summary"])


if __name__ == "__main__":
    unittest.main()
