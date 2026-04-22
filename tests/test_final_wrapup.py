import importlib.util
import tempfile
import types
import unittest
from pathlib import Path


def _load_wrapup_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_final_wrapup.py"
    spec = importlib.util.spec_from_file_location("eval_final_wrapup", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_result(root: Path, mode: str, batch_size: int, throughput: float, accepted: float, **extra) -> dict:
    artifact_dir = root / mode / f"b{batch_size}"
    prompt_group_metrics = {
        "g0": {"throughput_tok_s": throughput * 0.95, "accepted_suffix_mean": accepted * 0.98},
        "g1": {"throughput_tok_s": throughput * 1.00, "accepted_suffix_mean": accepted * 1.00},
        "g2": {"throughput_tok_s": throughput * 1.05, "accepted_suffix_mean": accepted * 1.02},
    }
    _write_json(artifact_dir / "prompt_group_metrics.json", prompt_group_metrics)
    row = {
        "mode": mode,
        "batch_size": batch_size,
        "throughput_tok_s": throughput,
        "accepted_suffix_mean": accepted,
        "cache_hit_mean": extra.pop("cache_hit_mean", 0.0),
        "artifact_dir": str(artifact_dir),
    }
    row.update(extra)
    return row


class FinalWrapupTests(unittest.TestCase):
    def test_generate_artifacts_creates_summary_and_figures(self):
        wrapup = _load_wrapup_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = types.SimpleNamespace(
                target="/models/qwen3-8b",
                training_metadata="/tmp/training_metadata.json",
                dflash_draft="/models/dflash",
                dflash_predictor="/models/predictor",
                batch_sizes="1,2,4",
                ar_k_values="4,6,8,12,15",
                ddtree_tree_budgets="8,16",
                ddtree_frontier_counts="1,2",
                main_output_len=32,
                extra_output_len=128,
                max_prompts=78,
                bootstrap_samples=200,
                bootstrap_seed=7,
            )

            ar_main = []
            ar_extra = []
            for batch_size, off_t, oracle_t, normal_t in [(1, 12.0, 18.0, 16.0), (2, 20.0, 28.0, 24.0), (4, 30.0, 42.0, 36.0)]:
                ar_main.extend(
                    [
                        _make_result(root / "ar_main", "ar_async_exact_off_normal", batch_size, off_t, 2.5, k=4, draft_label="qwen0.6b"),
                        _make_result(root / "ar_main", "ar_async_exact_on_oracle", batch_size, oracle_t, 2.8, k=4, draft_label="qwen0.6b"),
                        _make_result(root / "ar_main", "ar_async_normal", batch_size, normal_t, 2.7, k=4, draft_label="qwen0.6b", cache_hit_mean=0.25),
                    ]
                )
                ar_extra.extend(
                    [
                        _make_result(root / "ar_extra", "ar_async_exact_off_normal", batch_size, off_t * 0.9, 2.4, k=4, draft_label="qwen0.6b"),
                        _make_result(root / "ar_extra", "ar_async_exact_on_oracle", batch_size, oracle_t * 0.92, 2.6, k=4, draft_label="qwen0.6b"),
                    ]
                )

            dflash_main = []
            dflash_extra = []
            dflash_matrix_b = []
            for batch_size, exact_t, oracle_t, realized_t in [(1, 7.0, 7.2, 5.5), (2, 11.0, 11.3, 8.4), (4, 16.0, 16.2, 12.0)]:
                dflash_main.extend(
                    [
                        _make_result(root / "dflash_main", "exact_dflash", batch_size, exact_t, 2.1),
                        _make_result(root / "dflash_main", "dflash_ssd_exact_off_normal", batch_size, exact_t * 0.99, 2.1, cache_hit_mean=0.0),
                        _make_result(root / "dflash_main", "dflash_ssd_exact_on_oracle", batch_size, oracle_t, 2.15, cache_hit_mean=0.2),
                        _make_result(root / "dflash_main", "dflash_ssd_predicted_off_oracle", batch_size, oracle_t * 0.8, 1.8),
                        _make_result(root / "dflash_main", "dflash_ssd_predicted_on_oracle", batch_size, oracle_t * 0.78, 1.8, cache_hit_mean=0.15),
                        _make_result(root / "dflash_main", "dflash_ssd_predicted_on_normal", batch_size, realized_t, 1.9, cache_hit_mean=0.08, joint_branch_recall=0.1, cache_committed_token_fraction=0.1),
                    ]
                )
                dflash_extra.extend(
                    [
                        _make_result(root / "dflash_extra", "dflash_ssd_exact_off_normal", batch_size, exact_t * 0.92, 2.0),
                        _make_result(root / "dflash_extra", "dflash_ssd_exact_on_oracle", batch_size, oracle_t * 0.91, 2.05),
                    ]
                )
                dflash_matrix_b.append(
                    _make_result(
                        root / "dflash_matrix_b",
                        "dflash_ssd_predicted_on_normal",
                        batch_size,
                        realized_t,
                        1.9,
                        fanout_template_name="baseline48",
                        cache_hit_mean=0.08,
                        cache_committed_token_fraction=0.1,
                        joint_branch_recall=0.1,
                    )
                )

            ddtree_main = []
            ddtree_extra = []
            for batch_size, exact_t, oracle_t, realized_t in [(1, 8.0, 8.1, 6.3), (2, 12.0, 12.2, 9.4), (4, 17.0, 17.3, 13.6)]:
                ddtree_main.extend(
                    [
                        _make_result(root / "ddtree_main", "ddtree", batch_size, exact_t, 2.0, tree_budget=8, frontier_count=1),
                        _make_result(root / "ddtree_main", "ddtree_ssd_exact_off", batch_size, exact_t * 0.995, 2.0, tree_budget=8, frontier_count=1),
                        _make_result(root / "ddtree_main", "ddtree_ssd_exact_on_oracle", batch_size, oracle_t, 2.02, tree_budget=8, frontier_count=1, cache_hit_mean=0.18),
                        _make_result(root / "ddtree_main", "ddtree_ssd_predicted_on_surrogate", batch_size, realized_t, 1.85, tree_budget=8, frontier_count=1, cache_hit_mean=0.12),
                    ]
                )
                ddtree_extra.extend(
                    [
                        _make_result(root / "ddtree_extra", "ddtree", batch_size, exact_t * 0.94, 1.95, tree_budget=8, frontier_count=1),
                        _make_result(root / "ddtree_extra", "ddtree_ssd_exact_off", batch_size, exact_t * 0.935, 1.95, tree_budget=8, frontier_count=1),
                        _make_result(root / "ddtree_extra", "ddtree_ssd_exact_on_oracle", batch_size, oracle_t * 0.945, 1.98, tree_budget=8, frontier_count=1),
                    ]
                )

            summary = wrapup._generate_artifacts(
                args=args,
                artifact_root=root / "out",
                ar_candidates={"qwen0.6b": "/models/qwen0.6b"},
                missing_ar=["llama1b"],
                ar_main_results=ar_main,
                ar_extra_results=ar_extra,
                dflash_main_results=dflash_main,
                dflash_main_matrix_b=dflash_matrix_b,
                dflash_extra_results=dflash_extra,
                ddtree_main_results=ddtree_main,
                ddtree_extra_results=ddtree_extra,
            )

            self.assertTrue((root / "out" / "final_summary.json").exists())
            self.assertTrue((root / "out" / "final_tables.md").exists())
            self.assertTrue((root / "out" / "appendix_dflash_diagnostics.md").exists())
            self.assertTrue((root / "out" / "appendix_extra_regime.md").exists())
            self.assertTrue((root / "out" / "figure_oracle_ceiling.png").exists())
            self.assertTrue((root / "out" / "figure_normalized_speedup.png").exists())
            self.assertTrue((root / "out" / "figure_budget_frontier.png").exists())
            self.assertTrue((root / "out" / "figure_error_bars.png").exists())
            self.assertTrue(summary["recommendation"]["stop_project"])


if __name__ == "__main__":
    unittest.main()
