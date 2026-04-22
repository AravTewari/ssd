import types
import unittest

import torch

from ssd.engine.helpers.ddtree import (
    DDTreeEntry,
    build_best_first_tree,
    build_verify_mask,
    build_frontier_candidates,
    compile_verify_inputs,
    pack_tree_nodes,
    unpack_tree_nodes,
    walk_verified_tree,
)
from ssd.engine.helpers.speculate_types import SpeculateResult
from ssd.engine.sequence import Sequence
from ssd.engine.verifier import Verifier


def _build_entry(recovery_token: int, logits_q: torch.Tensor, branch_logits: torch.Tensor) -> DDTreeEntry:
    nodes = build_best_first_tree(logits_q, tree_budget=4)
    node_token_ids, node_depths, parents = pack_tree_nodes(nodes, tree_budget=4, device=logits_q.device)
    return DDTreeEntry(
        recovery_token=recovery_token,
        node_token_ids=node_token_ids,
        node_depths=node_depths,
        parents=parents,
        num_nodes=len(nodes),
        draft_tokens=torch.empty(0, dtype=torch.int64),
        logits_q=logits_q,
        branch_logits=branch_logits,
        predicted_target_features=None,
        frontier_candidates=build_frontier_candidates(
            recovery_token=recovery_token,
            branch_logits=branch_logits,
            nodes=nodes,
            frontier_count=2,
        ),
    )


class _FakeModelRunner:
    def __init__(self):
        self.config = types.SimpleNamespace(draft_backend="ddtree")
        self.verify_calls = []

    def call(self, method_name, *args):
        if method_name == "run_ddtree_verify":
            seqs, entries = args
            self.verify_calls.append((seqs, entries))
            return (
                [[11, 1]],
                [99],
                [torch.zeros((2, 8), dtype=torch.float32)],
                [1],
                [entries[0].num_nodes],
                0.001,
            )
        raise AssertionError(f"Unexpected method call {method_name}")


class DDTreeTests(unittest.TestCase):
    def test_build_best_first_tree_and_frontiers(self):
        logits_q = torch.tensor(
            [
                [0.0, 4.0, 3.0, -2.0],
                [0.0, 2.0, 5.0, -3.0],
            ],
            dtype=torch.float32,
        )
        branch_logits = torch.tensor(
            [
                [0.0, 5.0, 1.0, -1.0],
                [0.0, 1.0, 6.0, -1.0],
                [0.0, 0.0, 7.0, -1.0],
            ],
            dtype=torch.float32,
        )
        entry = _build_entry(recovery_token=10, logits_q=logits_q, branch_logits=branch_logits)

        self.assertEqual(entry.num_nodes, 4)
        self.assertEqual(entry.node_token_ids[0].item(), 1)
        self.assertEqual(entry.node_depths[0].item(), 1)
        self.assertLessEqual(entry.node_depths[:entry.num_nodes].max().item(), 2)
        self.assertIsNotNone(entry.frontier_candidates)
        self.assertEqual(len(entry.frontier_candidates), 2)
        self.assertEqual(entry.frontier_candidates[0].frontier_token_ids[0], 10)

    def test_verify_mask_is_ancestor_only(self):
        logits_q = torch.tensor(
            [
                [0.0, 4.0, 3.5, -2.0],
                [0.0, 5.0, 1.0, -3.0],
            ],
            dtype=torch.float32,
        )
        branch_logits = torch.tensor(
            [
                [0.0, 6.0, 1.0, -1.0],
                [0.0, 5.0, 1.0, -1.0],
                [0.0, 1.0, 7.0, -1.0],
            ],
            dtype=torch.float32,
        )
        entry = _build_entry(recovery_token=11, logits_q=logits_q, branch_logits=branch_logits)
        mask = build_verify_mask(prefix_len=3, parents=entry.parents, num_nodes=entry.num_nodes, device=torch.device("cpu"))
        q_len = entry.num_nodes + 1
        mask = mask.view(q_len, 3 + q_len)
        self.assertTrue(mask[0, :4].all())
        self.assertFalse(mask[0, 4:].any())
        # The second query row is a depth-2 child and should not attend to non-ancestor siblings.
        self.assertTrue(mask[2, 3].item())
        self.assertTrue(mask[2, 4].item())
        self.assertTrue(mask[2, 5].item())
        self.assertFalse(mask[2, 6].item())

    def test_compile_verify_inputs_uses_prefix_len_and_depths(self):
        logits_q = torch.tensor(
            [
                [0.0, 4.0, 3.5, -2.0],
                [0.0, 5.0, 1.0, -3.0],
            ],
            dtype=torch.float32,
        )
        branch_logits = torch.tensor(
            [
                [0.0, 6.0, 1.0, -1.0],
                [0.0, 5.0, 1.0, -1.0],
                [0.0, 1.0, 7.0, -1.0],
            ],
            dtype=torch.float32,
        )
        entry = _build_entry(recovery_token=11, logits_q=logits_q, branch_logits=branch_logits)
        verify_input_ids, verify_positions = compile_verify_inputs(
            recovery_token=entry.recovery_token,
            node_token_ids=entry.node_token_ids,
            node_depths=entry.node_depths,
            num_nodes=entry.num_nodes,
            prefix_len=7,
            device=torch.device("cpu"),
        )
        self.assertEqual(verify_input_ids.tolist()[0], 11)
        self.assertEqual(verify_positions.tolist()[0], 7)
        self.assertEqual(verify_positions.tolist()[1], 8)

    def test_walk_verified_tree_returns_accepted_path_and_bonus(self):
        logits_q = torch.tensor(
            [
                [0.0, 5.0, 4.0, -2.0],
                [0.0, 6.0, 1.0, -3.0],
            ],
            dtype=torch.float32,
        )
        branch_logits = torch.tensor(
            [
                [0.0, 7.0, 1.0, -1.0],
                [0.0, 6.0, 1.0, -1.0],
                [0.0, 1.0, 8.0, -1.0],
            ],
            dtype=torch.float32,
        )
        entry = _build_entry(recovery_token=11, logits_q=logits_q, branch_logits=branch_logits)
        nodes = unpack_tree_nodes(
            node_token_ids=entry.node_token_ids,
            node_depths=entry.node_depths,
            parents=entry.parents,
            num_nodes=entry.num_nodes,
        )
        flat_logits = torch.tensor(
            [
                [0.0, 9.0, 1.0, -1.0],   # root row matches token 1
                [0.0, 8.0, 1.0, -1.0],   # node row matches token 1 again
                [0.0, 1.0, 10.0, -1.0],  # leaf mismatch => recovery token 2
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )[: entry.num_nodes + 1]
        traversal = walk_verified_tree(
            recovery_token=entry.recovery_token,
            nodes=nodes,
            flat_logits=flat_logits,
        )
        self.assertEqual(traversal.accepted_suffix, [11, 1, 1])
        self.assertEqual(traversal.recovery_token, 2)
        self.assertEqual(traversal.accepted_query_indices, [0, 1, 2])
        self.assertEqual(traversal.visited_node_count, 2)

    def test_verifier_uses_target_ddtree_path(self):
        logits_q = torch.tensor(
            [
                [0.0, 4.0, 3.5, -2.0],
                [0.0, 5.0, 1.0, -3.0],
            ],
            dtype=torch.float32,
        )
        branch_logits = torch.tensor(
            [
                [0.0, 6.0, 1.0, -1.0],
                [0.0, 5.0, 1.0, -1.0],
                [0.0, 1.0, 7.0, -1.0],
            ],
            dtype=torch.float32,
        )
        entry = _build_entry(recovery_token=11, logits_q=logits_q, branch_logits=branch_logits)

        seq = Sequence([1, 2])
        seq.append_token(11)
        seq.num_cached_tokens = 2
        seq.num_draft_cached_tokens = 2
        entry.verify_input_ids, entry.verify_positions = compile_verify_inputs(
            recovery_token=entry.recovery_token,
            node_token_ids=entry.node_token_ids,
            node_depths=entry.node_depths,
            num_nodes=entry.num_nodes,
            prefix_len=seq.num_cached_tokens,
            device=torch.device("cpu"),
        )

        fake_runner = _FakeModelRunner()
        verifier = Verifier(
            lookahead=2,
            device=torch.device("cpu"),
            target_model_runner=fake_runner,
            tokenizer=None,
            metrics={
                "cache_hits": [],
                "accepted_suffix_lens_with_recovery": [],
                "accepted_suffix_lens_on_hit": [],
                "accepted_suffix_lens_on_miss": [],
                "target_verify_times": [],
            },
        )

        result = verifier.verify(
            [seq],
            SpeculateResult(
                speculations=torch.empty(0),
                logits_q=torch.empty(0),
                ddtree_entries=[entry],
            ),
        )

        self.assertEqual(result.new_suffixes, [[11, 1]])
        self.assertEqual(result.recovery_tokens, [99])
        self.assertEqual(result.ddtree_verified_node_counts, [1])
        self.assertEqual(len(fake_runner.verify_calls), 1)


if __name__ == "__main__":
    unittest.main()
