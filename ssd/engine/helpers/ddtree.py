from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable

import torch


@dataclass(frozen=True)
class DDTreeNode:
    index: int
    token_id: int
    parent_index: int
    depth: int
    path_log_weight: float
    path_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class DDTreeFrontierCandidate:
    frontier_token_ids: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    accepted_len_idx: int
    recovery_token: int
    score: float


@dataclass
class DDTreeEntry:
    recovery_token: int
    node_token_ids: torch.Tensor
    node_depths: torch.Tensor
    parents: torch.Tensor
    num_nodes: int
    draft_tokens: torch.Tensor
    logits_q: torch.Tensor
    branch_logits: torch.Tensor
    predicted_target_features: torch.Tensor | None = None
    frontier_candidates: list[DDTreeFrontierCandidate] | None = None
    verify_input_ids: torch.Tensor | None = None
    verify_positions: torch.Tensor | None = None

    @property
    def max_depth(self) -> int:
        if self.num_nodes == 0:
            return 0
        return int(self.node_depths[:self.num_nodes].max().item())


@dataclass
class DDTreeVerifyOutcome:
    accepted_suffix: list[int]
    recovery_token: int
    matched_depth: int
    visited_node_count: int


@dataclass
class DDTreeVerifyTraversal:
    accepted_suffix: list[int]
    recovery_token: int
    matched_depth: int
    visited_node_count: int
    accepted_query_indices: list[int]


def _topk_log_probs(logits_q: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits_q.float(), dim=-1)
    k = min(int(topk), int(log_probs.shape[-1]))
    return torch.topk(log_probs, k=k, dim=-1)


def build_best_first_tree(logits_q: torch.Tensor, tree_budget: int) -> list[DDTreeNode]:
    if logits_q.dim() != 2:
        raise ValueError(f"logits_q must be rank-2, got shape={tuple(logits_q.shape)}")
    if tree_budget <= 0:
        raise ValueError("tree_budget must be > 0")

    lookahead = int(logits_q.shape[0])
    if lookahead == 0:
        return []

    top_log_probs, top_token_ids = _topk_log_probs(logits_q, topk=tree_budget)
    nodes: list[DDTreeNode] = []
    heap: list[tuple[float, int, int, int, float, tuple[int, ...]]] = []
    heapq.heappush(heap, (-float(top_log_probs[0, 0].item()), -1, 1, 0, 0.0, ()))

    while heap and len(nodes) < tree_budget:
        neg_score, parent_index, depth, rank, parent_log_weight, parent_path = heapq.heappop(heap)
        token_id = int(top_token_ids[depth - 1, rank].item())
        edge_log_prob = float(top_log_probs[depth - 1, rank].item())
        path_log_weight = parent_log_weight + edge_log_prob
        path_token_ids = parent_path + (token_id,)
        node_index = len(nodes)
        nodes.append(
            DDTreeNode(
                index=node_index,
                token_id=token_id,
                parent_index=parent_index,
                depth=depth,
                path_log_weight=path_log_weight,
                path_token_ids=path_token_ids,
            )
        )

        if rank + 1 < top_token_ids.shape[1]:
            sibling_score = parent_log_weight + float(top_log_probs[depth - 1, rank + 1].item())
            heapq.heappush(
                heap,
                (-sibling_score, parent_index, depth, rank + 1, parent_log_weight, parent_path),
            )

        if depth < lookahead:
            child_score = path_log_weight + float(top_log_probs[depth, 0].item())
            heapq.heappush(
                heap,
                (-child_score, node_index, depth + 1, 0, path_log_weight, path_token_ids),
            )

    return nodes


def pack_tree_nodes(nodes: list[DDTreeNode], tree_budget: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = torch.full((tree_budget,), -1, dtype=torch.int64, device=device)
    depths = torch.full((tree_budget,), -1, dtype=torch.int64, device=device)
    parents = torch.full((tree_budget,), -1, dtype=torch.int64, device=device)
    for node in nodes[:tree_budget]:
        token_ids[node.index] = node.token_id
        depths[node.index] = node.depth
        parents[node.index] = node.parent_index
    return token_ids, depths, parents


def unpack_tree_nodes(
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    parents: torch.Tensor,
    num_nodes: int,
) -> list[DDTreeNode]:
    nodes: list[DDTreeNode] = []
    for node_index in range(num_nodes):
        token_id = int(node_token_ids[node_index].item())
        parent_index = int(parents[node_index].item())
        depth = int(node_depths[node_index].item())
        if parent_index < 0:
            path = (token_id,)
        else:
            path = nodes[parent_index].path_token_ids + (token_id,)
        nodes.append(
            DDTreeNode(
                index=node_index,
                token_id=token_id,
                parent_index=parent_index,
                depth=depth,
                path_log_weight=0.0,
                path_token_ids=path,
            )
        )
    return nodes


def build_frontier_candidates(
    recovery_token: int,
    branch_logits: torch.Tensor,
    nodes: list[DDTreeNode],
    frontier_count: int,
) -> list[DDTreeFrontierCandidate]:
    if branch_logits.dim() != 2:
        raise ValueError(f"branch_logits must be rank-2, got shape={tuple(branch_logits.shape)}")
    if frontier_count <= 0:
        raise ValueError("frontier_count must be > 0")

    log_probs = torch.log_softmax(branch_logits.float(), dim=-1)
    best_by_frontier: dict[tuple[int, ...], DDTreeFrontierCandidate] = {}

    states: list[tuple[int, tuple[int, ...], float]] = [(0, (), 0.0)]
    states.extend((node.depth, node.path_token_ids, node.path_log_weight) for node in nodes)
    for accepted_len_idx, path_token_ids, path_log_weight in states:
        bonus_token = int(log_probs[accepted_len_idx].argmax().item())
        bonus_log_prob = float(log_probs[accepted_len_idx, bonus_token].item())
        accepted_token_ids = (recovery_token,) + tuple(path_token_ids)
        frontier_token_ids = accepted_token_ids + (bonus_token,)
        candidate = DDTreeFrontierCandidate(
            frontier_token_ids=frontier_token_ids,
            accepted_token_ids=accepted_token_ids,
            accepted_len_idx=accepted_len_idx,
            recovery_token=bonus_token,
            score=path_log_weight + bonus_log_prob,
        )
        current = best_by_frontier.get(frontier_token_ids)
        if current is None or candidate.score > current.score:
            best_by_frontier[frontier_token_ids] = candidate

    ranked = sorted(best_by_frontier.values(), key=lambda item: item.score, reverse=True)
    return ranked[:frontier_count]


def walk_greedy_tree(
    recovery_token: int,
    nodes: list[DDTreeNode],
    next_token_fn: Callable[[tuple[int, ...]], int],
) -> DDTreeVerifyOutcome:
    children_by_parent: dict[int, list[DDTreeNode]] = {-1: []}
    for node in nodes:
        children_by_parent.setdefault(node.parent_index, []).append(node)
    for child_list in children_by_parent.values():
        child_list.sort(key=lambda node: node.index)

    accepted_suffix = [recovery_token]
    current_parent = -1
    matched_depth = 0
    visited_node_count = 0

    while True:
        prefix_tokens = tuple(accepted_suffix)
        next_token = int(next_token_fn(prefix_tokens))
        children = children_by_parent.get(current_parent, [])
        matched_child = None
        for child in children:
            if child.token_id == next_token:
                matched_child = child
                break
        if matched_child is None:
            return DDTreeVerifyOutcome(
                accepted_suffix=accepted_suffix,
                recovery_token=next_token,
                matched_depth=matched_depth,
                visited_node_count=visited_node_count,
            )
        accepted_suffix.append(next_token)
        current_parent = matched_child.index
        matched_depth = matched_child.depth
        visited_node_count += 1


def compile_verify_inputs(
    recovery_token: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    num_nodes: int,
    prefix_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_len = num_nodes + 1
    input_ids = torch.empty((q_len,), dtype=torch.int64, device=device)
    positions = torch.empty((q_len,), dtype=torch.int64, device=device)
    input_ids[0] = recovery_token
    positions[0] = prefix_len
    if num_nodes > 0:
        input_ids[1:q_len] = node_token_ids[:num_nodes].to(device=device, dtype=torch.int64, non_blocking=False)
        positions[1:q_len] = prefix_len + node_depths[:num_nodes].to(device=device, dtype=torch.int64, non_blocking=False)
    return input_ids, positions


def build_verify_mask(
    prefix_len: int,
    parents: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    q_len = num_nodes + 1
    total_len = prefix_len + q_len
    mask = torch.zeros((q_len, total_len), dtype=torch.bool, device=device)
    if prefix_len > 0:
        mask[:, :prefix_len] = True
    # Root token always attends to the full committed prefix and itself.
    mask[0, prefix_len] = True
    for node_idx in range(num_nodes):
        row_idx = node_idx + 1
        mask[row_idx, prefix_len] = True
        cur = node_idx
        while cur >= 0:
            mask[row_idx, prefix_len + cur + 1] = True
            cur = int(parents[cur].item())
    return mask.reshape(-1)


def walk_verified_tree(
    recovery_token: int,
    nodes: list[DDTreeNode],
    flat_logits: torch.Tensor,
) -> DDTreeVerifyTraversal:
    if flat_logits.dim() != 2:
        raise ValueError(f"flat_logits must be rank-2, got shape={tuple(flat_logits.shape)}")
    if flat_logits.shape[0] != len(nodes) + 1:
        raise ValueError(
            f"flat_logits rows must equal num_nodes + 1, got {flat_logits.shape[0]} rows for {len(nodes)} nodes"
        )

    children_by_parent: dict[int, list[DDTreeNode]] = {-1: []}
    for node in nodes:
        children_by_parent.setdefault(node.parent_index, []).append(node)
    for child_list in children_by_parent.values():
        child_list.sort(key=lambda node: node.index)

    accepted_suffix = [recovery_token]
    accepted_query_indices = [0]
    current_parent = -1
    matched_depth = 0
    visited_node_count = 0
    query_row = 0

    while True:
        next_token = int(flat_logits[query_row].argmax(dim=-1).item())
        children = children_by_parent.get(current_parent, [])
        matched_child = None
        for child in children:
            if child.token_id == next_token:
                matched_child = child
                break
        if matched_child is None:
            return DDTreeVerifyTraversal(
                accepted_suffix=accepted_suffix,
                recovery_token=next_token,
                matched_depth=matched_depth,
                visited_node_count=visited_node_count,
                accepted_query_indices=accepted_query_indices,
            )
        accepted_suffix.append(next_token)
        query_row = matched_child.index + 1
        accepted_query_indices.append(query_row)
        current_parent = matched_child.index
        matched_depth = matched_child.depth
        visited_node_count += 1
