from dataclasses import dataclass
import torch
from ssd.engine.sequence import Sequence
from abc import ABC, abstractmethod
from ssd.engine.helpers.ddtree import DDTreeEntry


@dataclass
class DDTreeDiagnosticBatch:
    cache_lookup_s: float = 0.0
    service_dflash_s: float = 0.0
    service_predictor_s: float = 0.0
    service_tree_build_s: float = 0.0
    background_dflash_s: float = 0.0
    background_predictor_s: float = 0.0
    background_tree_build_s: float = 0.0
    worker_total_s: float = 0.0
    speculate_wait_s: float = 0.0
    post_verify_wait_s: float = 0.0
    fallback_used: list[bool] | None = None
    actual_frontier_rank: list[int | None] | None = None
    frontier_candidate_count: list[int] | None = None

    @property
    def total_dflash_s(self) -> float:
        return self.service_dflash_s + self.background_dflash_s

    @property
    def total_predictor_s(self) -> float:
        return self.service_predictor_s + self.background_predictor_s

    @property
    def total_tree_build_s(self) -> float:
        return self.service_tree_build_s + self.background_tree_build_s

    @property
    def total_transport_s(self) -> float:
        return max(self.speculate_wait_s + self.post_verify_wait_s - self.worker_total_s, 0.0)


@dataclass
class DFlashDiagnosticBatch:
    cache_lookup_s: float = 0.0
    service_dflash_s: float = 0.0
    service_predictor_s: float = 0.0
    background_dflash_s: float = 0.0
    background_predictor_s: float = 0.0
    worker_total_s: float = 0.0
    speculate_wait_s: float = 0.0
    post_verify_wait_s: float = 0.0
    fallback_used: list[bool] | None = None
    true_branch_rank: list[int | None] | None = None
    num_branches_generated: list[int] | None = None
    actual_accept_supported: list[bool] | None = None
    actual_recovery_rank_given_accept: list[int | None] | None = None
    joint_branch_supported: list[bool] | None = None
    recovery_entropy_at_actual_accept: list[float | None] | None = None
    recovery_top1_margin_at_actual_accept: list[float | None] | None = None

    @property
    def total_dflash_s(self) -> float:
        return self.service_dflash_s + self.background_dflash_s

    @property
    def total_predictor_s(self) -> float:
        return self.service_predictor_s + self.background_predictor_s

    @property
    def total_transport_s(self) -> float:
        return max(self.speculate_wait_s + self.post_verify_wait_s - self.worker_total_s, 0.0)


@dataclass
class SpeculateResult:
    speculations: torch.Tensor
    logits_q: torch.Tensor
    cache_hits: torch.Tensor | None = None
    dflash_block_hidden: torch.Tensor | None = None
    dflash_diag: DFlashDiagnosticBatch | None = None
    ddtree_diag: DDTreeDiagnosticBatch | None = None
    ddtree_entries: list[DDTreeEntry] | None = None


@dataclass
class VerifyResult:
    new_suffixes: list[list[int]]
    recovery_tokens: list[int]
    eagle_acts: torch.Tensor | None = None  # Is this a tensor?
    dflash_target_features: list[torch.Tensor] | None = None
    dflash_target_features_full: torch.Tensor | None = None
    target_verify_s: float | None = None
    ddtree_verified_node_counts: list[int] | None = None


class SpeculatorBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass

    @abstractmethod
    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass


class VerifierBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        pass

    @abstractmethod
    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        pass
