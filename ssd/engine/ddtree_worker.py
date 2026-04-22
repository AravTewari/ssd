import os

import ssd.paths  # noqa: F401
import torch
import torch.distributed as dist

from ssd.config import Config
from ssd.engine.ddtree_runtime import DDTreeRuntime
from ssd.engine.helpers.ddtree import DDTreeEntry
from ssd.engine.helpers.speculate_types import DDTreeDiagnosticBatch


class DDTreeCommand:
    RESET = 0
    PREFILL = 1
    SPECULATE = 2
    EXIT = 3


class DDTreeWorkerHandle:
    def __init__(
        self,
        config: Config,
        device: torch.device,
        metrics: dict | None = None,
    ):
        self.config = config
        self.device = device
        self.worker_rank = config.num_gpus - 1
        self.tree_budget = config.ddtree_tree_budget
        self.feature_dim = config.dflash_target_feature_dim
        self.dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.metrics = metrics

    def _send_command(self, cmd: int) -> None:
        dist.send(torch.tensor([cmd], dtype=torch.int64, device=self.device), dst=self.worker_rank)

    def _send_int64(self, values: list[int] | torch.Tensor) -> None:
        if isinstance(values, torch.Tensor):
            tensor = values.to(device=self.device, dtype=torch.int64, non_blocking=False)
        else:
            tensor = torch.tensor(values, dtype=torch.int64, device=self.device)
        dist.send(tensor, dst=self.worker_rank)

    def _send_feature_list(self, feature_list: list[torch.Tensor | None]) -> None:
        lengths = [0 if feat is None else int(feat.shape[0]) for feat in feature_list]
        self._send_int64(lengths)
        total = sum(lengths)
        if total == 0:
            return
        flat = torch.cat(
            [
                feat.to(device=self.device, dtype=self.dtype, non_blocking=False)
                for feat in feature_list
                if feat is not None and feat.numel() > 0
            ],
            dim=0,
        )
        dist.send(flat, dst=self.worker_rank)

    def reset_all(self) -> None:
        self._send_command(DDTreeCommand.RESET)

    def prefill(self, seq_ids: list[int], prompt_target_features: list[torch.Tensor]) -> None:
        self._send_command(DDTreeCommand.PREFILL)
        self._send_int64([len(seq_ids)])
        self._send_int64(seq_ids)
        self._send_feature_list(prompt_target_features)

    def speculate(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        recovery_tokens: list[int],
        temperatures: list[float],
        target_features: list[torch.Tensor | None],
    ) -> tuple[list[DDTreeEntry], DDTreeDiagnosticBatch]:
        batch_size = len(seq_ids)
        self._send_command(DDTreeCommand.SPECULATE)
        self._send_int64([batch_size])
        self._send_int64(seq_ids)
        self._send_int64(frontier_versions)
        self._send_int64(recovery_tokens)
        dist.send(torch.tensor(temperatures, dtype=torch.float32, device=self.device), dst=self.worker_rank)
        self._send_feature_list(target_features)

        num_nodes = torch.empty((batch_size,), dtype=torch.int64, device=self.device)
        node_token_ids = torch.empty((batch_size, self.tree_budget), dtype=torch.int64, device=self.device)
        node_depths = torch.empty((batch_size, self.tree_budget), dtype=torch.int64, device=self.device)
        parents = torch.empty((batch_size, self.tree_budget), dtype=torch.int64, device=self.device)
        diag = torch.empty((3,), dtype=torch.float32, device=self.device)
        dist.recv(num_nodes, src=self.worker_rank)
        dist.recv(node_token_ids, src=self.worker_rank)
        dist.recv(node_depths, src=self.worker_rank)
        dist.recv(parents, src=self.worker_rank)
        dist.recv(diag, src=self.worker_rank)

        entries = []
        for row_idx, recovery_token in enumerate(recovery_tokens):
            entries.append(
                DDTreeEntry(
                    recovery_token=int(recovery_token),
                    node_token_ids=node_token_ids[row_idx].clone(),
                    node_depths=node_depths[row_idx].clone(),
                    parents=parents[row_idx].clone(),
                    num_nodes=int(num_nodes[row_idx].item()),
                    draft_tokens=torch.empty(0, dtype=torch.int64, device=self.device),
                    logits_q=torch.empty(0, dtype=self.dtype, device=self.device),
                    branch_logits=torch.empty(0, dtype=self.dtype, device=self.device),
                )
            )
        return entries, DDTreeDiagnosticBatch(
            service_dflash_s=float(diag[0].item()),
            service_predictor_s=float(diag[1].item()),
            service_tree_build_s=float(diag[2].item()),
        )

    def close(self) -> None:
        if dist.is_available() and dist.is_initialized():
            try:
                self._send_command(DDTreeCommand.EXIT)
            except Exception:
                pass


class DDTreeWorker:
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16

        torch.cuda.set_device(rank)
        default_port = int(os.environ.get("SSD_DIST_PORT", "1223"))
        dist.init_process_group(
            "nccl",
            f"tcp://localhost:{default_port}",
            world_size=config.num_gpus,
            rank=rank,
            device_id=self.device,
        )
        dist.new_group(ranks=[0])
        self.runtime = DDTreeRuntime(
            config=config,
            device=self.device,
            predictor_path=(config.dflash_predictor if config.ddtree_context_mode == "predicted" else None),
            metrics=None,
        )
        self.loop()

    def _recv_int64(self, shape: tuple[int, ...]) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=torch.int64, device=self.device)
        dist.recv(tensor, src=0)
        return tensor

    def _recv_feature_list(self, batch_size: int) -> list[torch.Tensor | None]:
        lengths = self._recv_int64((batch_size,))
        total = int(lengths.sum().item())
        flat = None
        if total > 0:
            flat = torch.empty((total, self.config.dflash_target_feature_dim), dtype=self.dtype, device=self.device)
            dist.recv(flat, src=0)
        feature_list = []
        offset = 0
        for length in lengths.tolist():
            if length <= 0:
                feature_list.append(None)
            else:
                feature_list.append(flat[offset:offset + length].clone())
                offset += length
        return feature_list

    def _reset(self) -> None:
        self.runtime.reset_states()

    def _prefill(self) -> None:
        batch_size = int(self._recv_int64((1,)).item())
        seq_ids = self._recv_int64((batch_size,)).tolist()
        prompt_target_features = self._recv_feature_list(batch_size)
        self.runtime.prefill_exact_context(seq_ids, prompt_target_features, frontier_version=0)

    @torch.inference_mode()
    def _speculate(self) -> None:
        batch_size = int(self._recv_int64((1,)).item())
        seq_ids = self._recv_int64((batch_size,)).tolist()
        frontier_versions = self._recv_int64((batch_size,)).tolist()
        recovery_tokens = self._recv_int64((batch_size,)).tolist()
        temperatures = torch.empty((batch_size,), dtype=torch.float32, device=self.device)
        dist.recv(temperatures, src=0)
        exact_feature_updates = self._recv_feature_list(batch_size)

        self.runtime.commit_exact_context(seq_ids, frontier_versions, exact_feature_updates)
        batch = self.runtime.build_exact_tree_batch(
            seq_ids=seq_ids,
            recovery_tokens=torch.tensor(recovery_tokens, dtype=torch.int64, device=self.device),
            temperatures=temperatures,
            return_predicted_features=False,
        )
        num_nodes = torch.tensor([entry.num_nodes for entry in batch.entries], dtype=torch.int64, device=self.device)
        node_token_ids = torch.stack([entry.node_token_ids for entry in batch.entries], dim=0)
        node_depths = torch.stack([entry.node_depths for entry in batch.entries], dim=0)
        parents = torch.stack([entry.parents for entry in batch.entries], dim=0)
        diag = torch.tensor(
            [batch.dflash_time_s, batch.predictor_time_s, batch.tree_build_time_s],
            dtype=torch.float32,
            device=self.device,
        )
        dist.send(num_nodes, dst=0)
        dist.send(node_token_ids, dst=0)
        dist.send(node_depths, dst=0)
        dist.send(parents, dst=0)
        dist.send(diag, dst=0)

    def loop(self) -> None:
        while True:
            cmd = int(self._recv_int64((1,)).item())
            if cmd == DDTreeCommand.RESET:
                self._reset()
            elif cmd == DDTreeCommand.PREFILL:
                self._prefill()
            elif cmd == DDTreeCommand.SPECULATE:
                self._speculate()
            elif cmd == DDTreeCommand.EXIT:
                break
            else:
                raise RuntimeError(f"Unknown DDTree command: {cmd}")
