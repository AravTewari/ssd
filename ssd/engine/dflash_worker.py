import os
from time import perf_counter

import ssd.paths  # noqa: F401
import torch
import torch.distributed as dist

from ssd.config import Config
from ssd.engine.dflash_runtime import DFlashRuntime


class DFlashCommand:
    RESET = 0
    PREFILL = 1
    SPECULATE = 2
    EXIT = 3


class DFlashWorkerHandle:
    def __init__(
        self,
        config: Config,
        device: torch.device,
        metrics: dict | None = None,
    ):
        self.config = config
        self.device = device
        self.worker_rank = config.num_gpus - 1
        self.lookahead = config.speculate_k
        self.feature_dim = config.dflash_target_feature_dim
        self.vocab_size = config.hf_config.vocab_size
        self.hidden_size = config.draft_hf_config.hidden_size
        self.metrics = metrics
        self.dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.return_hidden = config.dflash_trace_hidden

    def _send_command(self, cmd: int) -> None:
        dist.send(torch.tensor([cmd], dtype=torch.int64, device=self.device), dst=self.worker_rank)

    def _send_int64(self, values: list[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            tensor = values.to(device=self.device, dtype=torch.int64, non_blocking=False)
        else:
            tensor = torch.tensor(values, dtype=torch.int64, device=self.device)
        dist.send(tensor, dst=self.worker_rank)
        return tensor

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
        self._send_command(DFlashCommand.RESET)

    def prefill(self, seq_ids: list[int], prompt_target_features: list[torch.Tensor]) -> None:
        self._send_command(DFlashCommand.PREFILL)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        t0 = perf_counter()
        self._send_command(DFlashCommand.SPECULATE)
        self._send_int64([len(seq_ids)])
        self._send_int64(seq_ids)
        self._send_int64(frontier_versions)
        self._send_int64(recovery_tokens)
        temps = torch.tensor(temperatures, dtype=torch.float32, device=self.device)
        dist.send(temps, dst=self.worker_rank)
        self._send_feature_list(target_features)

        draft_tokens = torch.empty(
            (len(seq_ids), self.lookahead),
            dtype=torch.int64,
            device=self.device,
        )
        logits_q = torch.empty(
            (len(seq_ids), self.lookahead, self.vocab_size),
            dtype=self.dtype,
            device=self.device,
        )
        block_hidden = None
        if self.return_hidden:
            block_hidden = torch.empty(
                (len(seq_ids), self.lookahead + 1, self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
        dist.recv(draft_tokens, src=self.worker_rank)
        dist.recv(logits_q, src=self.worker_rank)
        if block_hidden is not None:
            dist.recv(block_hidden, src=self.worker_rank)
        if self.metrics is not None:
            self.metrics["dflash_draft_step_times"].append(perf_counter() - t0)
        return draft_tokens, logits_q, block_hidden

    def close(self) -> None:
        if dist.is_available() and dist.is_initialized():
            try:
                self._send_command(DFlashCommand.EXIT)
            except Exception:
                pass


class DFlashWorker:
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.return_hidden = config.dflash_trace_hidden

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

        self.runtime = DFlashRuntime(
            config=config,
            device=self.device,
            predictor_path=None,
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
        outputs = self.runtime.generate_block(
            seq_ids=seq_ids,
            recovery_tokens=torch.tensor(recovery_tokens, dtype=torch.int64, device=self.device),
            temperatures=temperatures,
            return_predicted_features=False,
        )

        dist.send(outputs.draft_tokens.contiguous(), dst=0)
        dist.send(outputs.logits_q.contiguous(), dst=0)
        if self.return_hidden:
            dist.send(outputs.block_hidden.contiguous(), dst=0)

    def loop(self) -> None:
        while True:
            cmd = int(self._recv_int64((1,)).item())
            if cmd == DFlashCommand.RESET:
                self._reset()
            elif cmd == DFlashCommand.PREFILL:
                self._prefill()
            elif cmd == DFlashCommand.SPECULATE:
                self._speculate()
            elif cmd == DFlashCommand.EXIT:
                break
            else:
                raise RuntimeError(f"Unknown DFlash command: {cmd}")
