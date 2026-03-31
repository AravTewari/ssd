from __future__ import annotations

from time import perf_counter

import torch
from transformers import AutoModel, AutoTokenizer

from ssd.config import Config
from ssd.engine.sequence import Sequence


class DreamDiffusionAdapter:
    def __init__(
        self,
        config: Config,
        target_tokenizer: AutoTokenizer,
        device: torch.device,
        metrics: dict | None = None,
    ):
        self.config = config
        self.device = device
        self.metrics = metrics

        dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.model = AutoModel.from_pretrained(
            config.draft,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device).eval()
        self.model_vocab_size = int(self.model.get_output_embeddings().weight.shape[0])

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.draft,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.target_tokenizer = target_tokenizer
        if self.target_tokenizer.pad_token_id is None and self.target_tokenizer.eos_token_id is not None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
        self.target_tokenizer.padding_side = "left"

        self._validate_compatibility()

    def _validate_compatibility(self):
        target_vocab = len(self.target_tokenizer)
        if target_vocab > self.model_vocab_size:
            raise ValueError(
                "dream_diffusion requires the target tokenizer ids to fit within the draft model vocab, "
                f"got target_size={target_vocab} draft_model_vocab={self.model_vocab_size}"
            )

        self._validate_special_token_alignment("eos")
        self._validate_special_token_alignment("pad")
        self._validate_special_token_alignment("bos")

        probes = [
            "Speculative decoding compatibility probe.",
            "The quick brown fox jumps over the lazy dog.",
            " 123 + 456 = 579",
        ]
        for probe in probes:
            target_ids = self.target_tokenizer.encode(probe, add_special_tokens=False)
            draft_ids = self.tokenizer.encode(probe, add_special_tokens=False)
            if target_ids != draft_ids:
                raise ValueError(
                    "dream_diffusion requires identical tokenization for target and draft tokenizers"
                )
            if target_ids and max(target_ids) >= self.model_vocab_size:
                raise ValueError(
                    "dream_diffusion found target token ids outside the draft model vocab during compatibility checks"
                )

    def _validate_special_token_alignment(self, token_name: str):
        token = getattr(self.target_tokenizer, f"{token_name}_token", None)
        token_id = getattr(self.target_tokenizer, f"{token_name}_token_id", None)
        if token is None or token_id is None:
            return
        draft_id = self.tokenizer.convert_tokens_to_ids(token)
        if draft_id != token_id:
            raise ValueError(
                f"dream_diffusion requires the target {token_name} token {token!r} to map to id {token_id} in the draft tokenizer, "
                f"got {draft_id}"
            )

    @staticmethod
    def _map_remasking_to_alg(remasking: str) -> str:
        mapping = {
            "origin": "origin",
            "maskgit_plus": "maskgit_plus",
            "topk_margin": "topk_margin",
            "entropy": "entropy",
            "low_confidence": "entropy",
        }
        if remasking not in mapping:
            raise ValueError(f"Unsupported dream_diffusion remasking mode: {remasking}")
        return mapping[remasking]

    @torch.inference_mode()
    def speculate(
        self,
        seqs: list[Sequence],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.target_tokenizer.pad(
            {"input_ids": [seq.token_ids for seq in seqs]},
            padding=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        captured_logits: list[torch.Tensor] = []

        def logits_hook(step, x, logits):
            captured_logits.append(logits[:, -lookahead:, :].detach().clone())
            return logits

        t0 = perf_counter()
        output = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=lookahead,
            return_dict_in_generate=True,
            steps=self.config.diffusion_steps,
            temperature=0.0,
            alg=self._map_remasking_to_alg(self.config.diffusion_remasking),
            alg_temp=0.0,
            generation_logits_hook_func=logits_hook,
        )
        if self.metrics is not None:
            self.metrics["diffusion_draft_step_times"].append(perf_counter() - t0)

        if not captured_logits:
            raise RuntimeError("dream_diffusion failed to capture logits from generation_logits_hook_func")

        logits_q = captured_logits[-1]
        if logits_q.ndim != 3 or logits_q.shape[:2] != (len(seqs), lookahead):
            raise RuntimeError(
                f"dream_diffusion captured malformed logits_q with shape {tuple(logits_q.shape)}"
            )

        sequences = output.sequences
        if sequences.ndim != 2 or sequences.shape[0] != len(seqs) or sequences.shape[1] < lookahead:
            raise RuntimeError(
                f"dream_diffusion returned malformed sequences with shape {tuple(sequences.shape)}"
            )
        draft_tokens = sequences[:, -lookahead:]
        return draft_tokens, logits_q
