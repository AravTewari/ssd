import os
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT
from ssd.utils.misc import infer_model_family

@dataclass
class Config:
    model: str = DEFAULT_TARGET
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    draft_backend: str = "ar"
    speculate_k: int = 1
    draft_async: bool = False
    ar_branch_cache: str = "on"
    ar_branch_key_mode: str = "normal"
    dflash_predictor: str | None = None
    dflash_trace_hidden: bool = False
    dflash_context_mode: str = "predicted"
    dflash_branch_cache: str = "on"
    dflash_branch_key_mode: str = "normal"
    dflash_enable_diagnostics: bool = False
    ddtree_tree_budget: int = 16
    ddtree_frontier_count: int = 1
    ddtree_context_mode: str = "predicted"
    ddtree_cache: str = "on"
    ddtree_frontier_mode: str = "surrogate"
    ddtree_enable_diagnostics: bool = False
    diffusion_steps: int = 128
    diffusion_remasking: str = "low_confidence"
    diffusion_mask_id: int = 126336
    dflash_block_size: int | None = None
    dflash_mask_token_id: int | None = None
    dflash_target_layer_ids: list[int] | None = None
    dflash_target_feature_dim: int | None = None
    
    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False 

    # eagle3
    use_eagle: bool = False 
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # Debugging
    verbose: bool = False 
    debug_mode: bool = False 
    max_steps: int | None = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    @staticmethod
    def _resolve_max_model_len(hf_config: AutoConfig, fallback: int) -> int:
        for attr in (
            "max_position_embeddings",
            "max_sequence_length",
            "model_max_length",
            "seq_len",
            "max_seq_len",
            "n_positions",
        ):
            value = getattr(hf_config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return fallback

    @staticmethod
    def _build_default_dflash_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
        if num_draft_layers <= 0:
            raise ValueError("dflash requires num_draft_layers > 0")
        if num_draft_layers == 1:
            return [num_target_layers // 2]
        start = 1
        end = num_target_layers - 3
        span = end - start
        return [
            int(round(start + (i * span) / (num_draft_layers - 1)))
            for i in range(num_draft_layers)
        ]

    @staticmethod
    def _validate_tokenizer_alignment(
        target_tokenizer: AutoTokenizer,
        draft_tokenizer: AutoTokenizer,
        draft_vocab_size: int,
        backend_name: str,
    ) -> None:
        for token_name in ("eos", "pad", "bos"):
            token = getattr(target_tokenizer, f"{token_name}_token", None)
            token_id = getattr(target_tokenizer, f"{token_name}_token_id", None)
            if token is None or token_id is None:
                continue
            draft_id = draft_tokenizer.convert_tokens_to_ids(token)
            if draft_id != token_id:
                raise ValueError(
                    f"{backend_name} requires the target {token_name} token {token!r} to map to id {token_id} in the draft tokenizer, "
                    f"got {draft_id}"
                )

        probes = [
            "Speculative decoding compatibility probe.",
            "The quick brown fox jumps over the lazy dog.",
            " 123 + 456 = 579",
        ]
        for probe in probes:
            target_ids = target_tokenizer.encode(probe, add_special_tokens=False)
            draft_ids = draft_tokenizer.encode(probe, add_special_tokens=False)
            if target_ids != draft_ids:
                raise ValueError(
                    f"{backend_name} requires identical tokenization for target and draft tokenizers"
                )
            if target_ids and max(target_ids) >= draft_vocab_size:
                raise ValueError(
                    f"{backend_name} found target token ids outside the draft vocab during compatibility checks"
                )

    def __post_init__(self):
        model = self.model 
        assert os.path.isdir(model)

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        assert self.draft_backend in {
            "ar",
            "llada_diffusion",
            "dream_diffusion",
            "dflash",
            "dflash_ssd",
            "ddtree",
            "ddtree_ssd",
        }, (
            f"Unsupported draft_backend={self.draft_backend}"
        )
        if self.draft_backend in {
            "llada_diffusion",
            "dream_diffusion",
            "dflash",
            "dflash_ssd",
            "ddtree",
            "ddtree_ssd",
        }:
            assert self.speculate, f"{self.draft_backend} requires speculate=True"
        self.hf_config = AutoConfig.from_pretrained(model)
        self.max_model_len = min(
            self.max_model_len, self._resolve_max_model_len(self.hf_config, self.max_model_len)) 
        if self.speculate: 
            draft = self.draft
            self.draft_hf_config = AutoConfig.from_pretrained(
                draft,
                trust_remote_code=(
                    self.draft_backend in {
                        "llada_diffusion",
                        "dream_diffusion",
                        "dflash",
                        "dflash_ssd",
                        "ddtree",
                        "ddtree_ssd",
                    }
                ),
            )
            self.max_model_len = min(
                self.max_model_len, self._resolve_max_model_len(self.draft_hf_config, self.max_model_len))
            if self.draft_backend == "llada_diffusion":
                assert not self.draft_async, "llada_diffusion only supports synchronous speculation"
                assert infer_model_family(self.model) == "qwen", (
                    "llada_diffusion currently only supports Qwen targets"
                )
                assert self.diffusion_steps > 0, "diffusion_steps must be > 0"
                assert self.diffusion_remasking == "low_confidence", (
                    "llada_diffusion v1 only supports low_confidence remasking"
                )
            elif self.draft_backend == "dream_diffusion":
                assert not self.draft_async, "dream_diffusion only supports synchronous speculation"
                assert infer_model_family(self.model) == "qwen", (
                    "dream_diffusion currently only supports Qwen targets"
                )
                assert self.diffusion_steps > 0, "diffusion_steps must be > 0"
                if self.diffusion_remasking == "low_confidence":
                    self.diffusion_remasking = "entropy"
                assert self.diffusion_remasking in {
                    "origin", "maskgit_plus", "topk_margin", "entropy",
                }, (
                    "dream_diffusion supports remasking modes: origin, maskgit_plus, topk_margin, entropy"
                )
            elif self.draft_backend in {"dflash", "dflash_ssd", "ddtree", "ddtree_ssd"}:
                assert infer_model_family(self.model) == "qwen", (
                    f"{self.draft_backend} currently only supports Qwen targets"
                )
                assert getattr(self.hf_config, "model_type", None) == "qwen3", (
                    f"{self.draft_backend} currently only supports Qwen3 targets"
                )
                assert self.num_gpus == 2, f"{self.draft_backend} requires num_gpus=2 (target on gpu0, draft on gpu1)"
                if self.draft_backend == "dflash" and self.draft_async:
                    raise ValueError(
                        "dflash exact async/spec-spec is unsupported because next-step drafting requires fresh "
                        "target hidden features produced by the current verify step"
                    )
                if self.draft_backend == "dflash_ssd" and not self.draft_async:
                    raise ValueError("dflash_ssd requires draft_async=True")
                if self.draft_backend == "ddtree" and self.draft_async:
                    raise ValueError("ddtree requires draft_async=False")
                if self.draft_backend == "ddtree_ssd" and not self.draft_async:
                    raise ValueError("ddtree_ssd requires draft_async=True")
                if self.kvcache_block_size != 128:
                    print(
                        f"[Config] Overriding {self.draft_backend} kvcache_block_size: {self.kvcache_block_size} -> 128",
                        flush=True,
                    )
                    self.kvcache_block_size = 128
                if self.tokenizer_path is not None and os.path.realpath(self.tokenizer_path) != os.path.realpath(self.model):
                    raise ValueError(
                        f"{self.draft_backend} does not support overriding tokenizer_path; use the target model tokenizer directly"
                    )
                self.enforce_eager = True
                if self.draft_backend == "dflash_ssd":
                    if self.dflash_predictor is None:
                        raise ValueError("dflash_ssd requires dflash_predictor=<checkpoint_dir>")
                    if not os.path.exists(self.dflash_predictor):
                        raise FileNotFoundError(f"dflash_ssd predictor checkpoint not found: {self.dflash_predictor}")
                    if self.dflash_context_mode not in {"predicted", "exact"}:
                        raise ValueError(
                            "dflash_ssd requires dflash_context_mode in {'predicted', 'exact'}"
                        )
                    if self.dflash_branch_cache not in {"on", "off"}:
                        raise ValueError(
                            "dflash_ssd requires dflash_branch_cache in {'on', 'off'}"
                        )
                    if self.dflash_branch_key_mode not in {"normal", "oracle"}:
                        raise ValueError(
                            "dflash_ssd requires dflash_branch_key_mode in {'normal', 'oracle'}"
                        )
                    if self.dflash_context_mode == "exact":
                        if self.dflash_branch_cache == "off":
                            if self.dflash_branch_key_mode != "normal":
                                raise ValueError(
                                    "dflash_ssd exact context with branch cache off only supports "
                                    "dflash_branch_key_mode=normal"
                                )
                        elif self.dflash_branch_cache == "on":
                            if self.dflash_branch_key_mode != "oracle":
                                raise ValueError(
                                    "dflash_ssd exact context with branch cache on only supports "
                                    "dflash_branch_key_mode=oracle"
                                )
                        else:
                            raise ValueError(
                                "dflash_ssd exact context requires dflash_branch_cache in {'on', 'off'}"
                            )
                    if self.dflash_context_mode == "predicted" and self.dflash_branch_cache == "off":
                        if self.dflash_branch_key_mode != "oracle":
                            raise ValueError(
                                "dflash_ssd predicted context without branch cache requires dflash_branch_key_mode=oracle"
                            )
                    if self.dflash_branch_key_mode == "oracle" and self.dflash_context_mode not in {"predicted", "exact"}:
                        raise ValueError(
                            "dflash_ssd oracle branch-key mode requires dflash_context_mode in {'predicted', 'exact'}"
                        )
                if self.draft_backend == "ddtree_ssd":
                    if self.dflash_predictor is None and self.ddtree_context_mode == "predicted":
                        raise ValueError("ddtree_ssd predicted context requires dflash_predictor=<checkpoint_dir>")
                    if self.ddtree_context_mode not in {"predicted", "exact"}:
                        raise ValueError(
                            "ddtree_ssd requires ddtree_context_mode in {'predicted', 'exact'}"
                        )
                    if self.ddtree_cache not in {"on", "off"}:
                        raise ValueError("ddtree_ssd requires ddtree_cache in {'on', 'off'}")
                    if self.ddtree_frontier_mode not in {"surrogate", "oracle"}:
                        raise ValueError(
                            "ddtree_ssd requires ddtree_frontier_mode in {'surrogate', 'oracle'}"
                        )
                    if self.ddtree_tree_budget <= 0:
                        raise ValueError("ddtree_ssd requires ddtree_tree_budget > 0")
                    if self.ddtree_frontier_count <= 0:
                        raise ValueError("ddtree_ssd requires ddtree_frontier_count > 0")
                    if self.ddtree_cache == "off" and self.ddtree_frontier_mode != "oracle":
                        raise ValueError(
                            "ddtree_ssd with ddtree_cache=off only supports ddtree_frontier_mode=oracle"
                        )
                    if self.ddtree_context_mode == "exact":
                        if self.ddtree_cache == "on" and self.ddtree_frontier_mode != "oracle":
                            raise ValueError(
                                "ddtree_ssd exact context with cache on only supports ddtree_frontier_mode=oracle"
                            )
                    elif self.ddtree_cache == "off" and self.ddtree_frontier_mode != "oracle":
                        raise ValueError(
                            "ddtree_ssd without cache only supports ddtree_frontier_mode=oracle"
                        )
                if self.draft_backend == "ddtree":
                    if self.ddtree_tree_budget <= 0:
                        raise ValueError("ddtree requires ddtree_tree_budget > 0")
                    if self.ddtree_frontier_count <= 0:
                        raise ValueError("ddtree requires ddtree_frontier_count > 0")
                dflash_cfg = getattr(self.draft_hf_config, "dflash_config", None) or {}
                self.dflash_block_size = int(getattr(self.draft_hf_config, "block_size", 0) or 0)
                self.dflash_mask_token_id = dflash_cfg.get("mask_token_id", None)
                self.dflash_target_layer_ids = dflash_cfg.get("target_layer_ids", None)
                if self.dflash_target_layer_ids is None:
                    self.dflash_target_layer_ids = self._build_default_dflash_target_layer_ids(
                        num_target_layers=self.hf_config.num_hidden_layers,
                        num_draft_layers=self.draft_hf_config.num_hidden_layers,
                    )
                self.dflash_target_layer_ids = [int(x) for x in self.dflash_target_layer_ids]

                assert self.dflash_block_size and self.dflash_block_size > 1, (
                    "dflash requires draft config.block_size > 1"
                )
                assert self.dflash_mask_token_id is not None, (
                    "dflash requires draft config.dflash_config.mask_token_id"
                )
                assert self.dflash_target_layer_ids, (
                    "dflash requires non-empty draft config.dflash_config.target_layer_ids"
                )
                self.speculate_k = self.dflash_block_size - 1
                self.dflash_target_feature_dim = self.hf_config.hidden_size * len(self.dflash_target_layer_ids)

                target_tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
                try:
                    draft_tokenizer = AutoTokenizer.from_pretrained(
                        self.draft,
                        trust_remote_code=True,
                        use_fast=True,
                    )
                except Exception:
                    draft_tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
                if target_tokenizer.pad_token_id is None and target_tokenizer.eos_token_id is not None:
                    target_tokenizer.pad_token = target_tokenizer.eos_token
                if draft_tokenizer.pad_token_id is None and draft_tokenizer.eos_token_id is not None:
                    draft_tokenizer.pad_token = draft_tokenizer.eos_token
                self._validate_tokenizer_alignment(
                    target_tokenizer=target_tokenizer,
                    draft_tokenizer=draft_tokenizer,
                    draft_vocab_size=self.draft_hf_config.vocab_size,
                    backend_name=self.draft_backend,
                )
                if self.dflash_mask_token_id >= self.hf_config.vocab_size:
                    raise ValueError(
                        f"{self.draft_backend} mask_token_id={self.dflash_mask_token_id} is outside the target vocab size {self.hf_config.vocab_size}"
                    )
            elif self.draft_backend == "ar" and self.draft_async:
                if self.ar_branch_cache not in {"on", "off"}:
                    raise ValueError(
                        "async AR speculation requires ar_branch_cache in {'on', 'off'}"
                    )
                if self.ar_branch_key_mode not in {"normal", "oracle"}:
                    raise ValueError(
                        "async AR speculation requires ar_branch_key_mode in {'normal', 'oracle'}"
                    )
                if self.ar_branch_cache == "off" and self.ar_branch_key_mode != "normal":
                    raise ValueError(
                        "async AR speculation with ar_branch_cache=off only supports ar_branch_key_mode=normal"
                    )
                if self.ar_branch_key_mode == "oracle" and self.use_eagle:
                    raise ValueError(
                        "async AR oracle mode is not implemented for EAGLE"
                    )

            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list 
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"
                
        if self.use_eagle:
            if self.eagle_layers is None:
                L = self.hf_config.num_hidden_layers
                # self.eagle_layers = [3, L//2, L-3]
                self.eagle_layers = [2, L//2, L-3] # [2, 16, 29] outputs, ie. [3, L//2+1, L-2] inputs
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_rope_theta = getattr(self.hf_config, 'rope_theta', 500000.0)
                draft_rope_theta = getattr(self.draft_hf_config, 'rope_theta', 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    self.draft_hf_config.rope_theta = target_rope_theta
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft_hf_config.max_position_embeddings for RoPE.
                target_max_pos = getattr(self.hf_config, 'max_position_embeddings', 8192)
                draft_max_pos = getattr(self.draft_hf_config, 'max_position_embeddings', 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    self.draft_hf_config.max_position_embeddings = target_max_pos
        
        assert self.max_num_batched_tokens >= self.max_model_len
