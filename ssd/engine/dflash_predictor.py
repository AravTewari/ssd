import json
import os
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class DFlashPredictorConfig:
    hidden_size: int
    target_feature_dim: int
    block_size: int
    position_dim: int
    mlp_hidden_size: int


class DFlashFeaturePredictor(nn.Module):
    def __init__(self, config: DFlashPredictorConfig):
        super().__init__()
        self.config = config
        self.position_embedding = nn.Embedding(config.block_size, config.position_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size + config.position_dim, config.mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(config.mlp_hidden_size, config.target_feature_dim),
        )

    def forward(self, block_hidden: torch.Tensor) -> torch.Tensor:
        if block_hidden.dim() != 3:
            raise ValueError(f"Expected block_hidden with shape [B, K+1, H], got {tuple(block_hidden.shape)}")
        batch_size, block_len, _ = block_hidden.shape
        if block_len > self.config.block_size:
            raise ValueError(
                f"Block hidden length {block_len} exceeds predictor block_size {self.config.block_size}"
            )
        pos_ids = torch.arange(block_len, device=block_hidden.device)
        pos_emb = self.position_embedding(pos_ids).unsqueeze(0).expand(batch_size, -1, -1)
        inputs = torch.cat([block_hidden, pos_emb], dim=-1)
        return self.mlp(inputs)

    @staticmethod
    def loss(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mse = torch.mean((predicted - target) ** 2)
        cosine = 1.0 - torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        loss = 0.5 * mse + 0.5 * cosine
        return loss, {
            "loss": loss.detach(),
            "mse": mse.detach(),
            "cosine": cosine.detach(),
        }

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, sort_keys=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "DFlashFeaturePredictor":
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            cfg = DFlashPredictorConfig(**json.load(f))
        model = cls(cfg)
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        return model

