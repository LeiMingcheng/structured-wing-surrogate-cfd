"""Hierarchical whole-case global FNO for fixed multi-block 3D grids."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from surrogate3d.models.base import BaseWholeCaseDirectModel
from surrogate3d.models.patching import PatchSize3D, patchify3d, unpatchify3d, validate_patch_size
from surrogate3d.models.spectral3d import SpectralBlock3d


def _block_shape_from_catalog(item: Mapping[str, Any]) -> tuple[int, int, int]:
    shape = item.get("cell_shape_ijk")
    if shape is None:
        raise KeyError("block_catalog items must include cell_shape_ijk")
    return tuple(int(v) for v in shape)


def _choose_attention_heads(width: int) -> int:
    for candidate in (8, 4, 2):
        if int(width) % candidate == 0 and int(width) >= candidate:
            return candidate
    return 1


def _coarse_grid_shape(patch_grid_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    ni, nj, nk = [int(v) for v in patch_grid_shape]
    return (
        min(2, ni),
        min(4, nj),
        min(2, nk),
    )


class StructuredBlockTokenizer(nn.Module):
    """Pool patch-grid features into a coarse token grid with spatial layout."""

    def __init__(self, width: int, condition_dim: int) -> None:
        super().__init__()
        self.width = int(width)
        self.condition_proj = nn.Linear(condition_dim, self.width)
        self.out = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.width * 2),
            nn.GELU(),
            nn.Linear(self.width * 2, self.width),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        coarse_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        pooled = F.adaptive_avg_pool3d(hidden, coarse_shape)
        tokens = pooled.permute(0, 2, 3, 4, 1).reshape(hidden.shape[0], -1, self.width)
        tokens = tokens + self.condition_proj(condition).unsqueeze(1)
        tokens = tokens + self.out(tokens)
        return tokens, coarse_shape


class PatchwiseGlobalReinjector(nn.Module):
    """Inject mixed global tokens back into each patch token via cross-attention."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.query_norm = nn.LayerNorm(self.width)
        self.kv_norm = nn.LayerNorm(self.width)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.width,
            num_heads=_choose_attention_heads(self.width),
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.width * 2),
            nn.GELU(),
            nn.Linear(self.width * 2, self.width),
        )

    def forward(self, hidden: torch.Tensor, global_tokens: torch.Tensor) -> torch.Tensor:
        batch, width, ni, nj, nk = hidden.shape
        queries = hidden.permute(0, 2, 3, 4, 1).reshape(batch, ni * nj * nk, width)
        queries = self.query_norm(queries)
        global_tokens = self.kv_norm(global_tokens)
        mixed, _ = self.attn(queries, global_tokens, global_tokens, need_weights=False)
        mixed = mixed + self.out(mixed)
        mixed = mixed.reshape(batch, ni, nj, nk, width).permute(0, 4, 1, 2, 3).contiguous()
        return hidden + mixed


class HierarchicalGlobalFNO3D(BaseWholeCaseDirectModel):
    """Whole-case model with structured global tokens and patch-wise reinjection."""

    def __init__(
        self,
        *,
        block_catalog: Sequence[Mapping[str, Any]],
        geometry_dim: int = 46,
        flow_condition_dim: int = 3,
        coord_channels: int = 6,
        out_channels: int = 6,
        patch_size: PatchSize3D = (8, 4, 8),
        width: int = 32,
        modes: tuple[int, int, int] = (8, 4, 6),
        n_layers: int = 4,
        condition_dim: int = 128,
        global_layers: int = 2,
        input_state_channels: int = 0,
        post_global_layers: int = 2,
    ) -> None:
        super().__init__(
            geometry_dim=geometry_dim,
            flow_condition_dim=flow_condition_dim,
            coord_channels=coord_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            width=width,
            modes=modes,
            n_layers=n_layers,
            condition_dim=condition_dim,
            global_layers=global_layers,
            input_state_channels=input_state_channels,
            post_global_layers=post_global_layers,
        )
        self.block_catalog = sorted(
            [dict(item) for item in block_catalog],
            key=lambda item: int(item["block_index"]),
        )
        self.block_indices = [int(item["block_index"]) for item in self.block_catalog]
        self.patch_size = tuple(int(v) for v in patch_size)
        self.width = int(width)
        self.condition_dim = int(condition_dim)
        self.input_state_channels = int(input_state_channels)
        self.input_channels = int(coord_channels) + self.input_state_channels
        self.post_global_layers = int(post_global_layers)

        self.condition_encoder = nn.Sequential(
            nn.Linear(geometry_dim + flow_condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.condition_to_width = nn.Linear(condition_dim, self.width)

        self.stems = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.tokenizers = nn.ModuleDict()
        self.reinjectors = nn.ModuleDict()
        self.post_global_blocks = nn.ModuleDict()
        self.block_patch_grid_shapes: dict[int, tuple[int, int, int]] = {}
        self.block_token_grid_shapes: dict[int, tuple[int, int, int]] = {}
        self.block_token_slices: dict[int, tuple[int, int]] = {}

        token_cursor = 0
        for item in self.block_catalog:
            block_index = int(item["block_index"])
            shape = _block_shape_from_catalog(item)
            validate_patch_size(shape, self.patch_size)
            patch_grid_shape = tuple(int(a // b) for a, b in zip(shape, self.patch_size))
            token_grid_shape = _coarse_grid_shape(patch_grid_shape)
            n_tokens = int(token_grid_shape[0] * token_grid_shape[1] * token_grid_shape[2])
            self.block_patch_grid_shapes[block_index] = patch_grid_shape
            self.block_token_grid_shapes[block_index] = token_grid_shape
            self.block_token_slices[block_index] = (token_cursor, token_cursor + n_tokens)
            token_cursor += n_tokens

            patch_dim_in = self.input_channels
            patch_dim_in *= self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
            patch_dim_out = out_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
            key = str(block_index)
            self.stems[key] = nn.Linear(patch_dim_in, self.width)
            self.tokenizers[key] = StructuredBlockTokenizer(self.width, condition_dim)
            self.reinjectors[key] = PatchwiseGlobalReinjector(self.width)
            self.post_global_blocks[key] = nn.ModuleList(
                [SpectralBlock3d(self.width, modes, condition_dim=condition_dim) for _ in range(self.post_global_layers)]
            )
            self.heads[key] = nn.Sequential(
                nn.LayerNorm(self.width),
                nn.Linear(self.width, self.width * 2),
                nn.GELU(),
                nn.Linear(self.width * 2, patch_dim_out),
            )

        self.local_layers = nn.ModuleList(
            [SpectralBlock3d(self.width, modes, condition_dim=condition_dim) for _ in range(int(n_layers))]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.width,
            nhead=_choose_attention_heads(self.width),
            dim_feedforward=self.width * 4,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.global_mixer = nn.TransformerEncoder(encoder_layer, num_layers=int(global_layers))
        self.block_embeddings = nn.Embedding(len(self.block_indices), self.width)
        self.local_token_pos_embeddings = nn.ParameterDict()
        for item in self.block_catalog:
            block_index = int(item["block_index"])
            n_tokens = int(torch.tensor(self.block_token_grid_shapes[block_index]).prod().item())
            self.local_token_pos_embeddings[str(block_index)] = nn.Parameter(
                torch.randn(n_tokens, self.width) * (self.width ** -0.5)
            )

    def _condition(self, geometry: torch.Tensor, flow_conditions: torch.Tensor) -> torch.Tensor:
        flow_processed = self.preprocess_flow_conditions(flow_conditions)
        return self.condition_encoder(torch.cat([geometry, flow_processed], dim=1))

    def _stem_block(
        self,
        block_index: int,
        block: Mapping[str, torch.Tensor],
        condition: torch.Tensor,
        initial_field: Optional[Mapping[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        coords = block["coords"]
        inputs = [coords]
        if self.input_state_channels > 0:
            if initial_field is None or block_index not in initial_field:
                raise ValueError("initial_field dictionary is required when input_state_channels > 0")
            inputs.append(initial_field[block_index])
        x = torch.cat(inputs, dim=1)
        patches = patchify3d(x, self.patch_size)
        hidden = self.stems[str(block_index)](patches)
        hidden = hidden.permute(0, 4, 1, 2, 3).contiguous()
        hidden = hidden + self.condition_to_width(condition)[..., None, None, None]
        return hidden

    def forward(
        self,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        blocks: Mapping[int, Mapping[str, torch.Tensor]],
        initial_field: Optional[Mapping[int, torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        condition = self._condition(geometry, flow_conditions)

        hidden_by_block: Dict[int, torch.Tensor] = {}
        block_tokens = []
        for pos, block_index in enumerate(self.block_indices):
            if block_index not in blocks:
                raise KeyError(f"Missing block {block_index} in whole-case batch")
            hidden = self._stem_block(block_index, blocks[block_index], condition, initial_field)
            for layer in self.local_layers:
                hidden = layer(hidden, condition)
            token_grid_shape = self.block_token_grid_shapes[block_index]
            tokens, _ = self.tokenizers[str(block_index)](hidden, condition, token_grid_shape)
            tokens = tokens + self.block_embeddings.weight[pos].view(1, 1, self.width)
            tokens = tokens + self.local_token_pos_embeddings[str(block_index)].unsqueeze(0)
            hidden_by_block[block_index] = hidden
            block_tokens.append(tokens)

        mixed_tokens = self.global_mixer(torch.cat(block_tokens, dim=1))

        outputs: Dict[int, torch.Tensor] = {}
        for block_index in self.block_indices:
            start, stop = self.block_token_slices[block_index]
            block_global_tokens = mixed_tokens[:, start:stop, :]
            hidden = self.reinjectors[str(block_index)](hidden_by_block[block_index], block_global_tokens)
            for layer in self.post_global_blocks[str(block_index)]:
                hidden = layer(hidden, condition)
            patches = hidden.permute(0, 2, 3, 4, 1).contiguous()
            pred_patches = self.heads[str(block_index)](patches)
            outputs[block_index] = unpatchify3d(
                pred_patches,
                channels=self.out_channels,
                patch_size=self.patch_size,
            )
        return outputs

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "block_indices": list(self.block_indices),
                "patch_size": list(self.patch_size),
                "width": self.width,
                "condition_dim": self.condition_dim,
                "post_global_layers": self.post_global_layers,
                "block_patch_grid_shapes": {
                    str(k): list(v) for k, v in self.block_patch_grid_shapes.items()
                },
                "block_token_grid_shapes": {
                    str(k): list(v) for k, v in self.block_token_grid_shapes.items()
                },
            }
        )
        return info


__all__ = [
    "HierarchicalGlobalFNO3D",
    "PatchwiseGlobalReinjector",
    "StructuredBlockTokenizer",
]
