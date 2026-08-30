"""Case-level dataset for fixed-topology multi-block 3D fields."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BlockRecord:
    block_index: int
    block_name: str
    block_signature: str
    shard_path: str
    local_index: int
    cell_shape_ijk: tuple[int, int, int]


class WholeCaseBlockDataset(Dataset):
    """Regroup structured block shards into whole-case samples.

    Each item contains all blocks for one ``sample_id / cond_idx`` case. Blocks
    keep their original tensor shapes and are keyed by integer ``block_index``.
    """

    def __init__(
        self,
        index_path: str | Path,
        *,
        num_cases: Optional[int] = None,
        cache_size_mb: int = 256,
        geometry_feature_names: Optional[Iterable[str]] = None,
        legacy_double_alignment_feature_names_by_family: Optional[
            Mapping[str, Iterable[str]]
        ] = None,
    ) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        self.root = self.index_path.parent
        self.cache_size_mb = int(cache_size_mb)
        self.index_df = pd.read_csv(self.index_path, low_memory=False)
        self.metadata = self._load_metadata()
        source_geometry_names = [str(name) for name in self.metadata.get("geometry_feature_names", [])]
        if source_geometry_names:
            requested_geometry_names = (
                [str(name) for name in geometry_feature_names]
                if geometry_feature_names is not None
                else source_geometry_names
            )
            missing_geometry_names = sorted(set(requested_geometry_names) - set(source_geometry_names))
            if missing_geometry_names:
                raise ValueError(
                    f"Dataset is missing requested geometry features: {missing_geometry_names}"
                )
            self.geometry_feature_names = requested_geometry_names
            self.geometry_feature_indices: Optional[list[int]] = [
                source_geometry_names.index(name) for name in requested_geometry_names
            ]
            self.geometry_dim = len(self.geometry_feature_names)
        else:
            metadata_geometry_dim = int(self.metadata.get("geometry_dim", 0))
            if geometry_feature_names is not None and len(list(geometry_feature_names)) != metadata_geometry_dim:
                raise ValueError(
                    "Cannot project an unnamed geometry vector to a different feature dimension"
                )
            self.geometry_feature_names = []
            self.geometry_feature_indices = None
            self.geometry_dim = metadata_geometry_dim
        self.legacy_double_alignment_indices_by_family: dict[
            str, list[tuple[int, int]]
        ] = {}
        for family, local_names_iter in (
            legacy_double_alignment_feature_names_by_family or {}
        ).items():
            local_names = [str(name) for name in local_names_iter]
            local_lookup = {name: idx for idx, name in enumerate(local_names)}
            self.legacy_double_alignment_indices_by_family[str(family)] = [
                (target_idx, local_lookup[name])
                for target_idx, name in enumerate(self.geometry_feature_names)
                if name in local_lookup
            ]
        self.block_catalog = self._load_block_catalog()
        self.block_indices = [int(item["block_index"]) for item in self.block_catalog]
        self._h5_handles: dict[str, h5py.File] = {}
        self._worker_id: Optional[int] = None

        case_groups: List[dict[str, Any]] = []
        grouped = self.index_df.groupby(["sample_id", "cond_idx"], sort=True)
        for (sample_id, cond_idx), frame in grouped:
            ordered = frame.sort_values("block_index")
            present = [int(v) for v in ordered["block_index"].tolist()]
            if present != self.block_indices:
                raise ValueError(
                    f"Case {sample_id}/cond_{cond_idx} has blocks {present}, expected {self.block_indices}"
                )
            first = ordered.iloc[0]
            case = {
                "sample_id": str(sample_id),
                "cond_idx": int(cond_idx),
                "family": str(first.get("family", "")),
                "band": str(first.get("band", "")),
                "cgns_basename": str(first.get("cgns_basename", "")),
                "source_case_dir": str(first.get("source_case_dir", "")),
                "flow": [
                    float(first["mach"]),
                    float(first["aoa"]),
                    float(first["reynolds"]),
                ],
                "rows": [
                    BlockRecord(
                        block_index=int(row.block_index),
                        block_name=str(row.block_name),
                        block_signature=str(row.block_signature),
                        shard_path=str(row.shard_path),
                        local_index=int(row.local_index),
                        cell_shape_ijk=(int(row.ni), int(row.nj), int(row.nk)),
                    )
                    for row in ordered.itertuples(index=False)
                ],
            }
            if "geometry_id" in ordered.columns:
                case["geometry_id"] = str(first["geometry_id"])
                case["subtype"] = str(first.get("subtype", ""))
            if "tier3_variant" in ordered.columns:
                case["tier3_variant"] = int(first["tier3_variant"])
            case_groups.append(case)

        if num_cases is not None:
            case_groups = case_groups[: int(num_cases)]
        self.cases = case_groups

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found next to {self.index_path}")
        with metadata_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _load_block_catalog(self) -> list[dict[str, Any]]:
        catalog = self.metadata.get("block_catalog")
        if catalog:
            return sorted([dict(item) for item in catalog], key=lambda item: int(item["block_index"]))

        required = {"block_index", "block_name", "block_signature", "ni", "nj", "nk"}
        missing = required.difference(self.index_df.columns)
        if missing:
            raise ValueError(
                "metadata.json does not include block_catalog and index.csv is missing "
                f"columns required to infer it: {sorted(missing)}"
            )

        inferred: list[dict[str, Any]] = []
        for block_index, frame in self.index_df.groupby("block_index", sort=True):
            first = frame.iloc[0]
            shape = [int(first["ni"]), int(first["nj"]), int(first["nk"])]
            inferred.append(
                {
                    "block_index": int(block_index),
                    "block_name": str(first["block_name"]),
                    "block_signature": str(first["block_signature"]),
                    "cell_shape_ijk": shape,
                    "n_records": int(len(frame)),
                }
            )
        return inferred

    def _get_h5_handle(self, shard_path: str) -> h5py.File:
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        current_worker = worker_info.id if worker_info is not None else -1
        if self._worker_id != current_worker:
            self.close()
            self._worker_id = current_worker

        if shard_path not in self._h5_handles:
            full_path = Path(shard_path)
            if not full_path.is_absolute():
                full_path = self.root / shard_path
            if not full_path.exists():
                raise FileNotFoundError(f"HDF5 shard not found: {full_path}")
            self._h5_handles[shard_path] = h5py.File(
                full_path,
                "r",
                rdcc_nbytes=self.cache_size_mb * 1024 * 1024,
                rdcc_nslots=1007,
                swmr=True,
                libver="latest",
            )
        return self._h5_handles[shard_path]

    @staticmethod
    def _add_normalized_indices(coords: torch.Tensor) -> torch.Tensor:
        """Append normalized structured-grid indices as model coordinate channels."""
        if coords.ndim != 4:
            raise ValueError(f"Expected coords with shape (3,I,J,K), got {tuple(coords.shape)}")
        _, ni, nj, nk = coords.shape
        dtype = coords.dtype
        i = torch.arange(ni, dtype=dtype).view(ni, 1, 1).expand(ni, nj, nk)
        j = torch.arange(nj, dtype=dtype).view(1, nj, 1).expand(ni, nj, nk)
        k = torch.arange(nk, dtype=dtype).view(1, 1, nk).expand(ni, nj, nk)
        i_norm = i / max(ni - 1, 1)
        j_norm = j / max(nj - 1, 1)
        k_norm = k / max(nk - 1, 1)
        return torch.cat(
            [coords, i_norm.unsqueeze(0), j_norm.unsqueeze(0), k_norm.unsqueeze(0)],
            dim=0,
        )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self.cases[int(idx)]
        blocks: Dict[int, Dict[str, torch.Tensor | str | tuple[int, int, int]]] = {}
        geometry: Optional[torch.Tensor] = None
        flow_conditions: Optional[torch.Tensor] = None
        forces: Optional[torch.Tensor] = None

        for record in case["rows"]:
            h5_file = self._get_h5_handle(record.shard_path)
            local_idx = record.local_index
            state = torch.tensor(h5_file["state"][local_idx], dtype=torch.float32)
            coords_pde = torch.tensor(h5_file["coords"][local_idx], dtype=torch.float32)
            cell_volumes = None
            if "cell_volumes" in h5_file:
                cell_volumes = torch.tensor(h5_file["cell_volumes"][local_idx], dtype=torch.float32)
            coords = self._add_normalized_indices(coords_pde)
            if geometry is None:
                geometry_values = h5_file["geometry"][local_idx]
                if self.geometry_feature_indices is not None:
                    geometry_values = geometry_values[self.geometry_feature_indices]
                legacy_indices = self.legacy_double_alignment_indices_by_family.get(
                    case["family"]
                )
                if legacy_indices is not None:
                    source_geometry = np.asarray(geometry_values, dtype=np.float32)
                    geometry_values = np.zeros(self.geometry_dim, dtype=np.float32)
                    for target_idx, source_idx in legacy_indices:
                        geometry_values[target_idx] = source_geometry[source_idx]
                geometry = torch.tensor(
                    geometry_values,
                    dtype=torch.float32,
                )
                flow_conditions = torch.tensor(h5_file["flow"][local_idx], dtype=torch.float32)
                forces = torch.tensor(h5_file["forces"][local_idx], dtype=torch.float32)
            block_payload: Dict[str, torch.Tensor | str | tuple[int, int, int]] = {
                "state": state,
                "target": state,
                "coords": coords,
                "coords_pde": coords_pde,
                "block_name": record.block_name,
                "block_signature": record.block_signature,
                "cell_shape_ijk": record.cell_shape_ijk,
            }
            if cell_volumes is not None:
                block_payload["cell_volumes"] = cell_volumes
            blocks[record.block_index] = block_payload

        assert geometry is not None
        assert flow_conditions is not None
        assert forces is not None
        payload = {
            "sample_id": case["sample_id"],
            "cond_idx": case["cond_idx"],
            "family": case["family"],
            "band": case["band"],
            "cgns_basename": case["cgns_basename"],
            "source_case_dir": case["source_case_dir"],
            "geometry": geometry,
            "flow_conditions": flow_conditions,
            "forces": forces,
            "blocks": blocks,
        }
        if "geometry_id" in case:
            payload["geometry_id"] = case["geometry_id"]
            payload["subtype"] = case["subtype"]
        if "tier3_variant" in case:
            payload["tier3_variant"] = case["tier3_variant"]
        return payload

    def get_block_catalog(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.block_catalog)

    def close(self) -> None:
        for handle in self._h5_handles.values():
            handle.close()
        self._h5_handles.clear()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def whole_case_collate(samples: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Batch whole-case samples while preserving per-block tensor shapes."""
    items = list(samples)
    if not items:
        raise ValueError("Cannot collate an empty whole-case batch")

    block_indices = sorted(items[0]["blocks"].keys())
    batched_blocks: Dict[int, Dict[str, Any]] = {}
    for block_index in block_indices:
        block_items = [item["blocks"][block_index] for item in items]
        batched_block: Dict[str, Any] = {
            "state": torch.stack([block["state"] for block in block_items], dim=0),
            "target": torch.stack([block["target"] for block in block_items], dim=0),
            "coords": torch.stack([block["coords"] for block in block_items], dim=0),
            "coords_pde": torch.stack([block["coords_pde"] for block in block_items], dim=0),
            "block_name": block_items[0]["block_name"],
            "block_signature": block_items[0]["block_signature"],
            "cell_shape_ijk": block_items[0]["cell_shape_ijk"],
        }
        has_cell_volumes = ["cell_volumes" in block for block in block_items]
        if any(has_cell_volumes):
            if not all(has_cell_volumes):
                raise ValueError(f"Inconsistent cell_volumes presence for block {block_index}")
            batched_block["cell_volumes"] = torch.stack(
                [block["cell_volumes"] for block in block_items],
                dim=0,
            )
        batched_blocks[int(block_index)] = batched_block

    batch = {
        "sample_id": [str(item["sample_id"]) for item in items],
        "cond_idx": torch.tensor([int(item["cond_idx"]) for item in items], dtype=torch.long),
        "family": [str(item["family"]) for item in items],
        "band": [str(item["band"]) for item in items],
        "cgns_basename": [str(item["cgns_basename"]) for item in items],
        "source_case_dir": [str(item["source_case_dir"]) for item in items],
        "geometry": torch.stack([item["geometry"] for item in items], dim=0),
        "flow_conditions": torch.stack([item["flow_conditions"] for item in items], dim=0),
        "forces": torch.stack([item["forces"] for item in items], dim=0),
        "blocks": batched_blocks,
    }
    if "geometry_id" in items[0]:
        batch["geometry_id"] = [str(item["geometry_id"]) for item in items]
        batch["subtype"] = [str(item["subtype"]) for item in items]
    if "tier3_variant" in items[0]:
        batch["tier3_variant"] = torch.tensor(
            [int(item["tier3_variant"]) for item in items],
            dtype=torch.long,
        )
    return batch


__all__ = ["BlockRecord", "WholeCaseBlockDataset", "whole_case_collate"]
