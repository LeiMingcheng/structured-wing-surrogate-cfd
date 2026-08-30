"""Whole-case prediction payloads for future ADFLOW NK resume adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np
import torch


PAYLOAD_VERSION = "surrogate3d-whole-case-payload-v1"


def _to_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _json_attr(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def write_whole_case_payload(
    payload_path: str | Path,
    *,
    predicted_blocks: Mapping[int, Any],
    batch: Mapping[str, Any],
    target_blocks: Optional[Mapping[int, Any]] = None,
    block_catalog: Optional[list[dict[str, Any]]] = None,
    sample_offset: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write one sample from a batched whole-case prediction to HDF5.

    ``predicted_blocks`` must already be in physical state space. The payload
    deliberately preserves block topology instead of flattening or masking the
    multi-block grid.
    """
    payload_path = Path(payload_path).expanduser().resolve()
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    sample_offset = int(sample_offset)

    with h5py.File(payload_path, "w") as h5:
        h5.attrs["version"] = PAYLOAD_VERSION
        h5.attrs["metadata_json"] = _json_attr(dict(metadata or {}))
        h5.attrs["block_catalog_json"] = _json_attr(list(block_catalog or []))
        for key in ("sample_id", "family", "band", "cgns_basename", "source_case_dir"):
            values = batch.get(key)
            if isinstance(values, (list, tuple)):
                h5.attrs[key] = str(values[sample_offset])
            elif values is not None:
                h5.attrs[key] = str(values)
        cond_idx = batch.get("cond_idx")
        tier3_variant = batch.get("tier3_variant")
        if isinstance(cond_idx, torch.Tensor):
            h5.attrs["cond_idx"] = int(cond_idx.detach().cpu()[sample_offset])
        if isinstance(tier3_variant, torch.Tensor):
            h5.attrs["tier3_variant"] = int(tier3_variant.detach().cpu()[sample_offset])

        for key in ("geometry", "flow_conditions", "forces"):
            value = batch.get(key)
            if value is not None:
                h5.create_dataset(
                    key,
                    data=_to_numpy(value[sample_offset], np.float32),
                    compression="lzf",
                )

        blocks_group = h5.create_group("blocks")
        for block_index in sorted(int(k) for k in predicted_blocks.keys()):
            block = batch["blocks"][block_index]
            group = blocks_group.create_group(str(block_index))
            group.attrs["block_name"] = str(block.get("block_name", ""))
            group.attrs["block_signature"] = str(block.get("block_signature", ""))
            group.attrs["cell_shape_ijk_json"] = _json_attr(block.get("cell_shape_ijk", ()))
            group.create_dataset(
                "predicted_state",
                data=_to_numpy(predicted_blocks[block_index][sample_offset], np.float32),
                compression="lzf",
            )
            if target_blocks is not None and block_index in target_blocks:
                group.create_dataset(
                    "target_state",
                    data=_to_numpy(target_blocks[block_index][sample_offset], np.float32),
                    compression="lzf",
                )
            if "coords_pde" in block:
                group.create_dataset(
                    "coords",
                    data=_to_numpy(block["coords_pde"][sample_offset], np.float64),
                    compression="lzf",
                )
            elif "coords" in block:
                group.create_dataset(
                    "coords",
                    data=_to_numpy(block["coords"][sample_offset, :3], np.float64),
                    compression="lzf",
                )
            if "coords" in block:
                group.create_dataset(
                    "coords_model",
                    data=_to_numpy(block["coords"][sample_offset], np.float32),
                    compression="lzf",
                )
    return payload_path


def load_whole_case_payload(payload_path: str | Path) -> dict[str, Any]:
    """Load a payload written by :func:`write_whole_case_payload`."""
    payload_path = Path(payload_path).expanduser().resolve()
    result: dict[str, Any] = {"path": str(payload_path), "blocks": {}}
    with h5py.File(payload_path, "r") as h5:
        result["version"] = str(h5.attrs.get("version", ""))
        result["metadata"] = json.loads(str(h5.attrs.get("metadata_json", "{}")))
        result["block_catalog"] = json.loads(str(h5.attrs.get("block_catalog_json", "[]")))
        for key in ("sample_id", "family", "band", "cgns_basename", "source_case_dir"):
            if key in h5.attrs:
                result[key] = str(h5.attrs[key])
        for key in ("cond_idx", "tier3_variant"):
            if key in h5.attrs:
                result[key] = int(h5.attrs[key])
        for key in ("geometry", "flow_conditions", "forces"):
            if key in h5:
                result[key] = np.asarray(h5[key])
        for block_key, group in h5["blocks"].items():
            block_index = int(block_key)
            block = {
                "block_name": str(group.attrs.get("block_name", "")),
                "block_signature": str(group.attrs.get("block_signature", "")),
                "cell_shape_ijk": tuple(json.loads(str(group.attrs.get("cell_shape_ijk_json", "[]")))),
                "predicted_state": np.asarray(group["predicted_state"]),
            }
            if "target_state" in group:
                block["target_state"] = np.asarray(group["target_state"])
            if "coords" in group:
                block["coords"] = np.asarray(group["coords"])
            if "coords_model" in group:
                block["coords_model"] = np.asarray(group["coords_model"])
            result["blocks"][block_index] = block
    return result


__all__ = ["PAYLOAD_VERSION", "load_whole_case_payload", "write_whole_case_payload"]
