"""Write surrogate3d whole-case predictions into ADflow CGNS restart files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from parametric_swept_wing_cfd.export.cgns_state import CGNS_DATA, VOLUME_STATE_FIELDS


DEFAULT_RESTART_FIELD_FLOORS: dict[str, float] = {
    "Density": 1.0e-6,
    "Pressure": 1.0e-6,
    "TurbulentSANuTilde": 0.0,
}


def _node_data(group) -> np.ndarray:
    if isinstance(group, h5py.Dataset):
        return np.asarray(group)
    if CGNS_DATA not in group:
        raise KeyError(f"CGNS node has no '{CGNS_DATA}' dataset: {group.name}")
    return np.asarray(group[CGNS_DATA])


def _decode_text(data: np.ndarray) -> str:
    values = np.asarray(data).reshape(-1)
    if values.dtype.kind in {"i", "u"}:
        raw = bytes(int(v) & 0xFF for v in values if int(v) != 0)
        return raw.decode("ascii", errors="ignore").strip()
    return b"".join(np.asarray(values, dtype="S")).decode("ascii", errors="ignore").strip()


def _grid_location(solution_group) -> str:
    if "GridLocation" not in solution_group:
        return ""
    return _decode_text(_node_data(solution_group["GridLocation"]))


def _rind(solution_group, ndim: int) -> np.ndarray:
    if "Rind" not in solution_group:
        return np.zeros(ndim * 2, dtype=np.int32)
    raw = np.asarray(_node_data(solution_group["Rind"]), dtype=np.int32).reshape(-1)
    if raw.size < ndim * 2:
        padded = np.zeros(ndim * 2, dtype=np.int32)
        padded[: raw.size] = raw
        return padded
    return raw[: ndim * 2]


def _find_solution_group(zone_group):
    for name in ("Flow solution", "FlowSolution", "FlowSolution#Centers"):
        if name in zone_group and hasattr(zone_group[name], "keys"):
            return zone_group[name]
    for value in zone_group.values():
        if hasattr(value, "keys") and "GridLocation" in value:
            return value
    return None


def _iter_volume_zones(h5file: h5py.File) -> list[tuple[str, Any, Any]]:
    zones = []
    for base_group in h5file.values():
        if not hasattr(base_group, "items"):
            continue
        for zone_name, zone_group in base_group.items():
            if not hasattr(zone_group, "keys") or "GridCoordinates" not in zone_group:
                continue
            solution = _find_solution_group(zone_group)
            if solution is None:
                continue
            zones.append((str(zone_name), zone_group, solution))
    return zones


def _field_dataset(solution_group, field_name: str) -> h5py.Dataset:
    node = solution_group[field_name]
    if isinstance(node, h5py.Dataset):
        return node
    if CGNS_DATA not in node:
        raise KeyError(f"Field {field_name} has no '{CGNS_DATA}' dataset in {solution_group.name}")
    return node[CGNS_DATA]


def _matching_solution_group(h5file: h5py.File, zone_name: str):
    for candidate_zone_name, _zone, solution in _iter_volume_zones(h5file):
        if candidate_zone_name == zone_name:
            return solution
    raise KeyError(f"Zone {zone_name!r} not found in {h5file.filename}")


def _write_interior(raw_dataset: h5py.Dataset, values_ijk: np.ndarray, rind: np.ndarray) -> None:
    """Write model-order ``(i,j,k)`` values into raw CGNS ``(k,j,i)`` interior."""
    values_kji = np.ascontiguousarray(np.transpose(values_ijk, (2, 1, 0)))
    ilo, ihi, jlo, jhi, klo, khi = [int(v) for v in rind[:6]]
    raw_shape = tuple(int(v) for v in raw_dataset.shape)
    expected = (
        raw_shape[0] - klo - khi,
        raw_shape[1] - jlo - jhi,
        raw_shape[2] - ilo - ihi,
    )
    if values_kji.shape != expected:
        raise ValueError(
            f"Interior shape mismatch for {raw_dataset.name}: values {values_kji.shape}, "
            f"expected {expected} from raw_shape={raw_shape}, rind={rind.tolist()}"
        )
    raw_dataset[
        slice(klo, raw_shape[0] - khi if khi else None),
        slice(jlo, raw_shape[1] - jhi if jhi else None),
        slice(ilo, raw_shape[2] - ihi if ihi else None),
    ] = values_kji.astype(raw_dataset.dtype, copy=False)


def apply_field_floors(
    state_cijk: np.ndarray,
    *,
    field_names: Sequence[str] = VOLUME_STATE_FIELDS,
    field_floors: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return a copy of ``state_cijk`` with selected channel lower bounds.

    This is intentionally a restart-export projection, not a training-time
    loss. It prevents ADflow from immediately failing on nonphysical negative
    density, pressure, or SA turbulence variables while preserving the raw
    surrogate payload for diagnostics.
    """
    floors = dict(field_floors or {})
    if not floors:
        return np.asarray(state_cijk)
    projected = np.array(state_cijk, copy=True)
    if projected.ndim != 4:
        raise ValueError(f"state_cijk must be (C,I,J,K), got {projected.shape}")
    for channel, field_name in enumerate(field_names):
        if channel >= projected.shape[0]:
            break
        if field_name in floors:
            projected[channel] = np.maximum(projected[channel], float(floors[field_name]))
    return projected


def _block_name_from_batch(block: Mapping[str, Any]) -> str:
    name = str(block.get("block_name", ""))
    if name:
        return name
    signature = str(block.get("block_signature", ""))
    if "domain-" in signature:
        suffix = signature.split("domain-", 1)[1].split("_", 1)[0]
        return "domain." + suffix
    return ""


def write_prediction_restart_cgns(
    *,
    template_cgns: str | Path,
    output_cgns: str | Path,
    predicted_blocks: Mapping[int, torch.Tensor | np.ndarray],
    batch: Mapping[str, Any],
    field_names: Sequence[str] = VOLUME_STATE_FIELDS,
    field_floors: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Copy an ADflow volume CGNS file and replace state fields with predictions.

    The prediction tensors are expected as ``(B, C, I, J, K)`` or ``(C, I, J, K)``
    in physical state units. Only the non-rind interior cells are replaced.
    """
    template_cgns = Path(template_cgns).expanduser().resolve()
    output_cgns = Path(output_cgns).expanduser().resolve()
    if not template_cgns.exists():
        raise FileNotFoundError(f"Template CGNS not found: {template_cgns}")
    output_cgns.parent.mkdir(parents=True, exist_ok=True)
    if template_cgns != output_cgns:
        shutil.copy2(template_cgns, output_cgns)

    blocks = batch["blocks"]
    block_by_zone: dict[str, int] = {}
    for block_index, block in blocks.items():
        name = _block_name_from_batch(block)
        if name:
            block_by_zone[name] = int(block_index)

    written: list[dict[str, Any]] = []
    with h5py.File(output_cgns, "r+") as h5file:
        zones = _iter_volume_zones(h5file)
        for zone_name, _zone, solution in zones:
            if zone_name not in block_by_zone:
                continue
            block_index = block_by_zone[zone_name]
            if block_index not in predicted_blocks:
                raise KeyError(f"Missing predicted block {block_index} for zone {zone_name}")
            location = _grid_location(solution)
            if location and location != "CellCenter":
                raise RuntimeError(f"Expected CellCenter fields in {zone_name}, got {location}")
            rind = _rind(solution, ndim=3)
            pred = predicted_blocks[block_index]
            if isinstance(pred, torch.Tensor):
                pred_np = pred.detach().cpu().numpy()
            else:
                pred_np = np.asarray(pred)
            if pred_np.ndim == 5:
                pred_np = pred_np[0]
            if pred_np.ndim != 4:
                raise ValueError(f"Prediction for block {block_index} must be (C,I,J,K), got {pred_np.shape}")
            if pred_np.shape[0] < len(field_names):
                raise ValueError(
                    f"Prediction for block {block_index} has {pred_np.shape[0]} channels, "
                    f"need {len(field_names)}"
                )
            pred_np = apply_field_floors(pred_np, field_names=field_names, field_floors=field_floors)
            for channel, field_name in enumerate(field_names):
                if field_name not in solution:
                    raise KeyError(f"Field {field_name} missing in zone {zone_name}")
                _write_interior(_field_dataset(solution, field_name), pred_np[channel], rind)
            written.append(
                {
                    "zone_name": zone_name,
                    "block_index": block_index,
                    "shape_ijk": list(pred_np.shape[1:]),
                    "rind_ilo_ihi_jlo_jhi_klo_khi": [int(v) for v in rind],
                }
            )
        h5file.attrs["surrogate3d_restart_metadata_json"] = json.dumps(
            {
                "source_template": str(template_cgns),
                "field_names": list(field_names),
                "field_floors": dict(field_floors or {}),
                "written_blocks": written,
                "metadata": dict(metadata or {}),
            },
            sort_keys=True,
        )
    if len(written) != len(predicted_blocks):
        written_blocks = {item["block_index"] for item in written}
        missing = sorted(set(int(k) for k in predicted_blocks).difference(written_blocks))
        raise RuntimeError(f"Did not write all predicted blocks into CGNS; missing {missing}")
    return output_cgns


def blend_restart_cgns(
    *,
    pred_cgns: str | Path,
    target_cgns: str | Path,
    output_cgns: str | Path,
    target_weight: float,
    field_names: Sequence[str] = VOLUME_STATE_FIELDS,
    field_floors: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a physical-field convex blend of two ADflow restart CGNS files.

    ``target_weight=1`` returns the target state, and ``target_weight=0`` returns
    the prediction state. Blending is done directly on CGNS cell-centered restart
    fields, after denormalization/log turbulence inversion has already happened.
    """
    pred_cgns = Path(pred_cgns).expanduser().resolve()
    target_cgns = Path(target_cgns).expanduser().resolve()
    output_cgns = Path(output_cgns).expanduser().resolve()
    if not pred_cgns.exists():
        raise FileNotFoundError(f"Predicted restart CGNS not found: {pred_cgns}")
    if not target_cgns.exists():
        raise FileNotFoundError(f"Target restart CGNS not found: {target_cgns}")
    weight = float(target_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"target_weight must be in [0, 1], got {target_weight}")

    output_cgns.parent.mkdir(parents=True, exist_ok=True)
    if pred_cgns != output_cgns:
        shutil.copy2(pred_cgns, output_cgns)

    floors = dict(field_floors or {})
    written: list[dict[str, Any]] = []
    with h5py.File(output_cgns, "r+") as out_h5, h5py.File(target_cgns, "r") as target_h5:
        zones = _iter_volume_zones(out_h5)
        for zone_name, _zone, out_solution in zones:
            target_solution = _matching_solution_group(target_h5, zone_name)
            out_location = _grid_location(out_solution)
            target_location = _grid_location(target_solution)
            if out_location and out_location != "CellCenter":
                raise RuntimeError(f"Expected CellCenter fields in {zone_name}, got {out_location}")
            if target_location and target_location != "CellCenter":
                raise RuntimeError(f"Expected CellCenter target fields in {zone_name}, got {target_location}")
            for field_name in field_names:
                if field_name not in out_solution:
                    raise KeyError(f"Field {field_name} missing in output zone {zone_name}")
                if field_name not in target_solution:
                    raise KeyError(f"Field {field_name} missing in target zone {zone_name}")
                out_dataset = _field_dataset(out_solution, field_name)
                target_dataset = _field_dataset(target_solution, field_name)
                if out_dataset.shape != target_dataset.shape:
                    raise ValueError(
                        f"Shape mismatch for {zone_name}/{field_name}: "
                        f"{out_dataset.shape} vs {target_dataset.shape}"
                    )
                blended = (1.0 - weight) * np.asarray(out_dataset) + weight * np.asarray(target_dataset)
                if field_name in floors:
                    blended = np.maximum(blended, float(floors[field_name]))
                out_dataset[...] = blended.astype(out_dataset.dtype, copy=False)
            written.append({"zone_name": zone_name, "fields": list(field_names)})
        out_h5.attrs["surrogate3d_restart_metadata_json"] = json.dumps(
            {
                "pred_cgns": str(pred_cgns),
                "target_cgns": str(target_cgns),
                "target_weight": weight,
                "field_names": list(field_names),
                "field_floors": floors,
                "written_zones": written,
                "metadata": dict(metadata or {}),
            },
            sort_keys=True,
        )
    return output_cgns


__all__ = [
    "DEFAULT_RESTART_FIELD_FLOORS",
    "apply_field_floors",
    "blend_restart_cgns",
    "write_prediction_restart_cgns",
]
