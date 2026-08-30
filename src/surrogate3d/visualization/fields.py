"""3D flow-field diagnostics for whole-case surrogate predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from surrogate3d.data.schema import STATE_FIELDS
from surrogate3d.evaluation.inference import WholeCasePrediction, prediction_to_cpu


FIELD_ALIASES = {
    "rho": "Density",
    "density": "Density",
    "u": "VelocityX",
    "vx": "VelocityX",
    "v": "VelocityY",
    "vy": "VelocityY",
    "w": "VelocityZ",
    "vz": "VelocityZ",
    "p": "Pressure",
    "pressure": "Pressure",
    "nut": "TurbulentSANuTilde",
    "nu_tilde": "TurbulentSANuTilde",
    "turbulent": "TurbulentSANuTilde",
}

_AIRFOIL_SLICE_CHORD_PAD_FRACTION = 0.16
_AIRFOIL_SLICE_MIN_CHORD_PAD = 0.04
_AIRFOIL_SLICE_NORMAL_HALF_HEIGHT_FRACTION = 0.95
_AIRFOIL_SLICE_MIN_NORMAL_HALF_HEIGHT = 0.16
_SCALAR_BAR_ARGS = {
    "height": 0.12,
    "width": 0.46,
    "position_x": 0.50,
    "position_y": 0.84,
    "vertical": False,
    "n_labels": 4,
    "fmt": "%.2f",
    "title_font_size": 36,
    "label_font_size": 32,
    "unconstrained_font_size": True,
    "outline": False,
    "fill": False,
}

_ERROR_SCALAR_BAR_ARGS = {
    **_SCALAR_BAR_ARGS,
    "fmt": "%.1e",
}


@dataclass(frozen=True)
class _MeshBundle:
    surface_meshes: list[object]
    slice_meshes: list[object]
    values: list[np.ndarray]


def _field_index(field: str) -> int:
    name = FIELD_ALIASES.get(str(field).lower(), field)
    if name not in STATE_FIELDS:
        raise ValueError(f"Unknown field {field!r}; expected one of {STATE_FIELDS}")
    return list(STATE_FIELDS).index(name)


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _block_indices(prediction: WholeCasePrediction, block_indices: Sequence[int] | None) -> list[int]:
    available = sorted(int(k) for k in prediction.pred_physical)
    if block_indices is None:
        return available
    selected = [int(k) for k in block_indices]
    missing = sorted(set(selected).difference(available))
    if missing:
        raise KeyError(f"Requested block indices are missing from prediction: {missing}")
    return selected


def _robust_limits(values: Iterable[np.ndarray], q: float = 0.005) -> tuple[float, float]:
    finite_parts = []
    for value in values:
        arr = np.asarray(value).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_parts.append(arr)
    if not finite_parts:
        return 0.0, 1.0
    merged = np.concatenate(finite_parts)
    lo = float(np.quantile(merged, q))
    hi = float(np.quantile(merged, 1.0 - q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(merged))
        hi = float(np.nanmax(merged))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def _symmetric_error_limit(values: Iterable[np.ndarray]) -> float:
    finite_parts = []
    for value in values:
        arr = np.asarray(value).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_parts.append(np.abs(arr))
    if not finite_parts:
        return 1.0
    hi = float(np.quantile(np.concatenate(finite_parts), 0.995))
    return max(hi, 1.0e-12)


def _parse_slice_fractions(value: str | Sequence[float] | None) -> tuple[float, ...]:
    if value is None:
        return (0.18, 0.35, 0.55, 0.75)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(float(part) for part in parts)
    return tuple(float(part) for part in value)


def _is_physical_axis(axis: str) -> bool:
    return str(axis).lower() in {"x", "y", "z"}


def _axis_vector(axis: str) -> tuple[float, float, float]:
    axis = str(axis).lower()
    if axis == "x":
        return (1.0, 0.0, 0.0)
    if axis == "y":
        return (0.0, 1.0, 0.0)
    if axis == "z":
        return (0.0, 0.0, 1.0)
    raise ValueError("physical slice axis must be one of 'x', 'y', or 'z'")


def _axis_coord_index(axis: str) -> int:
    axis = str(axis).lower()
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "z":
        return 2
    raise ValueError("physical slice axis must be one of 'x', 'y', or 'z'")


def _slice_origin(bounds: tuple[float, float, float, float, float, float], axis: str, value: float) -> tuple[float, float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    origin = [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0]
    origin[_axis_coord_index(axis)] = float(value)
    return tuple(float(v) for v in origin)


def _global_axis_bounds(
    prediction: WholeCasePrediction,
    *,
    block_indices: Sequence[int],
    axis: str,
    surface_layer_k: int | None = None,
) -> tuple[float, float]:
    coord_index = _axis_coord_index(axis)
    lo = np.inf
    hi = -np.inf
    for block_index in block_indices:
        coords = _as_numpy(prediction.batch["blocks"][block_index]["coords_pde"][0])
        if surface_layer_k is None:
            values = coords[coord_index]
        else:
            k = int(np.clip(surface_layer_k, 0, coords.shape[3] - 1))
            values = coords[coord_index, :, :, k]
        lo = min(lo, float(np.nanmin(values)))
        hi = max(hi, float(np.nanmax(values)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"Could not determine physical axis bounds for {axis!r}")
    return lo, hi


def _physical_slice_values(
    prediction: WholeCasePrediction,
    *,
    block_indices: Sequence[int],
    axis: str,
    index: int | None,
    fractions: Sequence[float],
    surface_layer_k: int,
) -> list[float]:
    lo, hi = _global_axis_bounds(
        prediction,
        block_indices=block_indices,
        axis=axis,
        surface_layer_k=surface_layer_k,
    )
    if index is not None:
        # For physical axes, --slice-index is interpreted as an absolute coordinate
        # rounded through argparse's int type. Prefer --slice-fractions for stations.
        return [float(index)]
    return [lo + float(fraction) * (hi - lo) for fraction in fractions]


def _selected_indices(length: int, *, index: int | None, fractions: Sequence[float]) -> list[int]:
    if length <= 0:
        return []
    if index is not None:
        return [int(np.clip(index, 0, length - 1))]
    selected = []
    for fraction in fractions:
        selected.append(int(np.clip(round(float(fraction) * (length - 1)), 0, length - 1)))
    return sorted(set(selected))


def _stride_2d(array: np.ndarray, stride: int) -> np.ndarray:
    step = max(1, int(stride))
    return array[::step, ::step]


def _structured_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, values: np.ndarray):
    import pyvista as pv

    if min(x.shape) < 2:
        return None
    x3 = np.ascontiguousarray(x[:, :, None])
    y3 = np.ascontiguousarray(y[:, :, None])
    z3 = np.ascontiguousarray(z[:, :, None])
    grid = pv.StructuredGrid(x3, y3, z3)
    grid.point_data["value"] = np.asarray(values, dtype=np.float32).ravel(order="F")
    return grid


def _structured_volume(coords: np.ndarray, values: np.ndarray, stride: int):
    import pyvista as pv

    step = max(1, int(stride))
    x = np.ascontiguousarray(coords[0, ::step, ::step, ::step])
    y = np.ascontiguousarray(coords[1, ::step, ::step, ::step])
    z = np.ascontiguousarray(coords[2, ::step, ::step, ::step])
    if min(x.shape) < 2:
        return None
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["value"] = np.asarray(values[::step, ::step, ::step], dtype=np.float32).ravel(order="F")
    return grid


def _structured_near_wall_volume(coords: np.ndarray, values: np.ndarray, stride: int, wall_layers: int | None):
    if wall_layers is None or int(wall_layers) <= 0:
        return _structured_volume(coords, values, stride=stride)
    k_stop = min(values.shape[2], int(wall_layers))
    return _structured_volume(coords[:, :, :, :k_stop], values[:, :, :k_stop], stride=stride)


def _extract_plane(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    axis: str,
    index: int,
    stride: int,
    wall_layers: int | None = None,
):
    axis = str(axis).lower()
    step = max(1, int(stride))
    k_stop = values.shape[2]
    if wall_layers is not None and int(wall_layers) > 0:
        k_stop = min(k_stop, int(wall_layers))
    if axis == "i":
        x = coords[0, index, ::step, :k_stop:step]
        y = coords[1, index, ::step, :k_stop:step]
        z = coords[2, index, ::step, :k_stop:step]
        v = values[index, ::step, :k_stop:step]
    elif axis == "j":
        x = coords[0, ::step, index, :k_stop:step]
        y = coords[1, ::step, index, :k_stop:step]
        z = coords[2, ::step, index, :k_stop:step]
        v = values[::step, index, :k_stop:step]
    elif axis == "k":
        x = coords[0, ::step, ::step, index]
        y = coords[1, ::step, ::step, index]
        z = coords[2, ::step, ::step, index]
        v = values[::step, ::step, index]
    else:
        raise ValueError("slice_axis must be one of 'i', 'j', or 'k'")
    return _structured_surface(x, y, z, v)


def _extract_physical_slice(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    axis: str,
    value: float,
    stride: int,
    bounds: tuple[float, float, float, float, float, float],
    wall_layers: int | None,
    near_wall_clip: bool = True,
):
    grid = _structured_near_wall_volume(coords, values, stride=stride, wall_layers=wall_layers)
    if grid is None:
        return None
    mesh = grid.slice(normal=_axis_vector(axis), origin=_slice_origin(bounds, axis, value))
    if mesh.n_points < 3:
        return None
    if near_wall_clip:
        clipped = _clip_slice_to_wall_envelope(mesh, coords, axis=axis, value=value)
        if clipped is not None and clipped.n_points >= 3:
            mesh = clipped
    return mesh


def _clip_slice_to_wall_envelope(mesh, coords: np.ndarray, *, axis: str, value: float):
    """Trim a physical slice to the local wall envelope instead of the farfield."""
    coord_index = _axis_coord_index(axis)
    wall = coords[:, :, :, 0]
    station = wall[coord_index]
    span = float(np.nanmax(station) - np.nanmin(station))
    tol = max(0.02 * span, 1.0e-4)
    mask = np.abs(station - float(value)) <= tol
    if int(mask.sum()) < 4:
        idx = int(np.nanargmin(np.abs(station - float(value))))
        unravel = np.unravel_index(idx, station.shape)
        if axis == "x":
            mask = np.zeros_like(station, dtype=bool)
            mask[unravel[0], :] = True
        elif axis == "y":
            mask = np.zeros_like(station, dtype=bool)
            mask[:, unravel[1]] = True
        else:
            mask = np.zeros_like(station, dtype=bool)
            mask[:, unravel[1]] = True
    wall_points = wall[:, mask]
    if wall_points.size == 0:
        return None
    mins = np.nanmin(wall_points, axis=1)
    maxs = np.nanmax(wall_points, axis=1)
    extents = np.maximum(maxs - mins, 1.0e-3)
    if axis == "z":
        chord = max(float(extents[0]), 1.0e-3)
        lo = mins.copy()
        hi = maxs.copy()
        chord_pad = max(_AIRFOIL_SLICE_CHORD_PAD_FRACTION * chord, _AIRFOIL_SLICE_MIN_CHORD_PAD)
        lo[0] = mins[0] - chord_pad
        hi[0] = maxs[0] + chord_pad
        mid_y = 0.5 * (mins[1] + maxs[1])
        half_height = max(
            _AIRFOIL_SLICE_NORMAL_HALF_HEIGHT_FRACTION * chord,
            _AIRFOIL_SLICE_MIN_NORMAL_HALF_HEIGHT,
        )
        lo[1] = mid_y - half_height
        hi[1] = mid_y + half_height
    else:
        padding = np.maximum(0.16 * extents, np.asarray([0.015, 0.015, 0.015], dtype=float))
        lo = mins - padding
        hi = maxs + padding
    clipped = _clip_mesh_to_bounds(mesh, lo=lo, hi=hi, skip_dim=coord_index)
    if clipped is not None and clipped.n_points >= 3:
        return clipped
    return _clip_mesh_points_to_bounds(mesh, lo=lo, hi=hi, skip_dim=coord_index)


def _clip_mesh_to_bounds(mesh, *, lo: np.ndarray, hi: np.ndarray, skip_dim: int):
    clipped = mesh
    for dim in range(3):
        if dim == skip_dim:
            continue
        lower_normal = [0.0, 0.0, 0.0]
        lower_normal[dim] = 1.0
        lower_origin = [0.0, 0.0, 0.0]
        lower_origin[dim] = float(lo[dim])
        clipped = clipped.clip(normal=tuple(lower_normal), origin=tuple(lower_origin), invert=False)
        if clipped.n_points < 3:
            return None

        upper_normal = [0.0, 0.0, 0.0]
        upper_normal[dim] = -1.0
        upper_origin = [0.0, 0.0, 0.0]
        upper_origin[dim] = float(hi[dim])
        clipped = clipped.clip(normal=tuple(upper_normal), origin=tuple(upper_origin), invert=False)
        if clipped.n_points < 3:
            return None
    return clipped


def _clip_mesh_points_to_bounds(mesh, *, lo: np.ndarray, hi: np.ndarray, skip_dim: int):
    points = np.asarray(mesh.points)
    keep = np.ones(points.shape[0], dtype=bool)
    for dim in range(3):
        if dim == skip_dim:
            continue
        keep &= points[:, dim] >= lo[dim]
        keep &= points[:, dim] <= hi[dim]
    if int(keep.sum()) < 3:
        return None
    return mesh.extract_points(keep, adjacent_cells=False, include_cells=True)


def _axis_length(shape_ijk: Sequence[int], axis: str) -> int:
    axis = str(axis).lower()
    if axis == "i":
        return int(shape_ijk[0])
    if axis == "j":
        return int(shape_ijk[1])
    if axis == "k":
        return int(shape_ijk[2])
    raise ValueError("slice_axis must be one of 'i', 'j', 'k', 'x', 'y', or 'z'")


def _field_mesh_bundle(
    prediction: WholeCasePrediction,
    *,
    field_idx: int,
    source: str,
    block_indices: Sequence[int],
    surface_layer_k: int,
    slice_axis: str,
    slice_index: int | None,
    slice_fractions: Sequence[float],
    surface_stride: int,
    slice_stride: int,
    slice_wall_layers: int | None,
    min_slice_axis_points: int,
) -> _MeshBundle:
    surface_meshes = []
    slice_meshes = []
    values_for_limits: list[np.ndarray] = []
    source_blocks = prediction.pred_physical if source == "prediction" else prediction.target_physical
    physical_slice_values = (
        _physical_slice_values(
            prediction,
            block_indices=block_indices,
            axis=slice_axis,
            index=slice_index,
            fractions=slice_fractions,
            surface_layer_k=surface_layer_k,
        )
    if _is_physical_axis(slice_axis)
        else []
    )
    primary_physical_blocks: dict[float, int] = {}
    if _is_physical_axis(slice_axis):
        coord_index = _axis_coord_index(slice_axis)
        for value in physical_slice_values:
            best_block = None
            best_score = -np.inf
            for block_index in block_indices:
                coords = _as_numpy(prediction.batch["blocks"][block_index]["coords_pde"][0])
                wall_k = int(np.clip(surface_layer_k, 0, coords.shape[3] - 1))
                wall = coords[:, :, :, wall_k]
                station_values = wall[coord_index]
                if value < float(np.nanmin(station_values)) or value > float(np.nanmax(station_values)):
                    continue
                x_extent = float(np.nanmax(wall[0]) - np.nanmin(wall[0]))
                y_extent = float(np.nanmax(wall[1]) - np.nanmin(wall[1]))
                score = x_extent * max(y_extent, 1.0e-6)
                if score > best_score:
                    best_score = score
                    best_block = int(block_index)
            if best_block is not None:
                primary_physical_blocks[float(value)] = best_block

    for block_index in block_indices:
        block = prediction.batch["blocks"][block_index]
        coords = _as_numpy(block["coords_pde"][0])
        values = _as_numpy(source_blocks[block_index][0, field_idx])
        k = int(np.clip(surface_layer_k, 0, values.shape[2] - 1))
        surface = _extract_plane(
            coords,
            values,
            axis="k",
            index=k,
            stride=surface_stride,
        )
        if surface is not None:
            surface_meshes.append(surface)
            values_for_limits.append(np.asarray(surface.point_data["value"]))

        if _is_physical_axis(slice_axis):
            coord_index = _axis_coord_index(slice_axis)
            wall_k = int(np.clip(surface_layer_k, 0, coords.shape[3] - 1))
            # A block should contribute to a physical airfoil section only if
            # its wall surface crosses that station. Using volume/nearfield
            # coordinates here lets farfield parts of unrelated blocks leak
            # into the plot and destroys the scale.
            coord_values = coords[coord_index, :, :, wall_k]
            block_lo = float(np.nanmin(coord_values))
            block_hi = float(np.nanmax(coord_values))
            bounds = (
                float(np.nanmin(coords[0])),
                float(np.nanmax(coords[0])),
                float(np.nanmin(coords[1])),
                float(np.nanmax(coords[1])),
                float(np.nanmin(coords[2])),
                float(np.nanmax(coords[2])),
            )
            for value in physical_slice_values:
                if primary_physical_blocks.get(float(value)) != int(block_index):
                    continue
                if value < block_lo or value > block_hi:
                    continue
                mesh = _extract_physical_slice(
                    coords,
                    values,
                    axis=slice_axis,
                    value=value,
                    stride=slice_stride,
                    bounds=bounds,
                    wall_layers=slice_wall_layers,
                    near_wall_clip=True,
                )
                if mesh is not None:
                    slice_meshes.append(mesh)
                    values_for_limits.append(np.asarray(mesh.point_data["value"]))
            continue

        axis_len = _axis_length(values.shape, slice_axis)
        if axis_len < int(min_slice_axis_points):
            continue
        for idx in _selected_indices(axis_len, index=slice_index, fractions=slice_fractions):
            mesh = _extract_plane(
                coords,
                values,
                axis=slice_axis,
                index=idx,
                stride=slice_stride,
                wall_layers=slice_wall_layers,
            )
            if mesh is not None:
                slice_meshes.append(mesh)
                values_for_limits.append(np.asarray(mesh.point_data["value"]))
    return _MeshBundle(surface_meshes=surface_meshes, slice_meshes=slice_meshes, values=values_for_limits)


def _error_mesh_bundle(target: _MeshBundle, prediction: _MeshBundle) -> _MeshBundle:
    surface_meshes = []
    slice_meshes = []
    values_for_limits: list[np.ndarray] = []

    for target_mesh, pred_mesh in zip(target.surface_meshes, prediction.surface_meshes):
        mesh = pred_mesh.copy(deep=True)
        values = np.asarray(pred_mesh.point_data["value"]) - np.asarray(target_mesh.point_data["value"])
        mesh.point_data["value"] = values.astype(np.float32)
        surface_meshes.append(mesh)
        values_for_limits.append(values)

    for target_mesh, pred_mesh in zip(target.slice_meshes, prediction.slice_meshes):
        mesh = pred_mesh.copy(deep=True)
        values = np.asarray(pred_mesh.point_data["value"]) - np.asarray(target_mesh.point_data["value"])
        mesh.point_data["value"] = values.astype(np.float32)
        slice_meshes.append(mesh)
        values_for_limits.append(values)

    return _MeshBundle(surface_meshes=surface_meshes, slice_meshes=slice_meshes, values=values_for_limits)


def _combined_bounds(mesh_bundles: Sequence[_MeshBundle]) -> tuple[float, float, float, float, float, float]:
    bounds = []
    for bundle in mesh_bundles:
        for mesh in [*bundle.surface_meshes, *bundle.slice_meshes]:
            bounds.append(mesh.bounds)
    if not bounds:
        return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    arr = np.asarray(bounds, dtype=float)
    return (
        float(np.min(arr[:, 0])),
        float(np.max(arr[:, 1])),
        float(np.min(arr[:, 2])),
        float(np.max(arr[:, 3])),
        float(np.min(arr[:, 4])),
        float(np.max(arr[:, 5])),
    )


def _camera_position(bounds: tuple[float, float, float, float, float, float]):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.asarray([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
    span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
    # Render in aerodynamic convention for these meshes: x is chordwise, z is
    # spanwise, and y is wall-normal/thickness. Fixed-z airfoil sections are
    # x-y planes, so y should be vertical in the image.
    position = center + np.asarray([1.05 * span, -1.75 * span, 0.78 * span])
    return tuple(position.tolist()), tuple(center.tolist()), (0.0, 1.0, 0.0)


def _parallel_scale(bounds: tuple[float, float, float, float, float, float]) -> float:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
    return max(0.55, 0.43 * diag)


def _set_camera(plotter, bounds: tuple[float, float, float, float, float, float]) -> None:
    plotter.camera_position = _camera_position(bounds)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = _parallel_scale(bounds)
    plotter.camera.clipping_range = (1.0e-3, 1.0e6)


def _add_bundle(
    plotter,
    bundle: _MeshBundle,
    *,
    clim: tuple[float, float],
    cmap: str,
    scalar_bar_title: str,
    show_scalar_bar: bool,
    scalar_bar_args: dict | None = None,
    surface_opacity: float = 0.92,
    slice_opacity: float = 0.85,
) -> None:
    scalar_bar_args = dict(_SCALAR_BAR_ARGS if scalar_bar_args is None else scalar_bar_args, title=scalar_bar_title)
    title_font_size = int(scalar_bar_args.get("title_font_size", 36))
    label_font_size = int(scalar_bar_args.get("label_font_size", 32))
    first = True
    for mesh in bundle.surface_meshes:
        actor = plotter.add_mesh(
            mesh,
            scalars="value",
            cmap=cmap,
            clim=clim,
            opacity=surface_opacity,
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=show_scalar_bar and first,
            scalar_bar_args=scalar_bar_args,
        )
        if show_scalar_bar and first:
            scalar_bar = plotter.scalar_bar
            scalar_bar.SetUnconstrainedFontSize(True)
            title_prop = scalar_bar.GetTitleTextProperty()
            title_prop.SetFontSize(title_font_size)
            title_prop.SetColor(0.0, 0.0, 0.0)
            label_prop = scalar_bar.GetLabelTextProperty()
            label_prop.SetFontSize(label_font_size)
            label_prop.SetColor(0.0, 0.0, 0.0)
        first = False
    for mesh in bundle.slice_meshes:
        actor = plotter.add_mesh(
            mesh,
            scalars="value",
            cmap=cmap,
            clim=clim,
            opacity=slice_opacity,
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=show_scalar_bar and first,
            scalar_bar_args=scalar_bar_args,
        )
        if show_scalar_bar and first:
            scalar_bar = plotter.scalar_bar
            scalar_bar.SetUnconstrainedFontSize(True)
            title_prop = scalar_bar.GetTitleTextProperty()
            title_prop.SetFontSize(title_font_size)
            title_prop.SetColor(0.0, 0.0, 0.0)
            label_prop = scalar_bar.GetLabelTextProperty()
            label_prop.SetFontSize(label_font_size)
            label_prop.SetColor(0.0, 0.0, 0.0)
        outline = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        if outline.n_points > 0:
            plotter.add_mesh(outline, color="black", line_width=1.1, opacity=0.9)
        first = False


def _render_single(
    *,
    bundle: _MeshBundle,
    clim: tuple[float, float],
    field_name: str,
    title: str,
    output_path: Path,
    bounds: tuple[float, float, float, float, float, float],
    cmap: str,
    window_size: tuple[int, int],
) -> Path:
    import pyvista as pv

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    _add_bundle(
        plotter,
        bundle,
        clim=clim,
        cmap=cmap,
        scalar_bar_title=" ",
        show_scalar_bar=True,
    )
    del title
    _set_camera(plotter, bounds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    plotter.close()
    return output_path


def _render_triptych(
    *,
    target: _MeshBundle,
    prediction: _MeshBundle,
    error: _MeshBundle,
    value_clim: tuple[float, float],
    error_clim: tuple[float, float],
    field_name: str,
    title: str,
    output_path: Path,
    bounds: tuple[float, float, float, float, float, float],
    cmap: str,
    error_cmap: str,
    window_size: tuple[int, int],
) -> Path:
    import pyvista as pv

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=window_size)
    plotter.set_background("white")
    panels = [
        ("target", target, value_clim, cmap, " "),
        ("prediction", prediction, value_clim, cmap, None),
        ("prediction - target", error, error_clim, error_cmap, "  "),
    ]
    for col, (label, bundle, clim, panel_cmap, scalar_title) in enumerate(panels):
        plotter.subplot(0, col)
        _add_bundle(
            plotter,
            bundle,
            clim=clim,
            cmap=panel_cmap,
            scalar_bar_title=scalar_title or "",
            show_scalar_bar=scalar_title is not None,
        )
        plotter.add_text(label, position="upper_left", font_size=10, color="black")
        _set_camera(plotter, bounds)
    plotter.link_views()
    del title
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    plotter.close()
    return output_path


def _case_title(prediction: WholeCasePrediction) -> str:
    batch = prediction.batch
    sample_id = str(batch.get("sample_id", ["case"])[0])
    cond_idx = int(batch["cond_idx"][0].detach().cpu()) if "cond_idx" in batch else -1
    flow = batch.get("flow_conditions")
    if isinstance(flow, torch.Tensor) and flow.numel() >= 2:
        mach = float(flow[0, 0].detach().cpu())
        aoa = float(flow[0, 1].detach().cpu())
        return f"{sample_id}, condition #{cond_idx} (Ma={mach:.3f}, AOA={aoa:.2f} deg.)"
    return f"{sample_id}, condition #{cond_idx}"


def plot_prediction_diagnostics(
    prediction: WholeCasePrediction,
    *,
    output_dir: str | Path,
    fields: Sequence[str] = ("Pressure", "VelocityX", "TurbulentSANuTilde"),
    block_indices: Sequence[int] | None = None,
    surface_layer_k: int = 0,
    slice_axis: str = "i",
    slice_index: int | None = None,
    slice_fractions: str | Sequence[float] | None = None,
    surface_stride: int = 1,
    slice_stride: int = 1,
    slice_wall_layers: int | None = 18,
    min_slice_axis_points: int = 32,
    cmap: str = "turbo",
    error_cmap: str = "coolwarm",
    window_size: tuple[int, int] = (1800, 720),
    comparison_window_size: tuple[int, int] = (2400, 720),
    surface_projection: str | None = None,
    slice_projection: str | None = None,
    surface_layer_j: int | None = None,
) -> list[Path]:
    """Create 3D wall-surface and volume-slice diagnostics.

    The wing volume CGNS marks the viscous wall on the ``k=1`` node boundary,
    so the model-facing cell-centered near-wall layer is ``k=0``. The legacy
    ``surface_layer_j``/projection arguments are accepted only for older callers
    and are intentionally ignored by the 3D renderer.
    """
    del surface_projection, slice_projection, surface_layer_j
    prediction = prediction_to_cpu(prediction)
    output_dir = Path(output_dir).expanduser().resolve()
    indices = _block_indices(prediction, block_indices)
    fractions = _parse_slice_fractions(slice_fractions)
    paths: list[Path] = []
    case_title = _case_title(prediction)

    for field in fields:
        field_idx = _field_index(field)
        field_name = STATE_FIELDS[field_idx]
        target = _field_mesh_bundle(
            prediction,
            field_idx=field_idx,
            source="target",
            block_indices=indices,
            surface_layer_k=surface_layer_k,
            slice_axis=slice_axis,
            slice_index=slice_index,
            slice_fractions=fractions,
            surface_stride=surface_stride,
            slice_stride=slice_stride,
            slice_wall_layers=slice_wall_layers,
            min_slice_axis_points=min_slice_axis_points,
        )
        pred = _field_mesh_bundle(
            prediction,
            field_idx=field_idx,
            source="prediction",
            block_indices=indices,
            surface_layer_k=surface_layer_k,
            slice_axis=slice_axis,
            slice_index=slice_index,
            slice_fractions=fractions,
            surface_stride=surface_stride,
            slice_stride=slice_stride,
            slice_wall_layers=slice_wall_layers,
            min_slice_axis_points=min_slice_axis_points,
        )
        err = _error_mesh_bundle(target, pred)
        value_clim = _robust_limits([*target.values, *pred.values])
        err_abs = _symmetric_error_limit(err.values)
        err_clim = (-err_abs, err_abs)
        bounds = _combined_bounds([target, pred, err])
        slice_desc = f"{slice_axis}-slices"
        if slice_index is not None:
            slice_desc = f"{slice_axis}={int(slice_index)} slice"
        if _is_physical_axis(slice_axis):
            slice_desc = f"physical {slice_axis}-station slices"
            if slice_index is not None:
                slice_desc = f"physical {slice_axis}={float(slice_index):.3g} slice"
        layer_desc = ""
        if not _is_physical_axis(slice_axis) and slice_axis != "k" and slice_wall_layers is not None:
            layer_desc = f", first {int(slice_wall_layers)} wall-normal layers"
        common_title = f"{case_title}\n{field_name}, wall k={surface_layer_k} + {slice_desc}{layer_desc}"
        paths.append(
            _render_single(
                bundle=pred,
                clim=value_clim,
                field_name=field_name,
                title=f"{common_title}\nprediction overview",
                output_path=output_dir / f"{field_name}_prediction_overview3d.png",
                bounds=bounds,
                cmap=cmap,
                window_size=window_size,
            )
        )
        paths.append(
            _render_triptych(
                target=target,
                prediction=pred,
                error=err,
                value_clim=value_clim,
                error_clim=err_clim,
                field_name=field_name,
                title=common_title,
                output_path=output_dir / f"{field_name}_comparison3d.png",
                bounds=bounds,
                cmap=cmap,
                error_cmap=error_cmap,
                window_size=comparison_window_size,
            )
        )
    return paths


__all__ = ["FIELD_ALIASES", "plot_prediction_diagnostics"]
