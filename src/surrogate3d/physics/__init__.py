"""Physics helpers for surrogate3d evaluation and training."""

from surrogate3d.physics.block_geometry import (
    StructuredBlockGeometry,
    block_geometry_from_prediction_batch,
    extract_block_geometry,
    extract_case_block_geometry_from_volume_cgns,
)
from surrogate3d.physics.surface_forces import (
    compute_surface_coefficient_metrics,
    compute_surface_field_metrics,
    compute_surface_force_coefficients_torch,
    prepare_surface_force_geometry,
)
from surrogate3d.physics.field_force import (
    ForceEvaluatorConfig,
    compute_force_coefficients_torch,
    compute_force_metric_bundle,
    compute_volume_surface_fields_torch,
    force_evaluator_config_from_dict,
    prepare_volume_force_geometry,
)
from surrogate3d.physics.wall_geometry import (
    BlockWallGeometry,
    extract_block_wall_geometry,
    extract_case_wall_geometry_from_volume_cgns,
    wall_geometry_from_prediction_batch,
)
from surrogate3d.physics.residual_sample import (
    ResidualBlockSample,
    ResidualCaseSample,
    WholeCaseResidualSampleBuilder,
)
from surrogate3d.physics.residual import PDEResidualCalculator

__all__ = [
    "BlockWallGeometry",
    "ForceEvaluatorConfig",
    "PDEResidualCalculator",
    "ResidualBlockSample",
    "ResidualCaseSample",
    "StructuredBlockGeometry",
    "WholeCaseResidualSampleBuilder",
    "block_geometry_from_prediction_batch",
    "compute_force_coefficients_torch",
    "compute_force_metric_bundle",
    "compute_surface_coefficient_metrics",
    "compute_surface_field_metrics",
    "compute_surface_force_coefficients_torch",
    "compute_volume_surface_fields_torch",
    "extract_block_geometry",
    "extract_block_wall_geometry",
    "extract_case_block_geometry_from_volume_cgns",
    "extract_case_wall_geometry_from_volume_cgns",
    "force_evaluator_config_from_dict",
    "prepare_surface_force_geometry",
    "prepare_volume_force_geometry",
    "wall_geometry_from_prediction_batch",
]
