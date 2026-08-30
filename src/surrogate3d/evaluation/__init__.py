"""Evaluation utilities for surrogate3d whole-case checkpoints."""

from surrogate3d.evaluation.adflow_restart import (
    ADflowRestartEvalConfig,
    build_adflow_restart_eval_command,
    build_adflow_restart_eval_env,
    default_adflow_eval_python,
    restart_eval_config_from_dict,
)
from surrogate3d.evaluation.force_eval import (
    ADflowForceEvalConfig,
    build_adflow_force_eval_command,
    compute_force_error_metrics,
    force_eval_config_from_dict,
    read_reference_force_coefficients,
    run_restart_force_evaluation,
)
from surrogate3d.evaluation.inference import (
    WholeCasePrediction,
    load_checkpoint_for_inference,
    predict_case,
    predict_loaded_case,
    select_case_index,
)
from surrogate3d.evaluation.residual_eval import (
    DEFAULT_RESIDUAL_FIELDS,
    ResidualChannelMetricsConfig,
    compare_exported_residual_channels,
    residual_metrics_config_from_dict,
    run_restart_residual_evaluation,
)
from surrogate3d.evaluation.surface_force_eval import (
    SurfaceForceConfig,
    compute_surface_force_coefficients_from_npz,
    read_surface_force_case_metadata,
    run_surface_force_evaluation,
    surface_force_config_from_dict,
)
from surrogate3d.evaluation.field_force_eval import (
    ForceEvalConfig,
    read_force_case_metadata,
    run_force_evaluation,
)
from surrogate3d.evaluation.field_metrics import (
    compute_block_field_metrics,
    compute_prediction_field_metrics,
)

__all__ = [
    "ADflowRestartEvalConfig",
    "ADflowForceEvalConfig",
    "DEFAULT_RESIDUAL_FIELDS",
    "ForceEvalConfig",
    "ResidualChannelMetricsConfig",
    "SurfaceForceConfig",
    "WholeCasePrediction",
    "build_adflow_restart_eval_command",
    "build_adflow_restart_eval_env",
    "build_adflow_force_eval_command",
    "compare_exported_residual_channels",
    "compute_force_error_metrics",
    "compute_block_field_metrics",
    "compute_prediction_field_metrics",
    "default_adflow_eval_python",
    "force_eval_config_from_dict",
    "load_checkpoint_for_inference",
    "predict_case",
    "predict_loaded_case",
    "read_reference_force_coefficients",
    "read_force_case_metadata",
    "read_surface_force_case_metadata",
    "residual_metrics_config_from_dict",
    "restart_eval_config_from_dict",
    "run_force_evaluation",
    "run_surface_force_evaluation",
    "run_restart_residual_evaluation",
    "run_restart_force_evaluation",
    "select_case_index",
    "surface_force_config_from_dict",
    "compute_surface_force_coefficients_from_npz",
]
