"""Whole-case 3D surrogate models for structured wing CFD datasets."""

__all__ = [
    "HierarchicalGlobalFNO3D",
    "load_whole_case_payload",
    "write_prediction_restart_cgns",
    "write_whole_case_payload",
]


def __getattr__(name: str):
    if name == "HierarchicalGlobalFNO3D":
        from surrogate3d.models import HierarchicalGlobalFNO3D

        return HierarchicalGlobalFNO3D
    if name in {
        "load_whole_case_payload",
        "write_prediction_restart_cgns",
        "write_whole_case_payload",
    }:
        from surrogate3d.nk_resume import (
            load_whole_case_payload,
            write_prediction_restart_cgns,
            write_whole_case_payload,
        )

        return {
            "load_whole_case_payload": load_whole_case_payload,
            "write_prediction_restart_cgns": write_prediction_restart_cgns,
            "write_whole_case_payload": write_whole_case_payload,
        }[name]
    raise AttributeError(f"module 'surrogate3d' has no attribute {name!r}")
