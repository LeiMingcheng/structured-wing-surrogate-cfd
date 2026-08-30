import math

import pytest
import torch

from surrogate3d.evaluation.field_metrics import compute_block_field_metrics


def test_compute_block_field_metrics_aggregates_blocks_and_volume_weights():
    reference = {
        0: torch.tensor([[[[[1.0, 2.0]]], [[[2.0, 4.0]]]]]),
        1: torch.tensor([[[[[3.0]]], [[[6.0]]]]]),
    }
    candidate = {
        0: reference[0] + 1.0,
        1: reference[1] + 2.0,
    }
    volumes = {
        0: torch.tensor([[[[1.0, 1.0]]]]),
        1: torch.tensor([[[[4.0]]]]),
    }

    unweighted = compute_block_field_metrics(
        candidate,
        reference,
        channel_names=["a", "b"],
    )
    weighted = compute_block_field_metrics(
        candidate,
        reference,
        channel_names=["a", "b"],
        volume_blocks=volumes,
    )

    assert unweighted["summary"]["mse"] == pytest.approx(2.0)
    assert unweighted["summary"]["l2"] == pytest.approx(math.sqrt(12.0))
    assert unweighted["channels"]["a"]["mse"] == pytest.approx(2.0)
    assert weighted["summary"]["mse"] == pytest.approx(3.0)
    assert weighted["summary"]["l2"] == pytest.approx(math.sqrt(36.0))
