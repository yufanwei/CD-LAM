from __future__ import annotations

import numpy as np
import pytest

from cd_lam.metrics.fdce import (
    FDCEResult,
    foreground_displacement_chamfer_error,
    pairwise_displacement_costs,
    symmetric_chamfer_distance,
)


def test_fdce_identical_and_global_offset_tracks_are_zero() -> None:
    reference = np.array(
        [
            [[0.0, 0.0], [10.0, 2.0]],
            [[1.0, 0.0], [10.0, 4.0]],
            [[3.0, 0.0], [9.0, 5.0]],
        ]
    )
    assert foreground_displacement_chamfer_error(reference, reference) == pytest.approx(
        0.0
    )
    generated = reference + np.array([100.0, -40.0])
    assert foreground_displacement_chamfer_error(generated, reference) == pytest.approx(
        0.0
    )


def test_fdce_single_track_analytical_mean_over_time_cost() -> None:
    generated = np.array([[[5.0, 5.0]], [[6.0, 5.0]], [[8.0, 5.0]]])
    reference = np.array([[[-2.0, 4.0]], [[-2.0, 4.0]], [[-2.0, 4.0]]])
    costs = pairwise_displacement_costs(generated, reference)
    # Rollout errors are 1 and 3 pixels; c_00=(1+3)/2=2 (Eq. A.3).
    np.testing.assert_allclose(costs, [[2.0]])
    assert foreground_displacement_chamfer_error(generated, reference) == pytest.approx(
        2.0
    )


def test_fdce_uses_pair_time_means_before_chamfer() -> None:
    # Generated track 0 matches reference track 0 at t=1 and track 1 at t=2.
    # The paper's fixed track-pair cost is 5 for every pairing.  A per-frame
    # Chamfer implementation would incorrectly return zero.
    generated = np.array(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[0.0, 0.0], [20.0, 0.0]],
            [[0.0, 0.0], [20.0, 0.0]],
        ]
    )
    reference = np.array(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[0.0, 0.0], [20.0, 0.0]],
            [[10.0, 0.0], [10.0, 0.0]],
        ]
    )
    assert foreground_displacement_chamfer_error(generated, reference) == pytest.approx(
        5.0
    )


def test_fdce_visibility_filter_discards_low_visibility_track() -> None:
    generated = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [100.0, 0.0]],
            [[2.0, 0.0], [200.0, 0.0]],
            [[3.0, 0.0], [300.0, 0.0]],
        ]
    )
    reference = generated[:, :1].copy()
    generated_visibility = np.array(
        [[True, True], [True, False], [True, False], [True, False]]
    )
    details = foreground_displacement_chamfer_error(
        generated,
        reference,
        generated_visibility,
        np.ones((4, 1), dtype=bool),
        min_visible_fraction=0.8,
        return_details=True,
    )
    assert isinstance(details, FDCEResult)
    assert details.score == pytest.approx(0.0)
    assert details.generated_tracks == 1
    assert details.reference_tracks == 1


def test_symmetric_point_chamfer_analytical() -> None:
    first = np.array([[0.0], [2.0]])
    second = np.array([[0.0]])
    # first->second mean = 1; second->first mean = 0; symmetric = 0.5.
    assert symmetric_chamfer_distance(first, second) == pytest.approx(0.5)


def test_fdce_rejects_missing_frame_zero_visibility() -> None:
    tracks = np.zeros((3, 1, 2), dtype=np.float32)
    visibility = np.array([[False], [True], [True]])
    with pytest.raises(ValueError, match="no generated tracks"):
        foreground_displacement_chamfer_error(
            tracks,
            tracks,
            visibility,
            np.ones_like(visibility),
            min_visible_fraction=0.0,
        )
