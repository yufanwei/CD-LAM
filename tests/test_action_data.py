from __future__ import annotations

import torch

from cd_lam.data.action import (
    adjacent_to_block_anchor,
    block_anchor_to_adjacent,
    minmax_normalize_absolute,
    minmax_normalized_deltas,
    normalized_block_anchor_to_raw_adjacent,
    strided_deltas,
)


def test_delta_of_minmax_normalized_actions_has_expected_algebra() -> None:
    actions = torch.tensor([[[1.0, -1.0], [3.0, 2.0], [7.0, 5.0], [9.0, 8.0]]])
    minimum = torch.tensor([-1.0, -4.0])
    maximum = torch.tensor([9.0, 8.0])
    stride = 2
    actual = minmax_normalized_deltas(actions, minimum, maximum, stride=stride)
    expected = 2.0 * strided_deltas(actions, stride=stride) / (maximum - minimum)
    torch.testing.assert_close(actual, expected)


def test_minmax_absolute_normalization_reaches_endpoints() -> None:
    actions = torch.tensor([[[0.0, 2.0], [10.0, 6.0]]])
    normalized = minmax_normalize_absolute(
        actions, torch.tensor([0.0, 2.0]), torch.tensor([10.0, 6.0])
    )
    torch.testing.assert_close(normalized, torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]]))


def test_constant_minmax_dimension_maps_to_zero() -> None:
    actions = torch.tensor([[[3.0, 2.0], [3.0, 6.0]]])
    normalized = minmax_normalize_absolute(
        actions, torch.tensor([3.0, 2.0]), torch.tensor([3.0, 6.0])
    )
    torch.testing.assert_close(normalized, torch.tensor([[[0.0, -1.0], [0.0, 1.0]]]))


def test_block_anchor_and_adjacent_delta_round_trip() -> None:
    adjacent = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]])
    cumulative = adjacent_to_block_anchor(adjacent, block_size=4)
    torch.testing.assert_close(
        cumulative,
        torch.tensor([[[1.0], [3.0], [6.0], [10.0], [5.0], [11.0], [18.0], [26.0]]]),
    )
    torch.testing.assert_close(
        block_anchor_to_adjacent(cumulative, block_size=4), adjacent
    )


def test_normalized_block_anchor_conversion_recovers_raw_adjacent_actions() -> None:
    raw_adjacent = torch.arange(1, 9, dtype=torch.float32).reshape(1, 4, 2)
    minimum = torch.tensor([-5.0, 2.0])
    maximum = torch.tensor([5.0, 6.0])
    normalized_adjacent = raw_adjacent * (2.0 / (maximum - minimum))
    normalized_cumulative = adjacent_to_block_anchor(normalized_adjacent, block_size=4)
    recovered = normalized_block_anchor_to_raw_adjacent(
        normalized_cumulative, minimum, maximum, block_size=4
    )
    torch.testing.assert_close(recovered, raw_adjacent)
