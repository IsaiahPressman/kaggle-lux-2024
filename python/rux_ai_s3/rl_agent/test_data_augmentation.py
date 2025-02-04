import itertools

import numpy as np
import pytest
import torch

from rux_ai_s3.constants import MAP_SIZE, MAX_UNITS
from rux_ai_s3.rl_agent.data_augmentation import (
    DataAugmenter,
    DriftReflect,
    PlayerReflect,
    Rot180,
)
from rux_ai_s3.types import Action

_ALL_AUGMENTERS = [
    Rot180(),
    PlayerReflect(),
    DriftReflect(),
]


class TestBaseDataAugmenter:
    @pytest.mark.parametrize(
        "augmenter",
        _ALL_AUGMENTERS,
    )
    def test_drift_direction_flipped(self, augmenter: DataAugmenter) -> None:
        transformed_coordinates = augmenter.transform_spatial(
            torch.from_numpy(np.mgrid[0:MAP_SIZE, 0:MAP_SIZE])
        ).permute(1, 2, 0)
        top_right_coordinates = (0, MAP_SIZE - 1)
        bottom_left_coordinates = (MAP_SIZE - 1, 0)
        top_right = tuple(transformed_coordinates[top_right_coordinates].tolist())
        bottom_left = tuple(transformed_coordinates[bottom_left_coordinates].tolist())
        if augmenter.drift_direction_flipped:
            assert top_right == bottom_left_coordinates
            assert bottom_left == top_right_coordinates
        else:
            assert top_right == top_right_coordinates
            assert bottom_left == bottom_left_coordinates

    @pytest.mark.parametrize(
        "augmenter",
        _ALL_AUGMENTERS,
    )
    def test_inverse_transform_spatial(self, augmenter: DataAugmenter) -> None:
        x = torch.rand([5, MAP_SIZE, MAP_SIZE])
        transformed = augmenter.transform_spatial(x)
        transformed_back = augmenter.inverse_transform_spatial(transformed)
        assert torch.equal(transformed_back, x)

    @pytest.mark.parametrize(
        "augmenter",
        _ALL_AUGMENTERS,
    )
    def test_transform_coordinates(self, augmenter: DataAugmenter) -> None:
        coordinates = torch.from_numpy(np.mgrid[0:MAP_SIZE, 0:MAP_SIZE])
        transformed_coordinates = augmenter.transform_coordinates(
            coordinates.permute(1, 2, 0), MAP_SIZE
        ).permute(2, 0, 1)
        expected_transformed_coordinates = augmenter.transform_spatial(coordinates)
        assert torch.equal(transformed_coordinates, expected_transformed_coordinates)

    @pytest.mark.parametrize(
        "augmenter",
        _ALL_AUGMENTERS,
    )
    def test_move_action_map(self, augmenter: DataAugmenter) -> None:
        move_action_map = augmenter.get_move_action_map()
        move_actions = (
            Action.UP,
            Action.RIGHT,
            Action.DOWN,
            Action.LEFT,
        )
        assert len(move_action_map) == len(move_actions)
        assert set(dict(move_action_map).keys()) == set(move_actions)
        assert set(dict(move_action_map).values()) == set(move_actions)

    @pytest.mark.parametrize(
        "augmenter",
        _ALL_AUGMENTERS,
    )
    def test_move_action_map_matches(self, augmenter: DataAugmenter) -> None:
        transformed_coordinates = augmenter.transform_spatial(
            torch.from_numpy(np.mgrid[0:MAP_SIZE, 0:MAP_SIZE])
        ).permute(1, 2, 0)
        move_action_map = augmenter.get_move_action_map()
        for x, y, (a_orig, a_transformed) in itertools.product(
            range(MAP_SIZE), range(MAP_SIZE), move_action_map
        ):
            translated_x, translated_y = translate((x, y), a_orig.as_move_delta())
            if not in_bounds((translated_x, translated_y), MAP_SIZE):
                continue

            tx, ty = transformed_coordinates[x, y]
            tx_expected, ty_expected = augmenter.transform_coordinates(
                torch.tensor([x, y]), MAP_SIZE
            )
            assert (tx, ty) == (tx_expected, ty_expected)

            translated_tx, translated_ty = translate(
                (tx, ty), a_transformed.as_move_delta()
            )
            t_translated_x, t_translated_y = augmenter.transform_coordinates(
                torch.tensor([translated_x, translated_y]), MAP_SIZE
            )
            assert (translated_tx, translated_ty) == (t_translated_x, t_translated_y)

    def test_inner_inverse_transform_action_space(self) -> None:
        x = torch.rand([1, MAX_UNITS, len(Action)])
        action_map = Rot180.get_move_action_map()
        transformed = DataAugmenter.inner_transform_action_space(x, action_map)
        transformed_back = DataAugmenter.inner_inverse_transform_action_space(
            transformed, action_map
        )
        assert torch.equal(transformed_back, x)


def translate(pos: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    x, y = pos
    dx, dy = delta
    return x + dx, y + dy


def in_bounds(pos: tuple[int, int], map_size: int) -> bool:
    assert map_size > 0
    x, y = pos
    return 0 <= x < map_size and 0 <= y < map_size


class TestRot180:
    @property
    def augmenter(self) -> Rot180:
        return Rot180()

    def test_transform_spatial(self) -> None:
        x = torch.tensor(
            [
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            ]
        )
        transformed = self.augmenter.transform_spatial(x)
        expected = torch.tensor(
            [
                [[15, 14, 13, 12], [11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]],
                [[15, 14, 13, 12], [11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]],
            ]
        )
        assert torch.equal(transformed, expected)

    def test_transform_coordinates(self) -> None:
        coordinates = torch.tensor(
            [
                [0, 0],
                [0, 2],
                [0, 23],
                [2, 1],
                [23, 1],
                [23, 22],
            ]
        )
        transformed = self.augmenter.transform_coordinates(coordinates, map_size=24)
        expected = torch.tensor(
            [
                [23, 23],
                [23, 21],
                [23, 0],
                [21, 22],
                [0, 22],
                [0, 1],
            ]
        )
        assert torch.equal(transformed, expected)
