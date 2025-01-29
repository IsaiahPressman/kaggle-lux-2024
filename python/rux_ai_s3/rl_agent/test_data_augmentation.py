import torch

from rux_ai_s3.rl_agent.data_augmentation import (
    DataAugmenter,
    MoveActionMapType,
    Rot180,
)
from rux_ai_s3.types import Action


def _test_inverse_transform_spatial(augmenter: DataAugmenter) -> None:
    x = torch.rand([5, 10, 10])
    transformed = augmenter.transform_spatial(x)
    transformed_back = augmenter.inverse_transform_spatial(transformed)
    assert torch.equal(transformed_back, x)


def _test_move_action_map(move_action_map: MoveActionMapType) -> None:
    move_actions = (
        Action.UP,
        Action.RIGHT,
        Action.DOWN,
        Action.LEFT,
    )
    assert len(move_action_map) == len(move_actions)
    assert set(dict(move_action_map).keys()) == set(move_actions)
    assert set(dict(move_action_map).values()) == set(move_actions)


class TestBaseDataAugmenter:
    def test_inner_inverse_transform_action_space(self) -> None:
        x = torch.rand([1, 16, len(Action)])
        action_map = Rot180.get_move_action_map()
        transformed = DataAugmenter.inner_transform_action_space(x, action_map)
        transformed_back = DataAugmenter.inner_inverse_transform_action_space(
            transformed, action_map
        )
        assert torch.equal(transformed_back, x)


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

    def test_inverse_transform_spatial(self) -> None:
        _test_inverse_transform_spatial(self.augmenter)

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

    def test_get_move_action_map(self) -> None:
        _test_move_action_map(self.augmenter.get_move_action_map())
