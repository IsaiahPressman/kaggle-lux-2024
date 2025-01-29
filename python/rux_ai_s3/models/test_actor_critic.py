import numpy as np
import pytest
import torch

from rux_ai_s3.models.actor_critic import ActorCriticOut
from rux_ai_s3.types import Action


class TestActorCriticOut:
    def test_to_env_actions(self) -> None:
        main_actions = torch.tensor(
            [
                [Action.NO_OP, Action.UP],
                [Action.RIGHT, Action.DOWN],
                [Action.LEFT, Action.SAP],
                [Action.SAP + 1, Action.SAP + 2],
                [Action.SAP + 3, Action.SAP + 5],
            ]
        )
        sap_actions = torch.tensor(
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 1],
                [24, 24 * 24 - 1],
            ]
        )
        unit_indices = np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[1, 1], [2, 2]],
                [[1, 0], [1, 1]],
            ]
        )
        expected_env_actions = np.array(
            [
                [[Action.NO_OP, 0, 0], [Action.UP, 0, 0]],
                [[Action.RIGHT, 0, 0], [Action.DOWN, 0, 0]],
                [[Action.LEFT, 0, 0], [Action.SAP, 0, 1]],
                [[Action.SAP, -1, 0], [Action.SAP, -2, -1]],
                [[Action.SAP, 0, 0], [Action.SAP, 22, 22]],
            ]
        )
        out = ActorCriticOut(
            main_log_probs=torch.empty(0),
            sap_log_probs=torch.empty(0),
            main_actions=main_actions,
            sap_actions=sap_actions,
            value=torch.empty(0),
        )
        assert np.array_equal(out.to_env_actions(unit_indices), expected_env_actions)

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            ((0, 0), 0),
            ((0, 1), 1),
            ((1, 0), 24),
            ((2, 2), 24 * 2 + 2),
            ((23, 23), 24 * 24 - 1),
        ],
    )
    def test_sap_action_projection(self, index: tuple[int], expected: int) -> None:
        action_tensor = torch.zeros((24, 24))
        action_tensor[index] = 1
        assert torch.argmax(action_tensor.flatten()) == expected
