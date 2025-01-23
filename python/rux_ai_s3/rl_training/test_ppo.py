import pytest
import torch

from . import ppo


def test_merge_log_probs() -> None:
    main_probs = torch.tensor(
        [
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.0, 0.1, 0.0],
            [0.2, 0.3, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0],
            [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        ]
    )
    # Usually these would be of shape (24 * 24,)
    sap_probs = torch.tensor(
        [
            [0.5, 0.4, 0.1],
            [0.2, 0.2, 0.6],
            [0.2, 0.2, 0.6],
        ]
    )
    units_mask = torch.tensor([True, True, False])
    ones_like_sum = torch.ones_like(main_probs.sum(dim=-1))
    assert torch.allclose(main_probs.sum(dim=-1), ones_like_sum)
    assert torch.allclose(sap_probs.sum(dim=-1), ones_like_sum)

    main_log_probs = main_probs.log()
    sap_log_probs = sap_probs.log()
    merged_log_probs = ppo.merge_log_probs(main_log_probs, sap_log_probs, units_mask)
    merged_probs = merged_log_probs.exp()
    assert torch.allclose(
        merged_probs.sum(dim=-1)[units_mask], ones_like_sum[units_mask]
    )
    expected_merged_probs = torch.tensor(
        [
            [0.2, 0.2, 0.2, 0.2, 0.1] + [0.05, 0.04, 0.01],  # noqa: RUF005
            [0.2, 0.3, 0.3, 0.1, 0.1] + [0.0, 0.0, 0.0],  # noqa: RUF005
            [0.125, 0.125, 0.125, 0.125, 0.125] + [0.0, 0.0, 0.0],  # noqa: RUF005
        ]
    )
    assert torch.allclose(merged_probs, expected_merged_probs)


def test_merge_log_probs_fails() -> None:
    main_probs = torch.tensor(
        [
            [0.2, 0.2, 0.2, 0.2, 0.0, 0.1, 0.1, 0.0],
            [0.2, 0.3, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0],
        ]
    )
    sap_probs = torch.tensor(
        [
            [0.5, 0.4, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    units_mask = torch.tensor([True, True])
    with pytest.raises(ValueError, match="Got multiple unmasked sap actions"):
        ppo.merge_log_probs(main_probs.log(), sap_probs.log(), units_mask)
