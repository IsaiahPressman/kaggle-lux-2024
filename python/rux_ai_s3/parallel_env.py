import functools
import itertools
from collections import deque
from typing import Any

import jax
import numpy as np
import numpy.typing as npt
from luxai_s3.params import EnvParams
from luxai_s3.state import gen_map
from pydantic import field_validator

from rux_ai_s3._lowlevel import ParallelEnv as LowLevelEnv
from rux_ai_s3._lowlevel import RewardSpace

from .types import Obs, ParallelEnvOut

__all__ = [
    "EnvConfig",
    "ParallelEnv",
    "RewardSpace",
]


class EnvConfig:
    n_envs: int
    frame_stack_len: int
    reward_space: RewardSpace

    @field_validator("reward_space", mode="before")
    @classmethod
    def _validate_reward_space(cls, reward_space: RewardSpace | str) -> RewardSpace:
        if isinstance(reward_space, RewardSpace):
            return reward_space

        return RewardSpace.from_str(reward_space)


class ParallelEnv:
    def __init__(
        self,
        n_envs: int,
        reward_space: RewardSpace,
        frame_stack_len: int,
    ) -> None:
        self.n_envs = n_envs
        self.reward_space = reward_space
        self.frame_stack_len = frame_stack_len
        fixed_params = EnvParams()

        self._random_state = jax.random.key(seed=42)
        self._env = LowLevelEnv(n_envs, reward_space)
        self._raw_gen_map_vmapped = jax.vmap(
            functools.partial(
                gen_map,
                params=None,
                map_type=fixed_params.map_type,
                map_height=fixed_params.map_height,
                map_width=fixed_params.map_width,
                max_energy_nodes=fixed_params.max_energy_nodes,
                max_relic_nodes=fixed_params.max_relic_nodes,
                relic_config_size=fixed_params.relic_config_size,
            )
        )
        self._last_out: ParallelEnvOut = self._make_empty_out()
        self._frame_history: deque[Obs] = self._make_empty_frame_history()
        self.hard_reset()

    @property
    def last_out(self) -> ParallelEnvOut:
        return self._last_out

    def get_frame_stacked_obs(self) -> Obs:
        return Obs.concatenate_frame_history(self._frame_history)

    def _gen_maps(self, n_maps: int) -> dict[str, Any]:
        self._random_state, *subkeys = jax.random.split(self._random_state, n_maps + 1)
        return self._raw_gen_map_vmapped(subkeys)

    def _make_empty_out(self) -> ParallelEnvOut:
        return ParallelEnvOut.from_raw(self._env.get_empty_outputs())

    def _make_empty_frame_history(self) -> deque[Obs]:
        return deque(
            [self._make_empty_out().obs for _ in range(self.frame_stack_len)],
            maxlen=self.frame_stack_len,
        )

    def _soft_reset(self) -> None:
        reset_count: int = self._last_out.done.sum().item()
        if reset_count == 0:
            return

        new_map_dict = self._gen_maps(reset_count)
        self._env.soft_reset(
            output_arrays=self._last_out,
            tile_type=np.asarray(new_map_dict["map_features"].tile_type),
            energy_nodes=np.asarray(new_map_dict["energy_nodes"]),
            energy_node_fns=np.asarray(new_map_dict["energy_node_fns"]),
            energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"]),
            relic_nodes=np.asarray(new_map_dict["relic_nodes"]),
            relic_node_configs=np.asarray(new_map_dict["relic_node_configs"]),
            relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"]),
        )

    def _update_frame_history(self) -> None:
        self._frame_history.append(self._last_out.obs)
        new_match_envs = self._env.get_new_match_envs()
        if not new_match_envs:
            return

        # Zero out all but the latest observation
        for obs in itertools.islice(self._frame_history, len(self._frame_history) - 1):
            for array in obs:
                array[new_match_envs] = 0

    def hard_reset(self) -> None:
        self._env.terminate_envs(list(range(self.n_envs)))
        self._last_out = self._make_empty_out()
        self._frame_history = self._make_empty_frame_history()
        self._soft_reset()
        self._update_frame_history()

    def step(self, actions: npt.NDArray[np.int64]) -> None:
        self._last_out = ParallelEnvOut.from_raw(self._env.par_step(actions))
        self._soft_reset()
        self._update_frame_history()

    @classmethod
    def from_config(cls, config: EnvConfig) -> "ParallelEnv":
        return ParallelEnv(
            n_envs=config.n_envs,
            reward_space=config.reward_space,
            frame_stack_len=config.frame_stack_len,
        )
