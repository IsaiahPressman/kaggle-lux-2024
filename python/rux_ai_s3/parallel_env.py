import functools
import itertools
from collections import deque
from typing import Any

import jax
import numpy as np
from luxai_s3.params import EnvParams
from luxai_s3.state import gen_map
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from rux_ai_s3.lowlevel import ParallelEnv as LowLevelEnv
from rux_ai_s3.lowlevel import RewardSpace, SapMasking
from rux_ai_s3.types import ActionArray, FrameStackedObs

from .types import Obs, ParallelEnvOut
from .utils import GEN_MAP_MOCK_PARAMS


class EnvConfig(BaseModel):
    n_envs: int
    frame_stack_len: int
    sap_masking: SapMasking
    reward_space: RewardSpace
    jax_device: str

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("reward_space", mode="before")
    @classmethod
    def _validate_reward_space(cls, reward_space: RewardSpace | str) -> RewardSpace:
        if isinstance(reward_space, str):
            return RewardSpace.from_str(reward_space)

        return reward_space

    @field_serializer("reward_space")
    def _serialize_reward_space(self, reward_space: RewardSpace) -> str:
        return str(reward_space)

    @field_validator("sap_masking", mode="before")
    @classmethod
    def _validate_sap_masking(cls, sap_masking: SapMasking | str) -> SapMasking:
        if isinstance(sap_masking, str):
            return SapMasking.from_str(sap_masking)

        return sap_masking

    @field_serializer("sap_masking")
    def _serialize_sap_masking(self, sap_masking: SapMasking) -> str:
        return str(sap_masking)

    @field_validator("jax_device")
    @classmethod
    def _validate_jax_device(cls, jax_device: str) -> str:
        try:
            cls._jax_device_from_str(jax_device)
        except Exception:
            raise ValueError(f"Invalid jax_device: '{jax_device}'") from None

        return jax_device

    def get_jax_device(self) -> jax.Device:
        return self._jax_device_from_str(self.jax_device)

    @staticmethod
    def _jax_device_from_str(jax_device: str) -> jax.Device:
        device_type, device_id = jax_device.split(":")
        return jax.devices(device_type)[int(device_id)]


class ParallelEnv:
    def __init__(
        self,
        n_envs: int,
        frame_stack_len: int,
        sap_masking: SapMasking,
        reward_space: RewardSpace,
        jax_device: jax.Device,
    ) -> None:
        self.n_envs = n_envs
        self.frame_stack_len = frame_stack_len
        self.reward_space = reward_space
        self.jax_device = jax_device

        fixed_params = EnvParams()
        with jax.default_device(self.jax_device):
            self._random_state = jax.random.key(seed=42)
            self._raw_gen_map_vmapped = jax.vmap(
                functools.partial(
                    gen_map,
                    params=GEN_MAP_MOCK_PARAMS,
                    map_type=fixed_params.map_type,
                    map_height=fixed_params.map_height,
                    map_width=fixed_params.map_width,
                    max_energy_nodes=fixed_params.max_energy_nodes,
                    max_relic_nodes=fixed_params.max_relic_nodes,
                    relic_config_size=fixed_params.relic_config_size,
                )
            )

        self._env = LowLevelEnv(n_envs, sap_masking, reward_space)
        self._last_out: ParallelEnvOut = self._make_empty_out()
        self._frame_history: deque[Obs] = self._make_empty_frame_history()
        self.hard_reset()

    @property
    def last_out(self) -> ParallelEnvOut:
        return self._last_out

    def get_frame_stacked_obs(self) -> FrameStackedObs:
        return FrameStackedObs.from_frame_history(list(self._frame_history), axis=2)

    def _gen_maps(self, n_maps: int) -> dict[str, Any]:
        with jax.default_device(self.jax_device):
            self._random_state, subkey = jax.random.split(self._random_state)
            return self._raw_gen_map_vmapped(jax.random.split(subkey, num=n_maps))

    def _make_empty_out(self) -> ParallelEnvOut:
        empty_out = ParallelEnvOut.from_raw(self._env.get_empty_outputs())
        return empty_out._replace(done=np.ones_like(empty_out.done))

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
            relic_spawn_schedule=np.asarray(new_map_dict["relic_spawn_schedule"]),
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

    def step(self, actions: ActionArray) -> None:
        self._last_out = ParallelEnvOut.from_raw(self._env.par_step(actions))
        self._soft_reset()
        self._update_frame_history()

    def run_all_jit_compilations(self) -> None:
        for i in range(1, self.n_envs):
            self._gen_maps(i)

    @classmethod
    def from_config(cls, config: EnvConfig) -> "ParallelEnv":
        return ParallelEnv(
            n_envs=config.n_envs,
            frame_stack_len=config.frame_stack_len,
            sap_masking=config.sap_masking,
            reward_space=config.reward_space,
            jax_device=config.get_jax_device(),
        )
