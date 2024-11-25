import functools
from typing import Any

import jax
import numpy as np
import numpy.typing as npt
from luxai_s3.params import EnvParams
from luxai_s3.state import gen_map

from rux_2024._lowlevel import ParallelEnv as LowLevelEnv

from .types import ParallelEnvOut


class ParallelEnv:
    def __init__(
        self,
        n_envs: int,
        auto_reset: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_envs = n_envs
        self.auto_reset = auto_reset
        fixed_params = EnvParams()

        self._random_state = jax.random.key(seed)
        self._env = LowLevelEnv(n_envs)
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
        self._last_out = ParallelEnvOut.from_raw(self._env.get_empty_outputs())
        self.hard_reset()

    @property
    def last_out(self) -> ParallelEnvOut:
        return self._last_out

    def _gen_maps(self, n_maps: int) -> dict[str, Any]:
        self._random_state, *subkeys = jax.random.split(self._random_state, n_maps + 1)
        return self._raw_gen_map_vmapped(subkeys)

    def hard_reset(self) -> None:
        self._env.terminate_envs(list(range(self.n_envs)))
        self._last_out = ParallelEnvOut.from_raw(self._env.get_empty_outputs())
        self.soft_reset()

    def soft_reset(self) -> None:
        needs_reset: int = self.last_out.done.sum().item()
        if needs_reset == 0:
            return

        new_map_dict = self._gen_maps(needs_reset)
        self._env.soft_reset(
            obs_arrays=self._last_out,
            tile_type=np.asarray(new_map_dict["map_features"].tile_type),
            energy_nodes=np.asarray(new_map_dict["energy_nodes"]),
            energy_node_fns=np.asarray(new_map_dict["energy_node_fns"]),
            energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"]),
            relic_nodes=np.asarray(new_map_dict["relic_nodes"]),
            relic_node_configs=np.asarray(new_map_dict["relic_node_configs"]),
            relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"]),
        )

    def step(self, actions: npt.NDArray[np.int_]) -> None:
        self._last_out = ParallelEnvOut.from_raw(self._env.par_step(actions))
        if self.auto_reset:
            self.soft_reset()
