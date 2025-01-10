import functools
import itertools
import json
from typing import Any

import jax
import numpy as np
import numpy.typing as npt
import pytest
from luxai_s3.state import gen_map
from luxai_s3.wrappers import LuxAIS3GymEnv

from rux_ai_s3.lowlevel import (
    FeatureEngineeringEnv,
    ParallelEnv,
    RewardSpace,
)
from rux_ai_s3.types import FeatureEngineeringOut, ParallelEnvOut
from rux_ai_s3.utils import to_json

_MAP_SIZE = (24, 24)
_N_ENVS = 8
_FLOAT_FLAG = -1_000
_INT_FLAG = 1_000


def test_reward_space() -> None:
    for rs in RewardSpace.list():
        assert RewardSpace.from_str(str(rs)) == rs

    with pytest.raises(ValueError, match="Invalid RewardSpace"):
        RewardSpace.from_str("INVALID_REWARD_SPACE")


class TestParallelEnv:
    @pytest.mark.slow
    def test_get_new_match_and_new_game_envs(self) -> None:
        env = ParallelEnv(_N_ENVS, RewardSpace.FINAL_WINNER)
        new_map_dict = self.gen_map_vmapped()
        env.soft_reset(
            output_arrays=env.get_empty_outputs(),
            tile_type=np.asarray(new_map_dict["map_features"].tile_type),
            energy_nodes=np.asarray(new_map_dict["energy_nodes"]),
            energy_node_fns=np.asarray(new_map_dict["energy_node_fns"]),
            energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"]),
            relic_nodes=np.asarray(new_map_dict["relic_nodes"]),
            relic_node_configs=np.asarray(new_map_dict["relic_node_configs"]),
            relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"]),
        )
        actions = np.zeros((_N_ENVS, 2, 16, 3), dtype=int)
        env.par_step(actions)

        envs_behind_one_step = [0, 1, 4, 7]
        envs_ahead_one_step = [
            i for i in range(_N_ENVS) if i not in envs_behind_one_step
        ]
        env.terminate_envs(envs_behind_one_step)
        env.soft_reset(
            output_arrays=env.get_empty_outputs(),
            tile_type=np.asarray(new_map_dict["map_features"].tile_type)[
                : len(envs_behind_one_step)
            ],
            energy_nodes=np.asarray(new_map_dict["energy_nodes"])[
                : len(envs_behind_one_step)
            ],
            energy_node_fns=np.asarray(new_map_dict["energy_node_fns"])[
                : len(envs_behind_one_step)
            ],
            energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"])[
                : len(envs_behind_one_step)
            ],
            relic_nodes=np.asarray(new_map_dict["relic_nodes"])[
                : len(envs_behind_one_step)
            ],
            relic_node_configs=np.asarray(new_map_dict["relic_node_configs"])[
                : len(envs_behind_one_step)
            ],
            relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"])[
                : len(envs_behind_one_step)
            ],
        )
        assert env.get_new_match_envs() == envs_behind_one_step
        assert env.get_new_game_envs() == envs_behind_one_step

        for _ in range(100):
            env.par_step(actions)

        assert env.get_new_match_envs() == envs_ahead_one_step
        assert not env.get_new_game_envs()

        env.par_step(actions)
        assert env.get_new_match_envs() == envs_behind_one_step
        assert not env.get_new_game_envs()

    @pytest.mark.slow
    def test_step(self) -> None:
        env = ParallelEnv(_N_ENVS, RewardSpace.FINAL_WINNER)
        # Reset env
        new_map_dict = self.gen_map_vmapped()
        env_out = ParallelEnvOut.from_raw_validated(env.get_empty_outputs())
        env.soft_reset(
            output_arrays=env_out,
            tile_type=np.asarray(new_map_dict["map_features"].tile_type),
            energy_nodes=np.asarray(new_map_dict["energy_nodes"]),
            energy_node_fns=np.asarray(new_map_dict["energy_node_fns"]),
            energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"]),
            relic_nodes=np.asarray(new_map_dict["relic_nodes"]),
            relic_node_configs=np.asarray(new_map_dict["relic_node_configs"]),
            relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"]),
        )

        actions = np.zeros((_N_ENVS, 2, 16, 3), dtype=int)
        for _ in range(303):
            assert not np.any(env_out.done)
            assert env_out.stats is None
            env_out = ParallelEnvOut.from_raw_validated(env.par_step(actions))

        assert np.all(env_out.done)
        expected_reward = np.zeros((_N_ENVS, 2), dtype=float)
        expected_reward[:] = [1, -1]
        assert np.all(env_out.reward == expected_reward)
        assert env_out.stats is not None
        assert env_out.stats.scalar_stats
        assert env_out.stats.array_stats

    @pytest.mark.slow
    def test_soft_reset(self) -> None:
        env = ParallelEnv(_N_ENVS, RewardSpace.FINAL_WINNER)
        env_out = ParallelEnvOut.from_raw_validated(env.get_empty_outputs())
        self.fill_env_out(env_out)
        new_map_dict = self.gen_map_vmapped()
        env.soft_reset(
            output_arrays=env_out,
            tile_type=np.asarray(new_map_dict["map_features"].tile_type),
            energy_nodes=np.asarray(new_map_dict["energy_nodes"]),
            energy_node_fns=np.asarray(new_map_dict["energy_node_fns"]),
            energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"]),
            relic_nodes=np.asarray(new_map_dict["relic_nodes"]),
            relic_node_configs=np.asarray(new_map_dict["relic_node_configs"]),
            relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"]),
        )
        for obs_array in env_out.obs:
            assert np.all(obs_array != _FLOAT_FLAG)

        assert np.all(np.logical_not(env_out.action_info.main_mask))
        assert np.all(np.logical_not(env_out.action_info.sap_mask))
        assert np.all(env_out.action_info.unit_indices != _INT_FLAG)
        assert np.all(env_out.action_info.unit_energies != _FLOAT_FLAG)
        assert np.all(np.logical_not(env_out.action_info.units_mask))
        # Reward and done are left as-is after a soft reset
        assert np.all(env_out.reward == _FLOAT_FLAG)
        assert np.all(env_out.done)

        # Now try only resetting some envs
        reset_env_ids = [0, 1, 5]
        not_reset_env_ids = [i for i in range(_N_ENVS) if i not in reset_env_ids]
        env.terminate_envs(reset_env_ids)

        env_out = ParallelEnvOut.from_raw_validated(env.get_empty_outputs())
        self.fill_env_out(env_out)
        env_out.reward[not_reset_env_ids] = _FLOAT_FLAG - 1
        env_out.reward[reset_env_ids] = _FLOAT_FLAG
        env_out.done[not_reset_env_ids] = False
        env_out.done[reset_env_ids] = True

        new_map_dict = self.gen_map_vmapped()
        env.soft_reset(
            output_arrays=env_out,
            tile_type=np.asarray(new_map_dict["map_features"].tile_type)[
                : len(reset_env_ids)
            ],
            energy_nodes=np.asarray(new_map_dict["energy_nodes"])[: len(reset_env_ids)],
            energy_node_fns=np.asarray(new_map_dict["energy_node_fns"])[
                : len(reset_env_ids)
            ],
            energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"])[
                : len(reset_env_ids)
            ],
            relic_nodes=np.asarray(new_map_dict["relic_nodes"])[: len(reset_env_ids)],
            relic_node_configs=np.asarray(new_map_dict["relic_node_configs"])[
                : len(reset_env_ids)
            ],
            relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"])[
                : len(reset_env_ids)
            ],
        )

        for obs_array in env_out.obs:
            assert np.all(obs_array[reset_env_ids] != _FLOAT_FLAG)
            assert np.all(obs_array[not_reset_env_ids] == _FLOAT_FLAG)

        assert np.all(np.logical_not(env_out.action_info.main_mask[reset_env_ids]))
        assert np.all(env_out.action_info.main_mask[not_reset_env_ids])
        assert np.all(np.logical_not(env_out.action_info.sap_mask[reset_env_ids]))
        assert np.all(env_out.action_info.main_mask[not_reset_env_ids])
        assert np.all(env_out.action_info.unit_indices[reset_env_ids] != _INT_FLAG)
        assert np.all(env_out.action_info.unit_indices[not_reset_env_ids] == _INT_FLAG)
        assert np.all(env_out.action_info.unit_energies[reset_env_ids] != _FLOAT_FLAG)
        assert np.all(
            env_out.action_info.unit_energies[not_reset_env_ids] == _FLOAT_FLAG
        )
        assert np.all(np.logical_not(env_out.action_info.units_mask[reset_env_ids]))
        assert np.all(env_out.action_info.units_mask[not_reset_env_ids])

        # Reward and done are left as-is after a soft reset
        assert np.all(env_out.reward[reset_env_ids] == _FLOAT_FLAG)
        assert np.all(env_out.reward[not_reset_env_ids] == _FLOAT_FLAG - 1)
        assert np.all(env_out.done[reset_env_ids])
        assert np.all(np.logical_not(env_out.done[not_reset_env_ids]))

    @staticmethod
    def gen_map_vmapped() -> dict[str, Any]:
        map_width, map_height = _MAP_SIZE
        gen_map_vmapped = jax.vmap(
            functools.partial(
                gen_map,
                params=None,
                map_type=1,
                map_height=map_height,
                map_width=map_width,
                max_energy_nodes=6,
                max_relic_nodes=6,
                relic_config_size=5,
            )
        )
        return gen_map_vmapped(jax.random.split(jax.random.key(42), _N_ENVS))

    @staticmethod
    def fill_env_out(env_out: ParallelEnvOut) -> None:
        for obs_array in env_out.obs:
            obs_array[:] = _FLOAT_FLAG

        env_out.action_info.main_mask[:] = True
        env_out.action_info.sap_mask[:] = True
        env_out.action_info.unit_indices[:] = _INT_FLAG
        env_out.action_info.unit_energies[:] = _FLOAT_FLAG
        env_out.action_info.units_mask[:] = True
        env_out.reward[:] = _FLOAT_FLAG
        env_out.done[:] = True


class TestFeatureEngineeringEnv:
    @pytest.mark.slow
    def test_step(self) -> None:
        team_id = 1
        lux_env = LuxAIS3GymEnv(numpy_output=True)
        lux_obs, info = lux_env.reset(seed=42)
        fe_env = FeatureEngineeringEnv(team_id=team_id, env_params=info["params"])

        actions = np.zeros((16, 3), dtype=int)
        fe_out = FeatureEngineeringOut.from_raw(
            fe_env.step(self.json_dump_lux_obs(lux_obs, team_id), actions)
        )
        # Omit nontemporal_global_obs as it's expected that many of these
        # values will be non-zero
        for array in itertools.chain(
            *[
                fe_out.obs.temporal_spatial_obs,
                # The first feature - spawn point - will be non-zero
                fe_out.obs.nontemporal_spatial_obs[1:],
                fe_out.obs.temporal_global_obs,
            ],
            *fe_out.action_info,
        ):
            assert np.all(array == 0)

        truncated: dict[str, npt.NDArray[np.bool_]] = {}
        for _ in range(505):
            assert not any(t.item() for t in truncated.values())
            lux_obs, _, _, truncated, _ = lux_env.step(
                {"player_0": actions, "player_1": actions}
            )
            fe_out = FeatureEngineeringOut.from_raw(
                fe_env.step(self.json_dump_lux_obs(lux_obs, team_id), actions)
            )
            for array in itertools.chain.from_iterable(
                [
                    fe_out.obs.temporal_spatial_obs,
                    fe_out.obs.nontemporal_spatial_obs,
                    # Exclude temporal_global_obs - points will be 0
                    fe_out.obs.nontemporal_global_obs,
                ]
            ):
                assert np.any(array != 0)

        assert all(t for t in truncated.values())

    @staticmethod
    def json_dump_lux_obs(
        lux_obs: dict[str, Any],
        team_id: int,
    ) -> str:
        return json.dumps(to_json(lux_obs[f"player_{team_id}"]))
