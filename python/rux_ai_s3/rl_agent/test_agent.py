import numpy as np
import pytest
from luxai_s3.wrappers import LuxAIS3GymEnv

from rux_ai_s3.rl_agent.agent import (
    AGENT_CONFIG_FILE,
    TRAIN_CONFIG_FILE,
    Agent,
    AgentConfig,
)
from rux_ai_s3.rl_training.train_config import TrainConfig
from rux_ai_s3.utils import load_from_yaml


def test_agent_config_file() -> None:
    assert AGENT_CONFIG_FILE.is_file()
    load_from_yaml(AgentConfig, AGENT_CONFIG_FILE)


def test_train_config_file() -> None:
    assert TRAIN_CONFIG_FILE.is_file()
    load_from_yaml(TrainConfig, TRAIN_CONFIG_FILE)


class TestAgent:
    @pytest.mark.slow
    def test_init(self) -> None:
        lux_env = LuxAIS3GymEnv(numpy_output=True)
        _, info = lux_env.reset(seed=42)
        env_cfg = info["params"]
        for id_, player in enumerate(["player_0", "player_1"]):
            agent = Agent(player, env_cfg)
            assert agent.team_id == id_

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "player",
        ["player_0", "player_1"],
    )
    def test_act(self, player: str) -> None:
        lux_env = LuxAIS3GymEnv(numpy_output=True)
        obs, info = lux_env.reset(seed=42)
        env_cfg = info["params"]
        agent = Agent(player, env_cfg)
        for step in range(5):
            actions = agent.act(step, obs[player], 60)
            assert actions.shape == (env_cfg["max_units"], 3)
            obs, *_ = lux_env.step(
                {
                    player: actions,
                    self.get_opponent(player): np.zeros(
                        (env_cfg["max_units"], 3), dtype=np.int64
                    ),
                }
            )

    @staticmethod
    def get_opponent(player: str) -> str:
        team_id = int(player.split("_")[1])
        return f"player_{1 - team_id}"
