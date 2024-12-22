from rl.train_ppo import CONFIG_FILE as PPO_CONFIG_FILE

from rux_ai_s3.rl_agent.train_config import TRAIN_CONFIG_FILE, TrainConfig
from rux_ai_s3.utils import load_from_yaml


def test_load_train_config_from_ppo() -> None:
    load_from_yaml(TrainConfig, PPO_CONFIG_FILE)


def test_train_config_file() -> None:
    assert TRAIN_CONFIG_FILE.is_file()
    load_from_yaml(TrainConfig, TRAIN_CONFIG_FILE)
