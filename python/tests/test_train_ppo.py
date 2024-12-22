from rl.train_ppo import CONFIG_FILE, PPOConfig
from rux_ai_s3.utils import load_from_yaml


def test_ppo_config_file() -> None:
    load_from_yaml(PPOConfig, CONFIG_FILE)
