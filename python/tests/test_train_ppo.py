import json

from rl.train_ppo import DEFAULT_CONFIG_FILE, PPOConfig
from rux_ai_s3.utils import load_from_yaml


def test_config_file() -> None:
    load_from_yaml(PPOConfig, DEFAULT_CONFIG_FILE)


def test_config_serializable() -> None:
    cfg = load_from_yaml(PPOConfig, DEFAULT_CONFIG_FILE)
    json.dumps(cfg.model_dump())
