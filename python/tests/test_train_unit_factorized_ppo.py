import json

from rl.train_unit_factorized_ppo import CONFIG_FILE, UnitFactorizedPPOConfig
from rux_ai_s3.utils import load_from_yaml


def test_config_file() -> None:
    load_from_yaml(UnitFactorizedPPOConfig, CONFIG_FILE)


def test_config_serializable() -> None:
    cfg = load_from_yaml(UnitFactorizedPPOConfig, CONFIG_FILE)
    json.dumps(cfg.model_dump())
