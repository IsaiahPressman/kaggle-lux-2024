from rl.train_ppo import CONFIG_FILE, PPOConfig


def test_ppo_config_file() -> None:
    PPOConfig.from_file(CONFIG_FILE)
