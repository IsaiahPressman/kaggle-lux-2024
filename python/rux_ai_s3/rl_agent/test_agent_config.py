from rux_ai_s3.rl_agent.agent_config import AGENT_CONFIG_FILE, AgentConfig
from rux_ai_s3.utils import load_from_yaml


def test_agent_config_file() -> None:
    assert AGENT_CONFIG_FILE.is_file()
    load_from_yaml(AgentConfig, AGENT_CONFIG_FILE)
