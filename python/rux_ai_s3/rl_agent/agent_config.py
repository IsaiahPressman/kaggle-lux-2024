from pathlib import Path

from pydantic import BaseModel

AGENT_CONFIG_FILE = Path(__file__).parent / "agent_config.yaml"


class AgentConfig(BaseModel):
    sample_main_actions: bool
    sample_sap_actions: bool
