from pydantic import BaseModel

from rux_ai_s3.lowlevel import SapMasking
from rux_ai_s3.models.build import (
    ActorCriticConfigT,
)
from rux_ai_s3.parallel_env import EnvConfig


class TrainConfig(BaseModel):
    env_config: EnvConfig
    rl_model_config: ActorCriticConfigT

    @property
    def frame_stack_len(self) -> int:
        return self.env_config.frame_stack_len

    @property
    def sap_masking(self) -> SapMasking:
        return self.env_config.sap_masking
