from pydantic import BaseModel

from rux_ai_s3.models.actor_critic import ActorCriticConfig
from rux_ai_s3.parallel_env import EnvConfig


class TrainConfig(BaseModel):
    env_config: EnvConfig
    rl_model_config: ActorCriticConfig

    @property
    def frame_stack_len(self) -> int:
        return self.env_config.frame_stack_len
