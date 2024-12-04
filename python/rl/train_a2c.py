import itertools
from collections.abc import Generator
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from rux_2024.parallel_env import ParallelEnv, RewardSpace

FILE: Final[Path] = Path(__file__)
CONFIG_FILE: Final[Path] = FILE.parent / "config" / "a2c.yaml"


class A2CConfig(BaseModel):
    # Training config
    max_batches: int | None
    lr: float
    steps_per_batch: int
    n_envs: int
    reward_space: RewardSpace

    # Model config
    d_model: int
    n_layers: int

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    def iter_batches(self) -> Generator[int, None, None]:
        if self.max_batches is None:
            return (i for i in itertools.count(1))

        return (i for i in range(1, self.max_batches + 1))

    @field_validator("reward_space", mode="before")
    @classmethod
    def _validate_reward_space(cls, reward_space: RewardSpace | str) -> RewardSpace:
        if isinstance(reward_space, RewardSpace):
            return reward_space

        return RewardSpace.from_str(reward_space)

    @classmethod
    def from_file(cls, path: Path) -> "A2CConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        return A2CConfig(**data)


def main() -> None:
    cfg = A2CConfig.from_file(CONFIG_FILE)
    env = ParallelEnv(
        n_envs=cfg.n_envs,
        reward_space=cfg.reward_space,
    )
    for batch_count in cfg.iter_batches():

    # TODO: Left off here
    raise NotImplementedError


if __name__ == "__main__":
    main()
