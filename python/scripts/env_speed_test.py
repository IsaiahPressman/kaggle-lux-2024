import argparse
import os
import time

import jax
import numpy as np
import tqdm
from pydantic import BaseModel
from rux_ai_s3.constants import MAX_UNITS
from rux_ai_s3.lowlevel import RewardSpace, SapMasking, assert_release_build
from rux_ai_s3.parallel_env import ParallelEnv
from rux_ai_s3.types import ActionArray

JAX_CPU = jax.devices("cpu")[0]
FRAME_STACK_LEN = 10
WARMUP_STEPS = 100


class UserArgs(BaseModel):
    n_envs: int
    benchmark_steps: int
    allow_debug_build: bool

    @classmethod
    def from_argparse(cls) -> "UserArgs":
        parser = argparse.ArgumentParser(
            description="Plays a multi-game match between two trained models"
        )
        parser.add_argument("n_envs", type=int)
        parser.add_argument("benchmark_steps", type=int)
        parser.add_argument("-a", "--allow_debug_build", action="store_true")
        args = parser.parse_args()
        return UserArgs(**vars(args))


def main() -> None:
    user_args = UserArgs.from_argparse()
    if not user_args.allow_debug_build:
        assert_release_build()

    env = ParallelEnv(
        n_envs=user_args.n_envs,
        frame_stack_len=FRAME_STACK_LEN,
        sap_masking=SapMasking.POINT_TILES,
        reward_space=RewardSpace.FINAL_WINNER,
        jax_device=JAX_CPU,
    )
    warmup(env)
    benchmark(env, user_args.benchmark_steps)


def warmup(env: ParallelEnv) -> None:
    print("Warming up...")
    env.run_all_jit_compilations()
    for _ in range(WARMUP_STEPS):
        env.step(sample_actions(env.n_envs))


def benchmark(env: ParallelEnv, benchmark_steps: int) -> None:
    env.hard_reset()
    start_time = time.perf_counter()
    for _ in tqdm.trange(benchmark_steps, desc="Benchmarking"):
        env.step(sample_actions(env.n_envs))

    total_time = time.perf_counter() - start_time
    steps_per_second = benchmark_steps * env.n_envs / total_time
    print(
        f"Ran benchmark for {env.n_envs} envs using {os.cpu_count()} CPUs\n"
        f"Steps per second: {steps_per_second:.2f}"
    )


def sample_actions(n_envs: int) -> ActionArray:
    return np.zeros((n_envs, 2, MAX_UNITS, 3), dtype=np.int64)


if __name__ == "__main__":
    main()
