import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import jax
import tqdm
from luxai_s3 import LuxAIS3Env
from luxai_s3.state import EnvState


@dataclass
class UserArgs:
    seed: int | None
    node_count: int | None

    @classmethod
    def from_argparse(cls) -> "UserArgs":
        parser = argparse.ArgumentParser(
            description="Generate energy node test case from seed"
        )
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--node_count", type=int, default=None)
        args = parser.parse_args()
        return UserArgs(**vars(args))


def main() -> None:
    args = UserArgs.from_argparse()
    if args.seed is not None:
        seed = args.seed
    elif args.node_count is not None:
        seed = get_seed_from_node_count(args.node_count)
    else:
        seed = get_random_seed()

    env = LuxAIS3Env()
    _, state = env.reset(jax.random.key(seed))
    path = get_path(seed)
    dump_test_case(state, seed, path)
    print(f"Saved test case for seed {seed} to: {path}")


def get_seed_from_node_count(node_count: int) -> int:
    env = LuxAIS3Env()
    seed = get_random_seed()
    _, state = env.reset(jax.random.key(seed))
    with tqdm.tqdm() as prog_bar:
        while sum(state.energy_nodes_mask) != node_count:
            seed = get_random_seed()
            _, state = env.reset(jax.random.key(seed))
            prog_bar.update()

    return seed


def get_random_seed() -> int:
    return random.randint(0, 2**32 - 1)


def get_path(seed: int) -> Path:
    return Path(f"get_energy_field_{seed}.json")


def dump_test_case(state: EnvState, seed: int, path: Path) -> None:
    to_dump = {
        "seed": seed,
        "energy_nodes": state.energy_nodes[state.energy_nodes_mask].tolist(),
        "energy_node_fns": state.energy_node_fns[state.energy_nodes_mask].tolist(),
        "energy_field": state.map_features.energy.tolist(),
    }
    with open(path, "w") as f:
        json.dump(to_dump, f)


if __name__ == "__main__":
    main()
