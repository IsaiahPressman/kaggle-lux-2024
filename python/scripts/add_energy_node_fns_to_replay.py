import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
from luxai_s3 import LuxAIS3Env
from luxai_s3.state import EnvState


@dataclass
class UserArgs:
    path: Path

    @classmethod
    def from_argparse(cls) -> "UserArgs":
        parser = argparse.ArgumentParser(
            description="Edit replay file to add energy node config"
        )
        parser.add_argument("path", type=Path)
        args = parser.parse_args()
        return UserArgs(**vars(args))


def main() -> None:
    args = UserArgs.from_argparse()
    with open(args.path, "r") as f:
        replay = json.load(f)

    seed = replay["metadata"]["seed"]
    env = LuxAIS3Env()
    rng_key = jax.random.key(seed)
    rng_key, reset_key = jax.random.split(rng_key)
    _, state = env.reset(reset_key)
    validate_seed(state, replay)

    replay["energy_node_fns"] = state.energy_node_fns[state.energy_nodes_mask].tolist()
    with open(args.path, "w") as f:
        json.dump(replay, f)

    print(f"Updated replay with seed {seed} at {args.path}")


def validate_seed(state: EnvState, replay: dict[str, Any]) -> None:
    obs = replay["observations"][0]
    replay_energy_field = obs["map_features"]["energy"]
    if state.map_features.energy.tolist() != replay_energy_field:
        raise RuntimeError(
            f"{state.map_features.energy.tolist()} != {replay_energy_field}"
        )

    replay_tile_type = obs["map_features"]["tile_type"]
    if state.map_features.tile_type.tolist() != replay_tile_type:
        raise RuntimeError(
            f"{state.map_features.tile_type.tolist()} != {replay_tile_type}"
        )


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
