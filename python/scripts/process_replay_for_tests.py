import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
from luxai_s3 import LuxAIS3Env
from luxai_s3.state import EnvState

REPLAY_FILENAME = "replay.json"
OBSERVATION_0_FILENAME = "observations_0.json"
OBSERVATION_1_FILENAME = "observations_1.json"


@dataclass
class UserArgs:
    path: Path
    include_observations: bool

    @classmethod
    def from_argparse(cls) -> "UserArgs":
        parser = argparse.ArgumentParser(
            description="Create merged replay file with energy node config "
            "and observation information"
        )
        parser.add_argument("path", type=Path)
        parser.add_argument("--include_observations", action="store_true")
        args = parser.parse_args()
        return UserArgs(**vars(args))


def main() -> None:
    args = UserArgs.from_argparse()
    with open(args.path / REPLAY_FILENAME) as f:
        replay = json.load(f)

    observations: list[dict[str, Any]] = []
    if args.include_observations:
        for filename in (OBSERVATION_0_FILENAME, OBSERVATION_1_FILENAME):
            with open(args.path / filename) as f:
                observations.append(json.load(f))

    seed = replay["metadata"]["seed"]
    env = LuxAIS3Env()
    rng_key = jax.random.key(seed)
    rng_key, reset_key = jax.random.split(rng_key)
    _, state = env.reset(reset_key)
    validate_seed(state, replay)

    replay["energy_node_fns"] = state.energy_node_fns[state.energy_nodes_mask].tolist()
    replay["player_observations"] = observations or None
    output_path = args.path / f"processed_replay_{seed}.json"
    with open(output_path, "w") as f:
        json.dump(replay, f)

    print(f"Saved processed replay with seed {seed} at {output_path}")


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
