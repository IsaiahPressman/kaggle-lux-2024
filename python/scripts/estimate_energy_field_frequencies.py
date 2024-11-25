import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import jax
import tqdm
from luxai_s3 import LuxAIS3Env

N_SAMPLES: Final[int] = 1_000_000


@dataclass
class UserArgs:
    seed: int | None
    batch_size: int

    @classmethod
    def from_argparse(cls) -> "UserArgs":
        parser = argparse.ArgumentParser(
            description="Estimate energy field frequencies with seed"
        )
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=100)
        args = parser.parse_args()
        return UserArgs(**vars(args))


@dataclass
class Frequency:
    energy: int
    count: int
    frequency: float


def main() -> None:
    args = UserArgs.from_argparse()
    seed = get_random_seed() if args.seed is None else args.seed
    energy_field_frequencies = estimate_energy_field_frequencies(seed, args.batch_size)
    print(
        f"Estimated the following energy field frequencies with seed {seed}:\n"
        f"{pretty_format_frequencies(energy_field_frequencies)}"
    )
    path = get_path()
    dump_frequencies(energy_field_frequencies, seed, path)
    print(f"Saved energy field frequencies for seed {seed} to: {path}")


def get_random_seed() -> int:
    return random.randint(0, 2**32 - 1)


def estimate_energy_field_frequencies(seed: int, batch_size: int) -> list[Frequency]:
    all_keys = jax.random.split(
        jax.random.key(seed), (N_SAMPLES // batch_size, batch_size)
    )
    env = LuxAIS3Env(auto_reset=False)
    frequencies: Counter[int] = Counter()
    for key_batch in tqdm.tqdm(all_keys, desc="Generating energy field batches"):
        _, state = jax.vmap(env.reset)(key_batch)
        frequencies.update(v for v in state.map_features.energy.flatten().tolist())

    result = []
    for energy, count in sorted(frequencies.items()):
        result.append(
            Frequency(
                energy=energy,
                count=count,
                frequency=count / frequencies.total(),
            )
        )

    return result


def pretty_format_frequencies(frequencies: list[Frequency]) -> str:
    return "\n".join(
        f"{freq.energy:>4}: {freq.frequency * 100:>5.2f}%" for freq in frequencies
    )


def get_path() -> Path:
    return Path("energy_field_frequencies.json").absolute()


def dump_frequencies(frequencies: list[Frequency], seed: int, path: Path) -> None:
    to_dump = {
        "seed": seed,
        "total_count": sum(f.count for f in frequencies),
        "energy_deltas": [f.energy for f in frequencies],
        "counts": [f.count for f in frequencies],
        "frequencies": [f.frequency for f in frequencies],
    }
    with open(path, "w") as f:
        json.dump(to_dump, f)


if __name__ == "__main__":
    main()
