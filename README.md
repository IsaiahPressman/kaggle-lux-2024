# Rux-AI-S3

This repository contains all the code used for team Frog Parade's gold medal approach for [Lux AI Season 3](https://www.kaggle.com/competitions/lux-ai-season-3), hosted on Kaggle.
The full write-up describing the approach can be found in [write-up.md](https://github.com/IsaiahPressman/kaggle-lux-2024/blob/main/write-up.md).

### Setup instructions
1. [Install rust](https://www.rust-lang.org/tools/install) and add nightly toolchain for `rustfmt`:
   1. `rustup update nightly`
   2. `rustup component add rustfmt --toolchain nightly`
2. [Install rye](https://rye.astral.sh/guide/installation/)
3. Install maturin: `rye install maturin`
4. Install packages: `rye sync`
5. Activate venv: `. .venv/bin/activate`
6. Generate full-game test cases: `./scripts/generate_full_game_test_cases.sh`
   - Optionally, generate additional memory module test cases (requires decently strong agent):
   `make build && ./scripts/generate_memory_test_cases.sh`
7. If everything is working, `make prepare` should now run without errors


### Generating simulator test cases
- Energy field test cases can be generated using `generate_energy_node_test_case.py`.
There should preferably be at least one test case per possible node count value. 
(2, 4, and 6 as of writing)
- Full game test cases can be generated individually by running
`JAX_PLATFORMS=cpu luxai-s3 python/test_agent/main.py python/test_agent/main.py 
--output replay.json --replay.no-compressed-obs`
followed by `process_replay_for_tests.py --include_observations` on the generated 
replay.json, observations_0.json, and observations_1.json files.


### Playing a game with the local agent
`JAX_PLATFORMS=cpu luxai-s3 python/main.py python/main.py --output replay.json`


### Creating a submission file
`./scripts/make_submission.sh`
