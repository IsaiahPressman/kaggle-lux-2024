# Rux-AI-S3

### Setup instructions
1. [Install rust](https://www.rust-lang.org/tools/install) and add nightly toolchain for `rustfmt`:
   1. `rustup update nightly`
   2. `rustup component add rustfmt --toolchain nightly`
2. [Install rye](https://rye.astral.sh/guide/installation/)
3. Install maturin: `rye install maturin`
4. Install packages: `rye sync`
5. Activate venv: `. .venv/bin/activate`
6. Generate full-game test cases: `./generate_full_game_test_cases.sh`
7. If everything is working, `make prepare` should now run without errors


### Generating simulator test cases
- Energy field test cases can be generated using `generate_energy_node_test_case.py`.
There should preferably be at least one test case per possible node count value. 
(2, 4, and 6 as of writing)
- Full game test cases can be generated individually by running
`JAX_PLATFORMS=cpu luxai-s3 python/test_agent/main.py python/test_agent/main.py 
--output replay.json --replay.no-compressed-obs`
followed by `process_replay_for_tests.py --include_observations` on the generated replay.json, observations_0.json, and observations_1.json files.


### Playing a game with the local agent
`JAX_PLATFORMS=cpu luxai-s3 python/main.py python/main.py --output replay.json`

### Creating a submission file
1. Build the docker image: `docker build -t rux-ai-s3 .`
2. Assuming the above runs without errors, we can copy the compiled + tarred submission file from the image as follows:
```
id=$(docker create rux-ai-s3)
docker cp $id:/home/rux_ai_s3/submission.tar.gz .
docker rm -v $id
```

### Parameter meta-learning (as of 11-2) [source](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/src/luxai_s3/params.py)
Fixed parameters:
- max_units
- match_count_per_episode
- max_steps_in_match
- map_height
- map_width
- num_teams

Parameters provided at game start:
- unit_move_cost [1, 6)
- unit_sap_cost [30, 51)
- unit_sap_range [3, 8)
- unit_sensor_range [2, 5)

Parameters that must be learned:
- nebula_tile_vision_reduction [0, 4)
- nebula_tile_energy_reduction [0, 0, 10, 25]
- unit_sap_dropoff_factor [0.25, 0.5, 1]
- unit_energy_void_factor [0.0625, 0.125, 0.25, 0.375]
- nebula_tile_drift_speed [-0.05, -0.025, 0.025, 0.05]
- energy_node_drift_speed [0.01, 0.02, 0.03, 0.04, 0.05]
- energy_node_drift_magnitude [3, 6)
