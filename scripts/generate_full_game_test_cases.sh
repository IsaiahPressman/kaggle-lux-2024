#!/bin/bash

declare -a seeds=("0" "6155879" "4086850" "2462211601")

for seed in "${seeds[@]}"
do
  echo "Running seed: $seed"
  JAX_PLATFORMS=cpu luxai-s3 python/test_agent/main.py python/test_agent/main.py \
    --output replay.json --replay.no-compressed-obs --seed "$seed" &&
    python python/scripts/process_replay_for_tests.py ./ --include_observations ||
    exit 1
done

mv processed_replay_*.json ./src/rules_engine/test_data/processed_replays/
rm observations_0.json observations_1.json replay.json
