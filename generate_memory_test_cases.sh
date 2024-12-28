#!/bin/bash

for seed in $(seq 0 19)
do
  echo "Running seed: $seed"
  JAX_PLATFORMS=cpu luxai-s3 python/main.py python/main.py \
    --output replay.json --replay.no-compressed-obs --seed "$seed" &&
    python python/scripts/process_replay_for_tests.py ./ ||
    exit 1
done

mv processed_replay_*.json ./src/feature_engineering/test_data/
rm replay.json
