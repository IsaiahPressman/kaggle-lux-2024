#!/bin/bash

for i in $(seq 1 20)
do
  echo "Running game #$i"
  JAX_PLATFORMS=cpu luxai-s3 python/main.py python/main.py \
    --output replay.json --replay.no-compressed-obs &&
    python python/scripts/process_replay_for_tests.py ./ ||
    exit 1
done

mv processed_replay_*.json ./src/feature_engineering/test_data/
rm replay.json
