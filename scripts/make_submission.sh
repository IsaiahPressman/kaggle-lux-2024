#!/bin/bash

archive_name="${1:-submission}".tar.gz
echo "Saving submission as $archive_name"

docker build --no-cache -t rux-ai-s3 . &&
  id=$(docker create rux-ai-s3) &&
  docker cp "$id":/home/rux_ai_s3/submission.tar.gz "$archive_name" &&
  docker rm -v "$id" &&
  echo "Saved submission to $archive_name" ||
  exit 1
