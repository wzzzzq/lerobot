#!/bin/bash
# Convert RoboTwin dataset from v2.1 to v3.0 format
# Set TMPDIR to avoid /tmp disk space issues

export TMPDIR=/pfs/pfs-ilWc5D/ziqianwang/tmp
export http_proxy=http://172.16.0.136:18000 
export https_proxy=http://172.16.0.136:18000

echo "TMPDIR is set to: $TMPDIR"

python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id=name/aloha_agix_sim \
    --root=/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin \
    --push-to-hub=false \
    --force-conversion \
    --resume

echo "Conversion completed!"
