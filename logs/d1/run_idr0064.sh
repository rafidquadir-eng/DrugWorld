#!/usr/bin/env bash
# D1 embedding, idr0064 (BIA S3, OME-Zarr): 4 workers on GPUs 0-3, 32 S3 connections each.
cd "$(dirname "$0")/../.."
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src ~/dwvenv/bin/python -W ignore scripts/20_embed.py --set idr0064 \
    --worker $g --nworkers 4 --streams 32 --cpus 4 > logs/d1/idr0064_w$g.log 2>&1 &
done
wait
echo "d1 idr0064 done $(date -u +%FT%TZ)" >> logs/d1/idr0064_done.txt
