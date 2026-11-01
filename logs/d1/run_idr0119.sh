#!/usr/bin/env bash
# D1 embedding, idr0119 (EBI HTTPS): 8 workers on GPUs 0-7, 3 connections each = 24 in total
# (EBI's per-client limit is about 24-32; reports/D0_THROUGHPUT.md), from this instance only.
# Fields are dealt to workers; a worker that exits non-zero is restarted (it skips the shards
# it has written) up to 3 times.
cd "$(dirname "$0")/../.."
for g in 0 1 2 3 4 5 6 7; do
  (
    for attempt in 1 2 3 4; do
      CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src OPENBLAS_NUM_THREADS=8 ~/dwvenv/bin/python -W ignore \
        scripts/20_embed.py --set idr0119 --worker $g --nworkers 8 --streams 3 --cpus 12 \
        >> logs/d1/idr0119_w$g.log 2>&1 && break
      echo "worker $g attempt $attempt exited non-zero $(date -u +%FT%TZ)" >> logs/d1/idr0119_w$g.log
    done
  ) &
done
wait
echo "d1 idr0119 done $(date -u +%FT%TZ)" >> logs/d1/idr0119_done.txt
