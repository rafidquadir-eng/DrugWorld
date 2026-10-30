#!/usr/bin/env bash
# D0 pilot, idr0119 vehicle wells: 8 workers (one per GPU), 3 HTTPS connections each (24 total).
cd "$(dirname "$0")/../.."
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src ~/dwvenv/bin/python -W ignore scripts/10_pilot.py --set idr0119 \
    --worker $g --nworkers 8 --streams 3 --cpus 14 > logs/pilot/idr0119_w$g.log 2>&1 &
done
wait
echo "pilot idr0119 done $(date -u +%FT%TZ)" >> logs/pilot/idr0119_done.txt
