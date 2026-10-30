#!/usr/bin/env bash
# Plan section 11 OS freeze, idempotent: apt timers and unattended-upgrades disabled, apt
# services masked, NVIDIA/CUDA/Fabric Manager packages held, needrestart list-only. Never
# reboots and never runs apt upgrade. Prints the verification state.
set -u
echo "# OS freeze check $(date -u +%FT%TZ) on $(hostname)"
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service 2>&1 | grep -v '^$' || true
sudo systemctl mask apt-daily.service apt-daily-upgrade.service 2>&1 || true
PK=$(dpkg -l | awk '$1 ~ /^.i/ {print $2}' | grep -E '^(nvidia-|libnvidia-|cuda|libcudnn|libnccl|xserver-xorg-video-nvidia)' )
[ -n "$PK" ] && sudo apt-mark hold $PK > /dev/null
if [ -d /etc/needrestart ]; then
  echo "\$nrconf{restart} = 'l';" | sudo tee /etc/needrestart/conf.d/99-list-only.conf > /dev/null
fi
echo "## verify"
for u in apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service apt-daily.service apt-daily-upgrade.service; do
  echo "$u: $(systemctl is-enabled $u 2>&1) / $(systemctl is-active $u 2>&1)"; done
echo "apt timers listed: $(systemctl list-timers --all | grep -c -i apt)"
echo "held packages ($(apt-mark showhold | wc -l)):"; apt-mark showhold
echo "CUDA system packages (torch ships its own CUDA): $(dpkg -l | awk '$1 ~ /^.i/ {print $2}' | grep -c -E '^(cuda|libcudnn|libnccl)')"
echo "needrestart: $(cat /etc/needrestart/conf.d/*.conf 2>/dev/null | tr '\n' ' ')"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
nvidia-smi -q | grep -i -m1 "fabric" ; systemctl is-active nvidia-fabricmanager
