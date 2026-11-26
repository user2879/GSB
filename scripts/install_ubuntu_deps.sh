#!/usr/bin/env bash
# For a normal root-enabled WSL Ubuntu shell. The validation container used by
# this delivery had Python 3.11.0rc1, NumPy 1.26.4 and sqlite3 already present;
# Tk 8.6.12 was made available through the runtime's local Tk shim.
set -euxo pipefail

sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  software-properties-common \
  sqlite3 \
  tk8.6 \
  libtk8.6 \
  libtcl8.6

# Ubuntu 22.04's default python3-tk targets distro Python 3.10. Add deadsnakes
# only if the python3.11-tk candidate is unavailable in your configured repos.
if ! apt-cache policy python3.11-tk | grep -q 'Candidate: [^()]'; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
fi
sudo apt-get install -y python3.11 python3.11-tk python3.11-venv python3-numpy
python3.11 -m pip install -r requirements.txt
