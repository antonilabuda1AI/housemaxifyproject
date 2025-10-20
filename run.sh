#!/usr/bin/env bash
set -euo pipefail

echo "House Maxify — starting in development mode"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Launching Flask app on http://localhost:5000 ..."
python app.py

