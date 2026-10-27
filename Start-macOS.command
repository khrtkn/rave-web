#!/bin/bash
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.9 or later is required. Install it from https://www.python.org/downloads/ ."
  read -r -p "Press Enter to close."
  exit 1
fi
python3 launch.py || {
  read -r -p "Could not start. Press Enter to close."
  exit 1
}
