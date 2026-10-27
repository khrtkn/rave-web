#!/usr/bin/env python3
"""Start the local static research demo. No inference or upload server is used."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from model_catalog import verify
try:
    verify()
except (OSError, ValueError, KeyError) as error:
    raise SystemExit(str(error))
runpy.run_path(str(ROOT/'outputs/rave-morph-lowlatency/serve.py'),run_name='__main__')
