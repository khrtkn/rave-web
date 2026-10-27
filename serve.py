#!/usr/bin/env python3
"""Start the local static research demo. No inference or upload server is used."""
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parent
if not (ROOT/'outputs/rave-morph-lowlatency/web/models/organ-512/organ-causal-512.onnx').exists():
    raise SystemExit('Run python3 scripts/fetch-assets.py first to restore the licensed runtime assets.')
runpy.run_path(str(ROOT/'outputs/rave-morph-lowlatency/serve.py'),run_name='__main__')
