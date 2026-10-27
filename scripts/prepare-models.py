#!/usr/bin/env python3
"""Obtain an original model from its provider, convert, validate, then register."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from model_catalog import MODELS, ROOT, register

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('organ', 'djembe'), required=True)
    parser.add_argument('--work-dir', type=Path, help='Source downloads and intermediate arrays (default: work/MODEL-build)')
    parser.add_argument('--models-dir', type=Path, default=MODELS, help='Generated model parent directory')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('Use Python 3.12 for model conversion; see MODEL_SOURCES.md.')
    if sys.flags.optimize:
        parser.error('Do not use -O or PYTHONOPTIMIZE: numerical validation requires assertions.')
    modules = ['numpy', 'onnx', 'onnxruntime'] + (['torch', 'torchaudio', 'gin', 'pytorch_lightning'] if args.model == 'organ' else [])
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        parser.error('Missing dependencies: ' + ', '.join(missing) + '. Install this model\'s requirements in its separate venv; see MODEL_SOURCES.md.')
    work = (args.work_dir or ROOT / 'work' / (args.model + '-build')).resolve()
    models = args.models_dir.resolve()
    def run(script, *options):
        environment = dict(os.environ)
        environment.pop('PYTHONOPTIMIZE', None)
        subprocess.run([sys.executable, str(script), *map(str, options)], cwd=ROOT, env=environment, check=True)
    if args.model == 'organ':
        tools = ROOT / 'outputs/rave-morph-lowlatency/model-tools'
        run(tools / 'prepare_sources.py', '--work-dir', work)
        for script in ('build_organ.py', 'verify_organ.py', 'audit_reference.py', 'build_organ_1024.py'):
            run(tools / script, '--work-dir', work, '--models-dir', models)
    else:
        tools = ROOT / 'outputs/rave-morph/model-tools'
        source = work / 'djembe.onnx'
        run(tools / 'prepare_djembe_source.py', '--output', source)
        for script in ('build_djembe_candidate.py', 'audit_djembe_candidate.py'):
            run(tools / script, '--source', source, '--out', models / 'djembe')
    catalog = register(models)
    print('Ready: ' + ', '.join(model['id'] for model in catalog['models']))
    print('Start the browser app: python3 launch.py' if models == MODELS else 'Custom output registered at: ' + str(models / 'catalog.json'))

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
