"""Register locally built, verified models. No model is downloaded here."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'outputs/rave-morph-lowlatency/web/models'
SPECS = {
    'organ-512': ('organ-causal-512', 'organ-causal-512.onnx', 512,
                  '218518cb0f7c2670a9cd3973c3e2972632893f599deacc851998b372a8f8a22e'),
    'organ-1024': ('organ-causal-1024', 'organ-causal-1024.onnx', 1024,
                   '218518cb0f7c2670a9cd3973c3e2972632893f599deacc851998b372a8f8a22e'),
    'djembe': ('djembe-stateful-2048', 'djembe-stateful-2048.onnx', 2048,
               'ea94557a8d9b40f22a1dadbc93ab98f46144a68ce0bed1403a1558a2d1007f16'),
}

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def local_file(base, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or any(p in ('..', '.') for p in relative.parts) or '\\' in name or ':' in name:
        raise ValueError('Expected a local relative file path: ' + name)
    path = base.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError('Missing or escaped local file: ' + str(path))
    return path

def check_blob(path, descriptor):
    if path.stat().st_size != descriptor['bytes']:
        raise ValueError('File size mismatch: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != descriptor['sha256']:
        raise ValueError('SHA-256 mismatch: ' + str(path))

def load_model(models, directory):
    model_id, filename, frame_size, source_sha = SPECS[directory]
    base = models / directory
    if not base.resolve().is_relative_to(models.resolve()):
        raise ValueError('Model directory escapes models root')
    contract = read_json(local_file(base, 'contract.json'))
    if contract['id'] != model_id or contract['frameSize'] != frame_size or contract['sampleRate'] != 48000 or contract.get('validatedNative') is not True:
        raise ValueError('Unexpected model contract: ' + directory)
    if contract.get('sourceCheckpointSha256', contract.get('sourceSha256')) != source_sha:
        raise ValueError('Unexpected upstream model: ' + directory)
    check_blob(local_file(base, filename), contract)
    fixture = read_json(local_file(base, 'parity-fixture.json'))
    if fixture['modelSha256'] != contract['sha256'] or fixture['model'] != model_id:
        raise ValueError('Fixture/model mismatch: ' + directory)
    if fixture.get('allNoiseInputs') != 0 or fixture.get('latentGain') != 1 or fixture.get('latentBias') != 0:
        raise ValueError('Fixture must use neutral controls and zero noise')
    if directory.startswith('organ') and fixture.get('resetOutputAndAllStatesBitwiseIdentical') is not True:
        raise ValueError('Organ fixture reset did not pass')
    for key in ('input', 'expectedOutput'):
        check_blob(local_file(base, fixture[key]['file']), fixture[key])
    # Builders write these reports only after numerical checks. These are local
    # consistency checks, not signatures or a substitute for rerunning validation.
    reports = ['native-validation.json'] if directory.startswith('organ') else ['candidate-validation.json']
    if directory == 'organ-512':
        reports += ['control-validation.json', 'source-forward-audit.json']
    for name in reports:
        report = read_json(local_file(base, name))
        if report.get('passed') is not True:
            raise ValueError('Model verification did not pass: ' + str(base / name))
        report_sha = report.get('onnxSha256', report.get('sha256'))
        if report_sha is not None and report_sha != contract['sha256']:
            raise ValueError('Verification/model mismatch: ' + directory)
    if directory.startswith('organ'):
        native = read_json(base / 'native-validation.json')
        if native['onnxBytes'] != contract['bytes'] or native['checkpointSha256'] != source_sha or native['configSha256'] != '975e0e8615204af6c9f8aa901275ac634d70f60011c898dbc103e1e4fb09ccb0' or native['allLearnedStateLoaded'] is not True:
            raise ValueError('Native/source mismatch: ' + directory)
        if directory == 'organ-1024':
            source = load_model(models, 'organ-512')
            if native['source512ModelSha256'] != source['sha256'] or native['all512InitializersBitwiseIdentical'] is not True:
                raise ValueError('1024 variant does not match its 512 source')
    else:
        candidate = read_json(base / 'candidate-validation.json')
        audit = read_json(local_file(base, 'noise-source-audit.json'))
        if candidate['sourceSha256'] != source_sha or candidate['allOriginalInitializersByteIdentical'] is not True or candidate['resetOutputAndAllStateTensorsBitwiseIdentical'] is not True or candidate['everyAudioAndStateOutputFinite'] is not True or audit['candidateSha256'] != contract['sha256'] or audit['sourceSha256'] != source_sha or audit['nativeFixtureReloadParity']['passed'] is not True:
            raise ValueError('Djembe source/audit mismatch')
    contract['url'] = directory + '/' + filename
    contract['parityFixture'] = directory + '/parity-fixture.json'
    contract['readOnlyComparison'] = False
    return contract

def register(models=MODELS):
    models = Path(models)
    catalog_path = models / 'catalog.json'
    previous = read_json(catalog_path) if catalog_path.exists() else {}
    # Validate all present supported builds before replacing a working catalog.
    entries = [load_model(models, name) for name in SPECS if (models / name / 'contract.json').exists()]
    if not entries:
        raise ValueError('No locally built models. Follow MODEL_SOURCES.md and run scripts/prepare-models.py --model organ (or djembe).')
    default = previous.get('defaultModel')
    if default not in {entry['id'] for entry in entries}:
        default = entries[0]['id']
    catalog = {'version': 1, 'defaultModel': default, 'models': entries,
               'note': 'Locally obtained and converted models. See MODEL_SOURCES.md for upstream attribution and terms.'}
    temporary = catalog_path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, catalog_path)
    return catalog

def verify(models=MODELS):
    models = Path(models)
    path = models / 'catalog.json'
    if not path.exists():
        raise ValueError('Models have not been prepared. Follow MODEL_SOURCES.md first.')
    catalog = read_json(path)
    if not catalog.get('models'):
        raise ValueError('No models registered. Run scripts/prepare-models.py --model organ (or djembe).')
    by_id = {spec[0]: name for name, spec in SPECS.items()}
    for entry in catalog['models']:
        if entry['id'] not in by_id or entry != load_model(models, by_id[entry['id']]):
            raise ValueError('Catalog differs from the verified local build. Run scripts/model_catalog.py to register again.')
    if catalog['defaultModel'] not in {entry['id'] for entry in catalog['models']}:
        raise ValueError('Default model is unavailable')
    return catalog

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=MODELS)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    try:
        result = verify(args.models_dir) if args.verify else register(args.models_dir)
        print('Verified local models: ' + ', '.join(m['id'] for m in result['models']))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error))
