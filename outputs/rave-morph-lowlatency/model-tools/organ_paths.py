"""Shared, repo-relative paths for the public Organ conversion commands."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK = REPO_ROOT / 'work/rave-lowlatency'
DEFAULT_MODELS = Path(__file__).resolve().parents[1] / 'web/models'
SOURCE_REVISIONS = {
    'RAVE': 'ab7d40d6740130f0268debbc4b6633318357ea7d',
    'cached_conv': 'd2da85992d030dcf1dd8ebf318c3d69753729295',
}


def options(description, *, imported=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--work-dir', type=Path, default=DEFAULT_WORK,
                        help='Pinned source/checkpoint and intermediate data directory.')
    parser.add_argument('--models-dir', type=Path, default=DEFAULT_MODELS,
                        help='Output models root; organ-512/ and organ-1024/ are created here.')
    parser.add_argument('--update-catalog', action='store_true',
                        help='Explicitly add the exported model to models-dir/catalog.json. Default: leave it untouched.')
    args = parser.parse_known_args()[0] if imported else parser.parse_args()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.models_dir = args.models_dir.expanduser().resolve()
    return args


def verify_source_revisions(work):
    for name, expected in SOURCE_REVISIONS.items():
        source = work / name
        if not (source / '.git').exists():
            raise RuntimeError('Run prepare_sources.py with this --work-dir first: ' + name)
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'])
        if actual != expected or dirty:
            raise RuntimeError('Pinned source revision mismatch or modified tracked source: ' + name)


def update_catalog(model, args):
    """Only the caller's explicit opt-in changes a catalog; other models are unnecessary."""
    if not args.update_catalog:
        return
    path = args.models_dir / 'catalog.json'
    catalog = json.loads(path.read_text()) if path.exists() else {'version': 1, 'models': []}
    entries = catalog.get('models', [])
    replaced = False
    for index, entry in enumerate(entries):
        if entry['id'] == model['id']:
            entries[index] = model
            replaced = True
            break
    if not replaced:
        entries.append(model)
    catalog['models'] = entries
    # Preserve a usable existing default. A missing optional model cannot become required.
    default = next((entry for entry in entries if entry['id'] == catalog.get('defaultModel')), None)
    if default is None or not (args.models_dir / default['url']).is_file():
        catalog['defaultModel'] = model['id']
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
