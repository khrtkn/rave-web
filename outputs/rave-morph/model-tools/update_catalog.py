#!/usr/bin/env python3
"""Merge independently validated models; never changes source weights."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1] / 'web' / 'models'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--default', default='djembe-stateful-2048')
    args = parser.parse_args()
    path = ROOT / 'catalog.json'
    current = json.loads(path.read_text())
    models = [m for m in current['models'] if m['id'].startswith('water-latent-')]
    for model in models:
        model.update(family='water', familyName='Water / IIL · Pond Brain', controlDescription='学習済み16座標のうち6座標を操作（PCAではありません）')
        if model['frameSize'] == 2048:
            model['parityFixture'] = 'parity-fixture.json'
    candidate = ROOT / 'djembe-candidate' / 'contract.json'
    if candidate.exists():
        model = json.loads(candidate.read_text())
        model.update(family='djembe', familyName='Djembe / Scyclone（実験）', controlDescription='射影後の実際の潜在4座標を操作・実験モデル', defaultOutputGain=0.04)
        model['url'] = 'djembe-candidate/' + Path(model['url']).name
        models.append(model)
    if args.default not in {m['id'] for m in models}:
        parser.error('default model is not present')
    current.update(defaultModel=args.default, models=models)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'models': [m['id'] for m in models], 'default': args.default}))


if __name__ == '__main__':
    main()
