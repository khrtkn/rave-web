#!/usr/bin/env python3
"""Reproduce original noise semantics, raw levels, and numerical baselines."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

from build_djembe_candidate import SHA, EXTRA_DELAY, explicit_noise, run_stream
from native_common import errors
from prepare_djembe_source import SOURCE_BYTES


def native(data, optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = optimization
    return ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Pinned original Scyclone djembe.onnx')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'web/models/djembe-candidate', help='Directory produced by build_djembe_candidate.py; audit reports are written here')
    args = parser.parse_args()
    out = args.out
    source = args.source.read_bytes()
    if len(source) != SOURCE_BYTES or hashlib.sha256(source).hexdigest() != SHA:
        raise ValueError('Djembe source differs from the pinned upstream byte size / SHA-256')
    candidate_data = (out/'djembe-stateful-2048.onnx').read_bytes()
    contract = json.loads((out/'contract.json').read_text())
    if len(candidate_data) != contract['bytes'] or hashlib.sha256(candidate_data).hexdigest() != contract['sha256']:
        raise ValueError('Djembe candidate differs from its generated contract')
    fixture = json.loads((out/'parity-fixture.json').read_text())
    fixture_arrays = {}
    for key in ['input', 'expectedOutput']:
        info = fixture[key]
        path = out/info['file']
        if path.resolve().parent != out.resolve():
            raise ValueError('Fixture path must stay in the build output directory')
        data = path.read_bytes()
        if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError('Saved Djembe parity fixture differs from its declared byte size / SHA-256')
        fixture_arrays[key] = np.frombuffer(data, dtype='<f4').reshape(1, 1, -1)
    if fixture['modelSha256'] != contract['sha256']:
        raise ValueError('Djembe fixture and model SHA-256 differ')
    fixture_output, _, _ = run_stream(native(candidate_data), fixture_arrays['input'])
    fixture_error = errors(fixture_output, fixture_arrays['expectedOutput'])
    if fixture_error['max_abs'] > fixture['tolerance']['maxAbs'] or fixture_error['rms'] > fixture['tolerance']['rms']:
        raise ValueError('Saved Djembe parity fixture failed native reload validation')
    length = 262144
    rng = np.random.default_rng(20261026)
    t = np.arange(length)/48000
    audio = (.12*np.sin(2*np.pi*(170+32*t)*t)+.03*rng.standard_normal(length)).astype(np.float32)[None, None]
    audio[:, :, :2048*32] = 0
    warm = 98304
    diagnostic = onnx.load_from_string(source)
    ops = []
    for node in diagnostic.graph.node:
        if node.op_type == 'RandomNormalLike':
            attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            assert attributes == {'dtype': 1}
            ops.append({'name': node.name, 'attributes': attributes, 'output': node.output[0]})
            diagnostic.graph.output.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, None))
    realized = native(diagnostic.SerializeToString()).run(None, {'audio_in': audio})
    reference_data = explicit_noise(source).SerializeToString()
    reference = native(reference_data)
    reference_realized = reference.run(None, {'audio_in': audio, 'encode_noise': realized[1], 'decode_noise': realized[2]})[0]
    realized_error = errors(realized[0], reference_realized)
    assert realized_error['max_abs'] == 0
    audit = {'originalNoiseOps': ops, 'originalNoiseEmpiricalStats': [
        {'name': node['name'], 'shape': list(arr.shape), 'mean': float(arr.mean()), 'std': float(arr.std())}
        for node, arr in zip(ops, realized[1:])],
        'originalRawOutput': {'postWarmRms': float(np.sqrt(np.mean(realized[0][:, :, warm:].astype(np.float64)**2))),
                              'postWarmPeak': float(np.max(np.abs(realized[0][:, :, warm:])))},
        'explicitNoiseVsOriginalExactSameRealizedNoise': realized_error,
        'nativeFixtureReloadParity': {**fixture_error, 'comparedSamples': fixture_output.size, 'passed': True},
        'sourceSha256': SHA, 'candidateSha256': contract['sha256'],
        'conclusion': 'Original ops have dtype only: ONNX defaults mean=0, scale=1. Substitution preserves the exact realized output. Original raw output overshoot is present before adaptation.'}
    noise = {'encode_noise': rng.standard_normal((1, 16, 128)).astype(np.float32),
             'decode_noise': rng.standard_normal((1, 12, 128)).astype(np.float32)}
    rows = []
    for scale in [0, .1, .25, .5, 1]:
        output = reference.run(None, {'audio_in': audio, **{k: v*scale for k, v in noise.items()}})[0][:, :, warm:]
        rows.append({'scaleAppliedToBothNoiseInputs': scale, 'rms': float(np.sqrt(np.mean(output.astype(np.float64)**2))),
                     'peak': float(np.max(np.abs(output)))})
    audit['explicitNoiseScaleDiagnostic'] = rows
    audit['scaleDiagnosticNote'] = 'An application control experiment only; scale=1 is the original distribution. No default noise scale was changed in either model contract or graph.'
    (out/'noise-source-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    results = {}
    for name, level in [('all', ort.GraphOptimizationLevel.ORT_ENABLE_ALL), ('disabled', ort.GraphOptimizationLevel.ORT_DISABLE_ALL)]:
        full = native(reference_data, level).run(None, {'audio_in': audio, **noise})[0]
        stream, _, _ = run_stream(native(candidate_data, level), audio, noise=noise)
        results[name] = {'full': full, 'stream': stream}
    baseline = {name+'_optimization_aligned_parity': errors(row['full'][:, :, warm:-EXTRA_DELAY], row['stream'][:, :, warm+EXTRA_DELAY:])
                for name, row in results.items()}
    for name in ['full', 'stream']:
        baseline[name+'_graph_all_vs_disabled'] = errors(results['all'][name][:, :, warm:], results['disabled'][name][:, :, warm:])
    baseline['originalDynamicGraphLengthBaselines'] = []
    for size in [131072, 196608, 245760]:
        shortened = reference.run(None, {'audio_in': audio[:, :, :size], **{k: v[:, :, :size//2048] for k, v in noise.items()}})[0]
        end = size-4096
        baseline['originalDynamicGraphLengthBaselines'].append({'shortenedInputSamples': size, 'comparedRange': [warm, end],
            'sameOriginalGraphSamePrefixInputsNoiseError': errors(results['all']['full'][:, :, warm:end], shortened[:, :, warm:end])})
    baseline['interpretation'] = 'Disabling graph optimization did not reduce the error. The unadapted dynamic graph itself varies numerically with signal length; this does not prove that all candidate residual error has that sole cause. Report measured normal-noise error explicitly.'
    baseline['sourceSha256'] = SHA
    baseline['candidateSha256'] = contract['sha256']
    (out/'numerical-baseline.json').write_text(json.dumps(baseline, indent=2)+'\n')
    print(json.dumps({'noiseAudit': audit, 'numericalBaselines': baseline}, indent=2))


if __name__ == '__main__':
    main()
