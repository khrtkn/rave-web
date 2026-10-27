#!/usr/bin/env python3
"""Reproduce original noise semantics, raw levels, and numerical baselines."""
import json
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

from build_djembe_candidate import SOURCE, SHA, EXTRA_DELAY, explicit_noise, run_stream
from build_water_latent import errors

OUT = Path(__file__).resolve().parents[1] / 'web/models/djembe-candidate'


def native(data, optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = optimization
    return ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider'])


def main():
    source = SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SHA
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
    (OUT/'noise-source-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    candidate_data = (OUT/'djembe-stateful-2048.onnx').read_bytes()
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
    (OUT/'numerical-baseline.json').write_text(json.dumps(baseline, indent=2)+'\n')
    print(json.dumps({'noiseAudit': audit, 'numericalBaselines': baseline}, indent=2))


if __name__ == '__main__':
    main()
