#!/usr/bin/env python3
"""Independent per-axis/reset tests and a raw-f32 browser parity fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_water_latent import LATENT_INDICES, errors, session, states_for, step


def binary(path, data):
    raw = np.asarray(data, dtype="<f4").tobytes(order="C")
    path.write_bytes(raw)
    return {"file": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def run(sess, initial_states, inputs, gain, bias):
    state = {name: value.copy() for name, value in initial_states.items()}
    output = []
    for feeds in inputs:
        y, state, _ = step(sess, state, feeds, gain, bias)
        assert np.isfinite(y).all()
        output.append(y)
    return np.concatenate(output, axis=2), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path(__file__).resolve().parents[1] / "web/models")
    args = parser.parse_args()
    root = args.models
    contract = json.loads((root / "contract.json").read_text())
    sess = session((root / contract["url"]).read_bytes())
    frame = contract["frameSize"]
    assert frame == 2048
    rng = np.random.default_rng(1234)
    neutral = np.ones((1, 16, 1), np.float32), np.zeros((1, 16, 1), np.float32)
    inputs = []
    for j in range(64):
        t = (np.arange(frame) + j*frame) / 48000
        audio = (0.15*np.sin(2*np.pi*(210+60*t)*t) + 0.03*rng.standard_normal(frame)).astype(np.float32)[None, None]
        inputs.append({"audio": audio, "encode_noise": rng.standard_normal((1, 128, 1)).astype(np.float32),
                       "decode_noise": rng.standard_normal((1, 128, 1)).astype(np.float32),
                       "synth_noise": rng.uniform(-1, 1, (1, 2, 16, 64)).astype(np.float32)})
    _, warm_states = run(sess, states_for(sess), inputs[:40], *neutral)
    base, _ = run(sess, warm_states, inputs[40:], *neutral)
    results = []
    for dimension in range(6):
        axis = {"dimension": dimension+1, "source_channel": LATENT_INDICES[dimension]}
        for operation, value in [("bias", 0.5), ("gain", 0.5)]:
            gain, bias = (x.copy() for x in neutral)
            (gain if operation == "gain" else bias)[0, dimension, 0] = value
            out, _ = run(sess, warm_states, inputs[40:], gain, bias)
            difference = errors(base, out)
            assert difference["rms"] > 1e-6, (dimension, operation, difference)
            axis[operation] = {"value": value, "difference": difference, "peak": float(np.max(np.abs(out))), "finite": True}
        results.append(axis)
    # Separate zero-noise fixture; includes both a transient and sustained signal.
    fixture_inputs = []
    for j in range(8):
        absolute = np.arange(frame) + j*frame
        t = absolute / 48000
        x = 0.12*np.sin(2*np.pi*220*t) + 0.04*np.sin(2*np.pi*733*t)
        x += np.where((absolute >= 5000) & (absolute < 5400), 0.15*np.exp(-(absolute-5000).clip(0)/90), 0)
        fixture_inputs.append({"audio": x.astype(np.float32)[None, None],
                               "encode_noise": np.zeros((1, 128, 1), np.float32),
                               "decode_noise": np.zeros((1, 128, 1), np.float32),
                               "synth_noise": np.zeros((1, 2, 16, 64), np.float32)})
    expected, expected_states = run(sess, states_for(sess), fixture_inputs, *neutral)
    repeated, repeated_states = run(sess, states_for(sess), fixture_inputs, *neutral)
    reproducibility = errors(expected, repeated)
    assert reproducibility["max_abs"] == 0
    assert all(np.array_equal(expected_states[k], repeated_states[k]) for k in expected_states)
    input_file = binary(root / "parity-input.f32", np.concatenate([x["audio"] for x in fixture_inputs], axis=2))
    expected_file = binary(root / "parity-expected.f32", expected)
    fixture = {
        "version": 1, "model": contract["id"], "modelSha256": contract["sha256"],
        "sampleRate": 48000, "frameSize": frame, "frames": 8, "channels": 1, "samples": frame*8,
        "format": "IEEE 754 little-endian float32, contiguous mono PCM, no header",
        "input": input_file, "expectedOutput": expected_file,
        "initialStates": "all state_* tensors zero", "latentGain": 1.0, "latentBias": 0.0,
        "allNoiseInputs": 0.0, "warmupFrames": 0, "outputAlignment": "same packet order; no crop, compensation, overlap, fade or limiter",
        "tolerance": {"maxAbs": 0.0002, "rms": 0.00002},
        "toleranceMeaning": "Compare all 16384 raw output samples against native ORT float32; this is a browser parity criterion, not proof it already passed in a browser",
        "nativeOutputPeak": float(np.max(np.abs(expected))), "nativeResetOutputError": reproducibility,
        "nativeResetAllStateTensorsIdentical": True,
    }
    (root / "parity-fixture.json").write_text(json.dumps(fixture, indent=2)+"\n")
    report_path = root / "native-validation.json"
    report = json.loads(report_path.read_text())
    report["independent_six_axis_test"] = {"warmup_blocks_2048": 40, "comparison_blocks_2048": 24,
        "source": "Deterministic chirp/noise, seed1234; identical warm states/input/noise, exactly one gain or bias differs", "results": results}
    report["zero_noise_reset_reproducibility"] = {"output_error": reproducibility, "all_state_tensors_identical": True}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"axes": results, "fixture": fixture}, indent=2))


if __name__ == "__main__":
    main()
