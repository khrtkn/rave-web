#!/usr/bin/env python3
"""Add real latent affine controls to the existing, hash-pinned RAVE water graph.

No retraining and no replacement network. Only the retained 16-channel latent
edge is edited. A 2048-sample variant adjusts two time-dependent constants and
the input/output shape declarations; learned weights and state shapes stay intact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

SOURCE_SHA256 = "b6c84085f475b3bddf69eb2ee0a85836230e0462d7d112cc3a2cb6e497914e47"
SOURCE_BYTES = 103384587
LATENT_INDICES = [97, 86, 120, 10, 55, 60, 1, 72, 25, 32, 88, 114, 66, 49, 59, 24]
EDGE = "/Gather_1_output_0"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def shape(info):
    return [d.dim_value for d in info.type.tensor_type.shape.dim]


def source_bytes(source):
    catalog = json.loads((source / "live/catalog.json").read_text())
    config = next(x for x in catalog if x["id"] == "water")
    data = b"".join((source / part).read_bytes() for part in config["asset"]["parts"])
    if len(data) != SOURCE_BYTES or digest(data) != SOURCE_SHA256:
        raise ValueError("Source model differs from the inspected, pinned RAVE water export")
    return data, config


def controlled_model(data, frame):
    if frame not in (2048, 4096):
        raise ValueError("This encoder has a 2048-sample total stride; only validated frame sizes are supported")
    model = onnx.load_from_string(data)
    weights = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    assert weights["onnx::Gather_4143"].tolist() == LATENT_INDICES
    retained = next(n for n in model.graph.node if n.name == "/Gather_1")
    assert retained.op_type == "Gather" and list(retained.output) == [EDGE]
    assert list(retained.input) == ["/Add_10_output_0", "onnx::Gather_4143"]
    decoder = next(n for n in model.graph.node if n.name == "/Conv_21")
    assert decoder.input[1] == "model.decoder.net.0.weight"
    assert weights[decoder.input[1]].shape == (768, 128, 3)
    consumers = [n for n in model.graph.node if EDGE in n.input]
    assert len(consumers) == 1 and consumers[0].name == "/Concat_17"
    assert weights["/Constant_182_output_0"].tolist() == [1, 4, 16, -1]
    np.testing.assert_array_equal(weights["/ScatterND_output_0"][:, :128], weights["/ScatterND_output_0"][:, 128:])

    for info in list(model.graph.input) + list(model.graph.output):
        if info.name in ("audio", "output"):
            info.type.tensor_type.shape.dim[2].dim_value = frame
        elif info.name in ("encode_noise", "decode_noise"):
            info.type.tensor_type.shape.dim[2].dim_value = frame // 2048
        elif info.name == "synth_noise":
            info.type.tensor_type.shape.dim[1].dim_value = frame // 1024
    if frame == 2048:
        for i, init in enumerate(model.graph.initializer):
            if init.name == "/ScatterND_output_0":
                model.graph.initializer[i].CopyFrom(numpy_helper.from_array(weights[init.name][:, :128].copy(), init.name))
            elif init.name == "/Constant_182_output_0":
                model.graph.initializer[i].CopyFrom(numpy_helper.from_array(np.array([1, 2, 16, -1], dtype=np.int64), init.name))
    # Existing internal value_info contains shape-specialized 4096 declarations.
    del model.graph.value_info[:]
    model.graph.input.extend([
        helper.make_tensor_value_info("latent_gain", TensorProto.FLOAT, [1, 16, 1]),
        helper.make_tensor_value_info("latent_bias", TensorProto.FLOAT, [1, 16, 1]),
    ])
    for node in consumers:
        for k, value in enumerate(node.input):
            if value == EDGE:
                node.input[k] = "latent_controlled"
    nodes = list(model.graph.node)
    index = next(i for i, node in enumerate(nodes) if node.name == "/Gather_1") + 1
    nodes[index:index] = [
        helper.make_node("Mul", [EDGE, "latent_gain"], ["latent_scaled"], name="RaveMorph/LatentGain"),
        helper.make_node("Add", ["latent_scaled", "latent_bias"], ["latent_controlled"], name="RaveMorph/LatentBias"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.doc_string = "Adapted RAVE water model. 16 retained learned latent coordinates; not PCA. z'=z*gain+bias. CC BY-NC 4.0. See ATTRIBUTION.md."
    helper.set_model_props(model, {
        "source_sha256": SOURCE_SHA256,
        "adaptation": "16 retained learned latent channels affine gain/bias; no learned weight changes",
        "latent_indices": json.dumps(LATENT_INDICES),
        "pca": "No PCA transform present in this exported graph; coordinates are learned gimbal-affine coordinates",
        "frame_size": str(frame),
        "license": "CC BY-NC 4.0",
    })
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def session(data):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(data, sess_options=opts, providers=["CPUExecutionProvider"])


def states_for(sess):
    return {x.name: np.zeros(x.shape, dtype=np.float32) for x in sess.get_inputs() if x.name.startswith("state_")}


def step(sess, states, inputs, gain=None, bias=None):
    feeds = {**states, **inputs}
    if gain is not None:
        feeds.update(latent_gain=gain, latent_bias=bias)
    started = time.perf_counter()
    outputs = dict(zip([x.name for x in sess.get_outputs()], sess.run(None, feeds)))
    elapsed = (time.perf_counter() - started) * 1000
    updated = {name: outputs["next_" + name] for name in states}
    return outputs["output"], updated, elapsed


def errors(a, b):
    e = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(e))), "rms": float(np.sqrt(np.mean(e * e)))}


def validate(original, full, half, blocks):
    sessions = [session(x) for x in (original, full, half)]
    states = [states_for(x) for x in sessions]
    rng = np.random.default_rng(20261026)
    neutral = np.ones((1, 16, 1), np.float32), np.zeros((1, 16, 1), np.float32)
    reference, untouched, smaller, times = [], [], [], [[], [], []]
    state_errors = []
    saved_inputs = []
    for block in range(blocks):
        t = (np.arange(4096) + block * 4096) / 48000
        # Changing amplitudes, frequencies, attacks, deterministic low-level noise.
        audio = ((0.10 + 0.06 * np.sin(2 * np.pi * 1.3 * t)) * np.sin(2 * np.pi * (170 + 110 * t) * t)
                 + 0.03 * np.sin(2 * np.pi * 977 * t) + 0.002 * rng.standard_normal(4096)).astype(np.float32)[None, None]
        if block % 11 == 0:
            audio[:, :, :256] = 0
        inputs = {"audio": audio, "encode_noise": rng.standard_normal((1, 128, 2)).astype(np.float32),
                  "decode_noise": rng.standard_normal((1, 128, 2)).astype(np.float32),
                  "synth_noise": rng.uniform(-1, 1, (1, 4, 16, 64)).astype(np.float32)}
        saved_inputs.append(inputs)
        a, states[0], ms = step(sessions[0], states[0], inputs); times[0].append(ms)
        b, states[1], ms = step(sessions[1], states[1], inputs, *neutral); times[1].append(ms)
        pieces = []
        for i in range(2):
            part = {"audio": inputs["audio"][:, :, i*2048:(i+1)*2048],
                    "encode_noise": inputs["encode_noise"][:, :, i:i+1],
                    "decode_noise": inputs["decode_noise"][:, :, i:i+1],
                    "synth_noise": inputs["synth_noise"][:, i*2:(i+1)*2]}
            out, states[2], ms = step(sessions[2], states[2], part, *neutral)
            times[2].append(ms); pieces.append(out)
        reference.append(a); untouched.append(b); smaller.append(np.concatenate(pieces, axis=2))
        state_errors.extend(errors(states[0][name], states[2][name])["max_abs"] for name in states[0])
    neutral_error = errors(np.concatenate(reference, axis=2), np.concatenate(untouched, axis=2))
    block_error = errors(np.concatenate(reference, axis=2), np.concatenate(smaller, axis=2))
    assert neutral_error["max_abs"] < 1e-5, neutral_error
    assert block_error["max_abs"] < 2e-4 and block_error["rms"] < 2e-5, block_error
    # Same source, noise and initial state, changing only latent controls.
    controlled = session(half)
    controlled_states = states_for(controlled)
    gain, bias = copy.deepcopy(neutral)
    gain[0, :6, 0] = [0.5, 1.5, 0.75, 1.25, 0.8, 1.2]
    bias[0, :6, 0] = [0.7, -0.5, 0.4, -0.3, 0.2, -0.1]
    changed = []
    for inputs in saved_inputs:
        for i in range(2):
            part = {"audio": inputs["audio"][:, :, i*2048:(i+1)*2048],
                    "encode_noise": inputs["encode_noise"][:, :, i:i+1],
                    "decode_noise": inputs["decode_noise"][:, :, i:i+1],
                    "synth_noise": inputs["synth_noise"][:, i*2:(i+1)*2]}
            out, controlled_states, _ = step(controlled, controlled_states, part, gain, bias)
            assert np.isfinite(out).all()
            changed.append(out)
    # Ignore the first 32 original blocks for audible difference measurements.
    start = min(32, blocks // 2) * 4096
    base = np.concatenate(smaller, axis=2)[:, :, start:]
    manipulated = np.concatenate(changed, axis=2)[:, :, start:]
    knob_error = errors(base, manipulated)
    assert knob_error["rms"] > 1e-6, knob_error
    return {
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__,
                        "provider": "CPUExecutionProvider", "intra_op_threads": 1, "inter_op_threads": 1},
        "blocks_4096": blocks, "audio_seconds": blocks * 4096 / 48000,
        "input": "Deterministic synthetic chirp/AM/multiple sine/noise/zero segments, seed 20261026; identical noise tensors shared between variants",
        "neutral_4096_vs_original": neutral_error,
        "neutral_2048_vs_original_4096": block_error,
        "maximum_final_state_abs_error_over_stream": float(max(state_errors)),
        "latent_knob_output_difference_after_warmup": knob_error,
        "neutral_output_rms_after_warmup": float(np.sqrt(np.mean(base.astype(np.float64)**2))),
        "changed_output_rms_after_warmup": float(np.sqrt(np.mean(manipulated.astype(np.float64)**2))),
        "changed_output_peak_after_warmup": float(np.max(np.abs(manipulated))),
        "timings_ms": {name: {"median": float(np.median(t)), "p95": float(np.percentile(t, 95)), "max": float(max(t)), "mean": float(np.mean(t))}
                       for name, t in zip(["original_4096", "latent_4096", "latent_2048"], times)},
        "timing_scope": "Native Python ONNX Runtime CPU session.run including output and state retrieval; not browser/WASM timings or E2E latency",
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing public-runtime directory containing live/catalog.json and its model chunks; source SHA-256 is verified")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "web/models")
    parser.add_argument("--blocks", type=int, default=48)
    args = parser.parse_args()
    if args.blocks < 8:
        parser.error("--blocks must be at least 8")
    data, config = source_bytes(args.source)
    models = {frame: controlled_model(data, frame) for frame in (4096, 2048)}
    encoded = {frame: model.SerializeToString() for frame, model in models.items()}
    report = validate(data, encoded[4096], encoded[2048], args.blocks)
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for frame, model in models.items():
        filename = f"water-latent-{frame}.onnx"
        (args.out / filename).write_bytes(encoded[frame])
        inputs = {x.name: shape(x) for x in model.graph.input}
        entry = {
            "id": f"water-latent-{frame}", "name": f"Water · RAVE latent ({frame})", "url": filename,
            "sampleRate": 48000, "frameSize": frame, "audioInput": "audio", "audioOutput": "output",
            "audioInputShape": [1, 1, frame], "audioOutputShape": [1, 1, frame], "dtype": "float32",
            "latentGainInput": "latent_gain", "latentBiasInput": "latent_bias", "latentShape": [1, 16, 1],
            "latentDimensions": 16, "exposedDimensions": 6, "latentCoordinateSystem": "retained learned gimbal-affine latent coordinates; not PCA",
            "latentSourceChannels": LATENT_INDICES, "neutralGain": 1, "neutralBias": 0,
            "noiseInputs": [{"name": name, "shape": inputs[name], "distribution": distribution}
                            for name, distribution in [("encode_noise", "normal"), ("decode_noise", "normal"), ("synth_noise", "uniform")]],
            "states": [{"input": name, "output": "next_"+name, "shape": inputs[name]} for name in inputs if name.startswith("state_")],
            "stateReset": "zeros", "streaming": True, "overlapSamples": 0, "blockDurationMs": frame / 48,
            "modelDelaySamples": None, "latencyNote": "Block duration only; native model delay and physical E2E latency have not been measured.",
            "opset": 17, "irVersion": 8, "bytes": len(encoded[frame]), "sha256": digest(encoded[frame]),
            "sourceSha256": SOURCE_SHA256, "sourceURL": config["source"], "author": config["author"],
            "license": "CC BY-NC 4.0", "licenseURL": "https://creativecommons.org/licenses/by-nc/4.0/",
            "adaptation": "Added latent gain/bias inputs; 2048 variant reduces block-specific sign mask and synth reshape constants; all learned weights unchanged",
            "validatedNative": True,
        }
        entries.append(entry)
    catalog = {"version": 1, "defaultModel": "water-latent-2048", "models": sorted(entries, key=lambda x: x["frameSize"]),
               "purpose": "Independent RAVE prototype with latent controls; not Neutone Morpho", "commercialUseAllowed": False}
    (args.out / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+"\n")
    (args.out / "contract.json").write_text(json.dumps(next(x for x in entries if x["frameSize"] == 2048), ensure_ascii=False, indent=2)+"\n")
    (args.out / "native-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
