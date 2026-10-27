# Graph review and timing boundary

The inspected original has opset 17, IR 8, 573 nodes, 235 initializers, and 37 explicit state input/output pairs. The adapted model has 575 nodes. `source-integrity.json` verifies that every `model.*` learned initializer is byte-identical and identifies the only changed block-specific constants.

The encoder begins with a 257-tap PQMF analysis Conv at stride 16, followed by five stride-2 downsampling Convs and a stride-4 Conv: `16 × 2^5 × 4 = 2048`. The 4096 original therefore has two encoded time steps; the 2048 variant has one. The retained latent channels, in the order controlled by the new inputs, are `[97,86,120,10,55,60,1,72,25,32,88,114,66,49,59,24]`.

## What is established about streaming delay

Each explicit history update concatenates the previous state with the current packet/activation, then retains the tail with a negative-start Slice. The graph is invoked using current inputs and past state only. It does not request a future packet. Both variants preserve this causal state transition, and the matched-noise block-size parity test supports the 2048 specialization without overlap-add.

This does **not** establish that output audio is time-aligned to the corresponding dry input samples. Cache lengths encode past receptive fields and phase alignment, and are not independently additive latency values. PQMF analysis/synthesis have filtering phase; learned nonlinear encoder/decoder operations and transposed convolutions do not each have a single signal-independent group delay. The source export's authoritative cumulative-delay/alignment metadata is absent from this ONNX. The original TorchScript model/checkpoint/exporter was not available for native timestamp alignment comparison.

Therefore `modelDelaySamples` remains `null`. No fixed dry offset is inferred from water-audio cross-correlation or from the cache sizes. Equal packet order addresses transport alignment only. A reliable fixed dry compensation requires the matching source exporter/model timing contract or a justified input-to-output alignment reference. Input/output covariance of a timbre-changing water model is not that reference.

For 2048 at 48 kHz, **42.6667 ms is a packet's audio duration**, not measured physical E2E latency. Actual microphone-to-speaker delay also involves packet collection, inference and queue scheduling, audio callback/device buffering, and any model alignment delay. Native `session.run` timings include output/state retrieval but do not predict browser WASM timings or E2E delay.

## Audio behavior limits

The source model card explicitly warns that Pond Brain decoders may produce a loud chirp at initialization. Initialize/warm the model without connecting its raw transient to the speaker, ramp the listening gain, and provide a limiter/mute in the application. These measures protect playback; they are not inserted into the ONNX parity fixture.

Native tests establish numerical preservation, state continuity, finite outputs for the tested controls, and independent effects of all six exposed axes. They do not establish perceptual equivalence to Morpho, a semantic meaning for any coordinate, or microphone/physical-loopback latency.
