# Djembe stateful candidate

An experimental, smaller RAVE model for additional inference headroom. It is separate from the Water model. It is not Morpho, and no Morpho model or binary was used.

## Provenance and license

The immediate original is [Scyclone's djembe ONNX](https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/assets/models/djembe.onnx), pinned commit `2b1990c4ad04fec1981d21bdde94fdafbadb84e0`, 16,622,944 bytes, SHA256 `ea94557a8d9b40f22a1dadbc93ab98f46144a68ce0bed1403a1558a2d1007f16`. Attribution: Torsion Audio; RAVE by Antoine Caillon and Philippe Esling / ACIDS IRCAM. The [source README's license statement](https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/README.md#licenses) assigns the model [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The same attribution/noncommercial conditions apply to this adapted model.

All 46 original initializers are byte-identical. Changes on 2026-10-26: explicit normal-noise inputs, past-state caches for 35 convolutions, four transposed-convolution caches with causal output shifts, a synthesis modulation phase adjustment, and gain/bias on the original four retained projected latent coordinates. No retraining, new model weights, or model approximation was used.

The public Scyclone tree (355 entries at the pinned commit) contains ONNX/ORT artifacts but no original djembe TorchScript/checkpoint/gin. Its README explicitly links [Scyclone-AI](https://github.com/Torsion-Audio/Scyclone-AI/tree/1bb0dfa1cb125986cc0aa2b97e00d8359639be4b), whose 15-entry tree supplies generic RAVE v1 training/config/export instructions, not the exact trained djembe checkpoint. This is a bounded source search, not a claim that no checkpoint exists anywhere.

## Model contract and delay

- Input/output: float32 mono `[1,1,2048]`, 48 kHz, `audio_in` → `audio_out`.
- Original total encoder stride: `16 × 4 × 4 × 4 × 2 = 2048`. This is a lighter model, not a smaller encoder hop.
- Noise: `encode_noise [1,16,1]`, `decode_noise [1,12,1]`, both standard normal. No synth-noise input.
- Controls: `latent_gain` / `latent_bias [1,4,1]`. Neutral gain 1 / bias 0. The original graph keeps four projected coordinates and fills the other twelve with noise. Four axes are exposed; no six-axis claim is made. The 16×16 projection and inverse are transpose pairs, but statistical PCA provenance is not inferred from that algebra alone.
- State: 39 input/output pairs, zero reset. Continuous states replace independent padded packets; no stateless overlap-add is used.
- Initialization: 64 silent warmup packets recommended. The reference alignment test excludes the first 98,304 samples (48 packets); it does not claim startup-transient equivalence.

To stream the four transposed convolutions, the adapter retains two past input frames at each stage and emits a causally shifted slice. The added shifts, expressed at the final sample rate, are `2×512 + 2×128 + 2×32 + 1×16 = 1360 samples`, or **28.3333 ms**. This is 85 subband samples; synthesis modulation on odd subbands needs the corresponding odd-phase sign adjustment. The native long-signal alignment test verifies that combination.

The known added shift is recorded as `additionalDelaySamples`, `dryOffsetSamples`, and `modelDelaySamples` = 1360 for application compensation. **The original model's absolute dry/wet alignment remains unknown**, recorded as `sourceModelDelaySamples:null`. Compensating 1360 addresses only the known adaptation component. A 2048 packet lasts 42.6667 ms; neither number is a physical microphone-to-speaker latency measurement.

## Numerical evidence

`candidate-validation.json` reports:

- Source long-signal versus streaming candidate, identical zero noise, aligned by 1360: maximum absolute error `4.74e-9`, RMS `2.18e-10`, over 162,480 samples after warmup.
- Identical explicit standard-normal noise: maximum `4.58e-4`, RMS `1.58e-5` (reference RMS `1.739`). This is measured approximate float32 equivalence, not exact parity. It exceeds the strict zero-noise absolute threshold and remains explicitly experimental.
- All four dimensions' gain and bias independently change output, with the same input/noise and warm state. Resetting reproduces the output and all 39 states bit for bit.
- Latest native CPU one-thread median about 2.18 ms / p95 3.25 ms per 2048 packet. These are native timings, not browser timings; see the app's browser measurements separately.

`numerical-baseline.json` records that disabling ONNX graph optimization did not remove the normal-noise error. The unmodified dynamic model itself varies with input length (maximum about `6.7e-5` on equal-prefix inputs/noise). That observation does not establish that every residual candidate error has the same cause.

`parity-fixture.json` and its raw little-endian float32 files compare **the first eight candidate packets against the candidate's native execution**, with all states/noise zero, gain 1/bias 0, no warmup, limiter, fade or alignment shift. This browser-runtime check is distinct from the original-model long-signal alignment test above.

## Actual raw output levels

The source's two `RandomNormalLike` operators have only `dtype:1` attributes; their original distribution is N(0,1). `noise-source-audit.json` captures their realized noise values and shows that supplying those exact tensors to the explicit-noise reference reproduces the source output with zero error. No source scale was discarded.

The **unadapted** source itself produces large raw levels on the test signal (observed RMS around 2, peaks over 14). Reducing both noise inputs to 0.1 still produced a peak over 3 in the diagnostic. Noise reduction changes the model's operation and is not a replacement for output level management. No default noise scale was silently changed by this adapter.

The native [InferenceThread](https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/source/dsp/onnx/InferenceThread.cpp#L26) copies raw inference output, while the [plugin processing chain](https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/source/PluginProcessor.cpp#L238) includes compression. Its auto-makeup behavior does not provide a strict peak bound. Use controlled listening gain, mute and a peak limiter in the application; leave the parity fixture raw. No perceptual-quality equivalence is established by these numerical tests.

## Reproduce

From the workspace using the model-tools virtual environment:

```sh
work/rave-morph-model-venv/bin/python outputs/rave-morph/model-tools/build_djembe_candidate.py
work/rave-morph-model-venv/bin/python outputs/rave-morph/model-tools/audit_djembe_candidate.py
```

Both scripts read the original djembe file without modifying it. Outputs stay in this candidate directory; Water's catalog, contract, and models are not rewritten. The original source path can be supplied to the builder with `--source`. The original ONNX SHA is checked before modification.
