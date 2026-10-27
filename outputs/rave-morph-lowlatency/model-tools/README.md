# Real 512-hop RAVE export tools

This directory builds the pretrained IIL `organ_archive_b512_r48000` model into a browser-compatible stateful ONNX. It does not train a replacement model and does not use Neutone Morpho assets. Earlier `outputs/rave-morph/` files are read only.

## Reproduce from a fresh repository root

Prerequisites for **building**: Python 3.12, Git, network access to public PyPI/GitHub/Hugging Face, and several GB of free disk space for the 670 MB training checkpoint, dependencies and generated files. Commands below run from the root of this repository. This workflow obtains the original model from its public provider; it does not download a project-hosted converted model. Water and Djembe files are not prerequisites.

```sh
python3.12 -m venv work/rave-lowlatency/venv
work/rave-lowlatency/venv/bin/python -m pip install --no-cache-dir -r outputs/rave-morph-lowlatency/model-tools/requirements-lock.txt
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/prepare_sources.py
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/build_organ.py --update-catalog
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/verify_organ.py
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/audit_reference.py
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/build_organ_1024.py --update-catalog
```

`prepare_sources.py` fetches only the two explicit anonymous official model URLs and two pinned public source repositories. It verifies file sizes/SHA256, Git revisions and tracked source cleanliness; it refuses to overwrite a mismatched existing final file or reset an existing source checkout. A network failure may leave a workspace-local `.partial` file. The script records sources in `work/rave-lowlatency/source-manifest.json`. The checked-in `source-manifest.json` is the original source reference and is not rewritten.

All source/checkpoint files and the virtual environment stay under `work/rave-lowlatency/`. Generated models, fixtures, numerical results and synthetic WAV files go to `outputs/rave-morph-lowlatency/web/models/organ-512/` and `organ-1024/`. The normal scripts do not modify a catalog; the explicit `--update-catalog` flag above registers the newly generated model and preserves other entries. An existing default whose model is unavailable is replaced by the generated model. Neither builder reads or requires comparison models. Generated ONNX files and checkpoints are local build outputs, not source files to commit.

Every command also accepts `--work-dir PATH` and `--models-dir PATH`. Pass the same values to all five commands for an isolated build. `--models-dir` is the parent of `organ-512/` and `organ-1024/`, not an individual model directory. For example:

```sh
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/prepare_sources.py --work-dir work/organ-isolated
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/build_organ.py --work-dir work/organ-isolated --models-dir work/organ-generated
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/verify_organ.py --work-dir work/organ-isolated --models-dir work/organ-generated
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/audit_reference.py --work-dir work/organ-isolated --models-dir work/organ-generated
work/rave-lowlatency/venv/bin/python outputs/rave-morph-lowlatency/model-tools/build_organ_1024.py --work-dir work/organ-isolated --models-dir work/organ-generated
```

The isolated example leaves the browser app's catalog unchanged. Export must finish before verification; `build_organ_1024.py` additionally needs the 512 ONNX, contract and deterministic arrays produced by the first build. It validates that newly generated ONNX against its own contract hash and the fixed original checkpoint. It does not require an earlier project release hash.

The lock file captures the actual successful environment, including PyTorch/torchaudio 2.5.0, ONNX 1.23.2, ONNX Runtime 1.31.0, NumPy 1.26.4, Lightning 1.9.0, and SciPy 1.12.0. The original fork's SciPy 1.10 pin predates Python 3.12 support; the compatibility version was tested against the actual checkpoint. Setuptools 69.5.1 supplies `pkg_resources` required by Lightning 1.9. `nn_tilde` introduced the matching PyTorch 2.5.0 constraint in this environment.

Crucially, the numerical model imports the pinned `victor-shepardson/cached_conv` checkout from the workspace before site-packages. A PyPI `cached-conv==2.5.0` package also appears in the dependency lock through `nn_tilde`; it is **not** the causal implementation used by this exporter. `build_organ.py` explicitly sets both source import paths. No system installation is changed.

## Source identity

- [Official IIL model/config directory](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/tree/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000): checkpoint 669,961,497 bytes, SHA256 `218518cb0f7c2670a9cd3973c3e2972632893f599deacc851998b372a8f8a22e`.
- `organ-source-config.gin` is the unchanged 7,087-byte configuration, SHA256 `975e0e8615204af6c9f8aa901275ac634d70f60011c898dbc103e1e4fb09ccb0`.
- [RAVE fork](https://github.com/victor-shepardson/RAVE/tree/ab7d40d6740130f0268debbc4b6633318357ea7d), branch `vs-ups`; its `master` branch is not an interchangeable causal-export implementation.
- [cached_conv fork](https://github.com/victor-shepardson/cached_conv/tree/d2da85992d030dcf1dd8ebf318c3d69753729295).

The checkpoint is loaded with `torch.load(..., weights_only=True, map_location='cpu')`. All learned state is present. The only missing current-source buffer is `latent_valid_crop`, absent from the older training checkpoint and unused by the direct encode/decode inference path. Training-only discriminator and loss modules are discarded after loading; their weights are not shipped to the browser.

## Streaming and model fidelity

The true encoder stride is `8 PQMF bands × 4 × 4 × 4 × 1 = 512`. Native 512 input produces one `[1,128,1]` posterior step and 512 output samples. A 256 input fails the original encoder's kernel-size requirement; this exporter does not pretend to provide a 256-hop model.

The checkpoint was trained with causal padding and `causal_convtranspose=True`. The exporter uses that noncached model as its reference. It replaces each convolution's **actual source left padding** with a past-input state, and each trained causal transposed convolution's left-input pad with its past-frame state. It preserves original strides, dilation, groups, output padding, PQMF ×8 scaling/channel ordering, Snake activations, and amplitude modulation/tanh. The resulting graph has 36 explicit states and requires no overlap-add or new causalization shift.

The fork's generic `use_cached_conv(True)` factory is intentionally not used: its transposed-convolution dispatch ignores the causal selector and can choose another timing behavior. The explicit wrapper is validated directly against the trained noncached causal network.

Weight normalization is folded using the source's own PyTorch operation, without retraining. Full checkpoint parameter loading and weight-norm neutral-output comparison are recorded in `native-validation.json`. The source AdaIN learn/update flags are zero, so tracing fixes the verified inference configuration; runtime adaptation of those statistics would require a separate implementation/export.

`audit_reference.py` additionally compares the **unchanged `RAVE.forward`** against neutral controls with the same realized standard-normal samples. The 16,384-sample comparison is bit-identical. Long-reference versus 512 stateful native processing has max error `3.13e-7`; ONNX versus native stateful has max `4.02e-7` in the recorded 6.016-second test.

## PCA controls and noise

The checkpoint contains its actual 128×128 `latent_pca`, `latent_mean`, and cumulative fidelity metadata. [The pinned RAVE source computes these PCA components](https://github.com/victor-shepardson/RAVE/blob/ab7d40d6740130f0268debbc4b6633318357ea7d/rave/model.py#L531). All 128 posterior coordinates remain present. Only the first six PCA directions receive controllable deltas:

```text
z = posterior_mean + (softplus(posterior_scale) + 1e-4) * epsilon
p = PCA_first6 * (z - latent_mean)
z_controlled = z + PCA_first6_transpose * (p * (gain - 1) + bias)
```

Neutral gain 1/bias 0 gives zero delta and preserves the original source forward path. This is a full-posterior checkpoint implementation, not the official TorchScript export's optional fidelity truncation/noise-fill policy. Axis signs follow the checkpoint matrix; no official TS sign convention is assumed. The first six metadata entries reach cumulative fidelity about 0.3455; axes have no guaranteed semantic labels.

`encode_noise [1,128,1]` is explicit N(0,1), preserving the original forward's temperature-1 sampling. Zero epsilon is used deliberately for the browser parity fixture. `noise_module=None` removes the separate waveform-noise generator, not the variational posterior sampling.

`verify_organ.py` independently tests all six gains and all six biases against identical inputs, noise and warm states; tests reset equality for every state; and creates raw float32 WAV comparisons. See `control-validation.json` for measured differences and levels. Those checks do not claim perceptual equivalence to Morpho.

## Timing interpretation

512 samples at 48 kHz is **10.6667 ms of audio per packet**. Native long-reference alignment passes with **zero additional streaming shift**. The checkpoint's `receptive_field=[22207,470]` describes dependency support, not a measured fixed delay. Original absolute dry/wet alignment and physical microphone-to-speaker latency remain unmeasured; the contract keeps `modelDelaySamples:null` and `additionalDelaySamples:0` distinct.

The recorded native CPU one-thread medians are about 1.92 ms (PyTorch wrapper) and 2.81 ms (ONNX Runtime) per packet. They are not browser/WASM timings or end-to-end latency. Browser numerical comparison, state ownership, live underruns and timing are measured separately by the app's verification tools.

The fixture runs the first eight 512-sample packets from zero state/noise, with no warmup, gain adjustment, crop, limiter or fade. The demo WAVs are also raw neural output, with one silent second at the start; they are not normalized to make the converted sound appear louder.

## Optional1024-sample packets

`build_organ_1024.py` reuses the verified source loader/wrapper and the prepared deterministic512 demo arrays, then adds `web/models/organ-1024/` and one catalog variant. Run it after the512 builder. It preserves the existing512 model and catalog default. All74 ONNX initializers,36 state mappings, and128 posterior channels are unchanged. Each packet contains two true512-stride latent frames, with Gaussian epsilon `[1,128,2]` and the same six PCA controls `[1,6,1]` broadcast over both. It is not a retrained or quantized model.

Its6.016-second native long-reference vs1024 stream comparison is bit-identical; native512 vs1024 output differs by max3.13e-7. Native ONNX1024 vsTorch1024 differs by max1.71e-6. Every initializer is checked bitwise against512. Floating-point final states across packet sizes are compared numerically, and are not asserted bit-identical. Its independent4096-sample fixture uses four raw1024 packets from zero state/noise. See `organ-1024/native-validation.json` and `research/organ-1024-native.md`.

1024 packet availability is21.333ms. The40/32ms app profiles trade transport headroom against drop risk. WebGPU can benefit from halving the number of dispatches, even if one packet takes more milliseconds; only a browser live measurement can establish that benefit. A shared seed does not guarantee bit-identical random epsilon ordering across packet sizes; numerical cross-packet comparisons explicitly supply the same realized noise tensor.

## Attribution and terms

The IIL model repository and the pinned RAVE fork provide [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) terms. Preserve attribution to Intelligent Instruments Lab, RAVE/ACIDS/IRCAM, and the causal source/export work of Victor Shepardson. Changes are described above and in the model contract. No commercial-use permission is asserted. See the model directory's `ATTRIBUTION.md`.
