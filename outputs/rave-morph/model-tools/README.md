# RAVE latent model tools

This is an independent RAVE implementation. It does not contain Neutone Morpho models or implement Morpho's proprietary engine.

`build_water_latent.py` reconstructs the **existing local RAVE water ONNX**, checks its exact SHA256, adds two input-controlled affine nodes at the actual 16-channel encoder-to-decoder latent edge, and validates the result using native ONNX Runtime. It also generates a 2048-sample variant. No learned weights are changed; no substitute network is created.

## Reproduce

Use Python 3.12 in a workspace-local virtual environment. From the task workspace:

```sh
python3.12 -m venv work/rave-morph-model-venv
work/rave-morph-model-venv/bin/python -m pip install -r outputs/rave-morph/model-tools/requirements.txt
work/rave-morph-model-venv/bin/python outputs/rave-morph/model-tools/build_water_latent.py --source /absolute/path/to/cloudflare-live/public-runtime
work/rave-morph-model-venv/bin/python outputs/rave-morph/model-tools/validate_controls.py
work/rave-morph-model-venv/bin/python outputs/rave-morph/model-tools/audit_sources.py --source /absolute/path/to/cloudflare-live/public-runtime
```

The original source directory must contain `live/catalog.json` and all water model chunk files listed there. It is read only and must be provided with `--source`. The builder refuses a different source hash. Dependencies are pinned to versions used in this run; the browser runtime is managed separately by the web app. The checked runtime binaries can also be restored with the repository's `scripts/fetch-assets.py` without rebuilding these historical comparison models.

Outputs go to `../web/models/`: `catalog.json`, `contract.json`, 2048/4096 ONNX models, native validation, source-integrity evidence, and the binary parity fixture. `validate_controls.py` separately verifies each of six dimensions' gain and bias, reset reproducibility, and writes the 8-frame fixture. Browser parity is verified by the web app against this fixture; native success is not a browser performance claim.

## Actual graph changes

The retained latent edge `/Gather_1_output_0` has 16 channels selected from a 128-channel sampled posterior. It is after `model.gimbal` per-channel affine normalization and before the graph mixes in the remaining 112 decoder-noise channels, reverses the gimbal affine, and feeds `model.decoder.net.0`.

We insert `z_controlled = z * latent_gain + latent_bias` on that retained edge only. Both inputs are float32 `[1,16,1]`; neutral is gain 1 and bias 0. The browser exposes the first six retained coordinates. They are **learned latent coordinates, not PCA axes or known semantic descriptors**. The exported graph contains no PCA projection matrix.

The controlled latent is written into decoder history `state_15` after the existing inverse normalization, so decoder state naturally carries the previously controlled latent across packets. Changing a knob does not retroactively rewrite that history. Smooth controls and retain all state tensors during a stream; reset means zero every state tensor.

The 2048 variant preserves every learned initializer and every state shape. It shortens the periodic PQMF sign-mask constant and the noise-synth time-reshape constant and updates shape declarations. Original 4096 output versus two successive 2048 packets is tested using the same segmented noise tensors and continued state.

The encoder's total stride is 2048. A direct 1024-sample specialization would not produce one full encoder step under this graph contract. Lower-hop processing needs additional architectural/export investigation; changing the declared sample count alone is not a validated solution.

See `GRAPH_NOTES.md` for the latency limits and `../web/models/ATTRIBUTION.md` for provenance and noncommercial terms.

## Djembe and the combined app catalog

The separate Djembe conversion is documented in `../web/models/djembe-candidate/README.md`. It preserves a different real RAVE model with four retained latent coordinates, adds streaming state and a known relative delay, and remains experimental. Run `build_djembe_candidate.py` and `audit_djembe_candidate.py` as described there. Neither script edits the original source model or the Water artifacts.

After both model builds, refresh the app's combined catalog:

```sh
python3 outputs/rave-morph/model-tools/update_catalog.py --default djembe-stateful-2048
```

For a Water-only build, pass `--default water-latent-2048` instead. The app uses `web/models/catalog.json`; `contract.json` remains a per-model contract and is not the current selection. Runtime/model dependencies are already bundled for ordinary browser use.

## Separate smaller candidate

`build_djembe_candidate.py` and `audit_djembe_candidate.py` produce and inspect a separate 16.6 MB stateful Djembe candidate. It has four retained projected latent axes, the same 2048 encoder stride, and a validated added causalization shift of 1360 samples. Water outputs are not rewritten by these scripts. See `../web/models/djembe-candidate/README.md` for its strict zero-noise parity, approximate normal-noise error, raw-output overshoot, native timings, source bounds, and browser fixture.
