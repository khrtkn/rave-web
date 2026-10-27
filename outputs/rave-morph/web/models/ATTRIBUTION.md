# Model attribution and permitted scope

These are modified **RAVE water / Pond Brain** models. They are not Neutone Morpho models.

- Source model family and author: Intelligent Instruments Lab (IIL), water recordings for Jenna Sutela's *Pond Brain* installations. The source model card describes the water model as modified RAVE v1, 48 kHz, block 2048, 16 controllable latent dimensions, trained on Freesound water recordings.
- Primary source card, pinned repository revision: [IIL RAVE models](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/blob/c25a03d625840c40cd3a48779ed72f2a1947d7b4/README.md). Upstream file named by that card: `water_pondbrain_b2048_r48000_z16.ts`.
- Model license: [Creative Commons Attribution–NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/). Attribution, noncommercial use, and the license's other requirements apply to these adapted weights. This prototype has no commercial-use authorization.
- Immediate adaptation source: the existing saved browser application's stateful ONNX, identified as Water / IIL / Pond Brain in its `live/catalog.json`. Original ONNX SHA256: `b6c84085f475b3bddf69eb2ee0a85836230e0462d7d112cc3a2cb6e497914e47`, 103,384,587 bytes. Reassembled from 198 local chunks without altering them.
- Changes made on 2026-10-26: added float32 `latent_gain` and `latent_bias` inputs at the actual retained latent boundary; created a 2048 specialization by changing two block-dependent constants and shape declarations. No learned weights were retrained or replaced. The graph contains learned gimbal-affine coordinates; this adaptation does not claim PCA or semantic axes.

The source-card TorchScript and the immediate ONNX are distinct artifacts. We do not assert identical hashes, an unobserved converter implementation, or parity against the upstream TorchScript. Numerical tests compare the adapted graphs against the exact immediate ONNX above.

The source card says the models are causal/streaming exports and warns of potentially loud initialization chirps from Pond Brain decoders. The application must manage initialization and listening gain. The native fixture is raw output with no limiter or fade, for numerical comparison only.

The RAVE research/code project is [ACIDS / IRCAM RAVE](https://github.com/acids-ircam/RAVE); the IIL card links its own fork/export history. Its code license does not replace the model's CC BY-NC terms. ONNX Runtime has its separate MIT license, supplied by the application runtime distribution.
