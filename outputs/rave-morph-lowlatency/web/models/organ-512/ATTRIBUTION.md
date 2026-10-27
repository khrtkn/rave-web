# IIL Organ 512 model attribution

This is an adapted pretrained RAVE model, independent of Neutone Morpho.

- Model source: [Intelligent Instruments Lab RAVE models, pinned commit c25a03d625840c40cd3a48779ed72f2a1947d7b4](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/tree/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000).
- Immediate source checkpoint: `version_1/checkpoints/last.ckpt`, SHA256 `218518cb0f7c2670a9cd3973c3e2972632893f599deacc851998b372a8f8a22e`; matching `config.gin` SHA256 `975e0e8615204af6c9f8aa901275ac634d70f60011c898dbc103e1e4fb09ccb0`.
- Model license: the pinned [IIL model card](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/blob/c25a03d625840c40cd3a48779ed72f2a1947d7b4/README.md) identifies CC BY-NC 4.0. This adapted model is for noncommercial use under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- RAVE research/architecture: Antoine Caillon and Philippe Esling / ACIDS–IRCAM. Causal source fork used here: [Victor Shepardson RAVE, ab7d40d6740130f0268debbc4b6633318357ea7d](https://github.com/victor-shepardson/RAVE/tree/ab7d40d6740130f0268debbc4b6633318357ea7d), with [cached_conv d2da85992d030dcf1dd8ebf318c3d69753729295](https://github.com/victor-shepardson/cached_conv/tree/d2da85992d030dcf1dd8ebf318c3d69753729295). Their source licenses remain applicable.
- Adaptation on 2026-10-26: exported the actual learned causal 512-hop encoder/decoder to ONNX with 36 explicit past states, explicit variational Gaussian samples, and delta gain/bias controls in six stored PCA directions. All 128 posterior coordinates are retained. No retraining or replacement weights were used; PyTorch weight normalization was folded for inference. Native parity evidence is supplied alongside the model.

The source directory name identifies this checkpoint. No unsupported claim is made about an exact training-data archive or correspondence to a separately published TorchScript model. The unrelated 2048-hop Organ model from the earlier browser catalog is a distinct artifact.

ONNX Runtime is a separate MIT-licensed runtime; its license is provided with the browser runtime distribution. Model noncommercial terms are not replaced by the runtime license.
