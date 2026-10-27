# IIL Organ 1024 packet model attribution

Same actual pretrained Organ checkpoint and learned weights as [the512 export](../organ-512/ATTRIBUTION.md), with the same CC BY-NC4.0 terms, authors, pinned sources and checkpoint/config hashes. This is independent of Neutone Morpho.

Adaptation2026-10-26: batch two true512-stride latent frames into each1024-sample audio packet. Preserve all128 posterior dimensions,36 explicit states and six PCA delta controls. No retraining, quantization or new model weights. Every ONNX initializer is bitwise identical to the512 export. Only fixed packet shapes and trace constants differ.

Source: [IIL checkpoint c25a03d](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/tree/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000). License: [CC BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/). See the full attribution linked above for RAVE/IRCAM, Victor Shepardson forks and runtime licenses.

1024 samples is the packet size, not a new encoder compression ratio or a measured physical latency. No additional shift is needed against the native reference; absolute dry/wet and physical round-trip delay remain unmeasured.
