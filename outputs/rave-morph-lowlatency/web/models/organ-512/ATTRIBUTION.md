# IIL Organ 512 model attribution

Adapted pretrained RAVE model; independent of Neutone Morpho.

- Model provider: Intelligent Instruments Lab (IIL).
- Pinned checkpoint/config: [organ_archive_b512_r48000 at c25a03d625840c40cd3a48779ed72f2a1947d7b4](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/tree/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000).
- Checkpoint SHA256: `218518cb0f7c2670a9cd3973c3e2972632893f599deacc851998b372a8f8a22e`; config SHA256: `975e0e8615204af6c9f8aa901275ac634d70f60011c898dbc103e1e4fb09ccb0`.
- RAVE research/architecture: Antoine Caillon and Philippe Esling / ACIDS-IRCAM.
- Causal source: [Victor Shepardson / RAVE ab7d40d6740130f0268debbc4b6633318357ea7d](https://github.com/victor-shepardson/RAVE/tree/ab7d40d6740130f0268debbc4b6633318357ea7d), [cached_conv d2da85992d030dcf1dd8ebf318c3d69753729295](https://github.com/victor-shepardson/cached_conv/tree/d2da85992d030dcf1dd8ebf318c3d69753729295).
- Model/source terms: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), as stated in the [pinned IIL model card](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/blob/c25a03d625840c40cd3a48779ed72f2a1947d7b4/README.md) and pinned source licenses. Retain attribution, source/license links, original notices and change descriptions. Commercial permission has not been obtained.

Changes: exported the learned causal encoder/decoder to ONNX with 36 explicit past states, explicit standard-normal posterior sampling, and gain/bias delta controls in six stored PCA directions. All 128 posterior coordinates are retained. Weight normalization is folded for inference. No retraining or replacement learned weights. See the adjacent numerical audit and contract.

The exact training-recording list for this b512 checkpoint is not established here. A separately published b2048 Organ model is a distinct artifact.

ONNX Runtime is a separate MIT-licensed runtime; retain its LICENSE and ThirdPartyNotices.txt when distributing it. Its terms do not replace the model's noncommercial conditions.
