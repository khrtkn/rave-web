#!/usr/bin/env python3
"""Read-only original-file integrity and learned-weight identity audit."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx

from build_water_latent import SOURCE_SHA256, source_bytes

EXPECTED = {
    "ensemble-app.js": "5dc4a91e3fa9b930796be6ea4f65c6ebf103ff22dc27b1e0ca8aff68a4a24e5c",
    "public-runtime/live/stream-worker.js": "3308820a1ab558e114262bab1aa04c8a1a4aca8aa032a0d7cd3a925fdc6c5e25",
    "public-runtime/live/render-clip.mjs": "31e859e58426031a15d72cd3697009a43acb25767150d27d13b047352544c8ba",
    "public-runtime/live/catalog.json": "33b9617096b597fed7c12006114b3399b9647402c0d30217d2055f4eb7d250e2",
    "public-runtime/asset-cache.js": "bf0abe04966db23aa3eaac4ea1650db51ecb0b6d167d36b198ba109dafb0a056",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original public-runtime directory; source hashes are verified")
    parser.add_argument("--models", type=Path, default=Path(__file__).resolve().parents[1] / "web/models")
    args = parser.parse_args()
    checks = []
    for name, expected in EXPECTED.items():
        p = args.source.parent / name
        actual = sha(p.read_bytes())
        checks.append({"path": str(p), "sha256": actual, "expectedSha256": expected, "unchanged": actual == expected})
        assert actual == expected, name
    original, _ = source_bytes(args.source)
    src = onnx.load_from_string(original)
    originals = {x.name: x.SerializeToString() for x in src.graph.initializer}
    learned = [name for name in originals if name.startswith("model.")]
    variants = []
    for n in (2048, 4096):
        target = args.models / f"water-latent-{n}.onnx"
        model = onnx.load(target)
        initializers = {x.name: x.SerializeToString() for x in model.graph.initializer}
        assert set(originals) == set(initializers)
        changed = [name for name in originals if originals[name] != initializers[name]]
        assert not any(name in learned for name in changed)
        assert changed == ([] if n == 4096 else ["/Constant_182_output_0", "/ScatterND_output_0"]), changed
        assert len(model.graph.node) == len(src.graph.node) + 2
        variants.append({"file": target.name, "sha256": sha(target.read_bytes()), "nodeCount": len(model.graph.node),
                         "learnedInitializerCount": len(learned), "allLearnedInitializersByteIdentical": True,
                         "changedInitializerNames": changed})
    report = {"checkedDate": "2026-10-26", "sourceModelSha256": SOURCE_SHA256, "originalFileChecks": checks,
              "originalFilesAllUnchanged": True, "variants": variants,
              "scope": "Five original app/runtime/catalog/cache files plus the reassembled model SHA and all model.* learned initializers. Other original directories were not modified or globally integrity-checked."}
    (args.models / "source-integrity.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
