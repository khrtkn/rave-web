#!/usr/bin/env python3
"""Fetch one pinned, unmodified public Djembe ONNX for a local build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

UPSTREAM_COMMIT = '2b1990c4ad04fec1981d21bdde94fdafbadb84e0'
SOURCE_BYTES = 16622944
SOURCE_SHA256 = 'ea94557a8d9b40f22a1dadbc93ab98f46144a68ce0bed1403a1558a2d1007f16'
SOURCE_URL = f'https://raw.githubusercontent.com/Torsion-Audio/Scyclone/{UPSTREAM_COMMIT}/assets/models/djembe.onnx'
SOURCE_PAGE = f'https://github.com/Torsion-Audio/Scyclone/blob/{UPSTREAM_COMMIT}/assets/models/djembe.onnx'
LICENSE_SOURCE_URL = f'https://github.com/Torsion-Audio/Scyclone/blob/{UPSTREAM_COMMIT}/README.md#licenses'
LICENSE_URL = 'https://creativecommons.org/licenses/by-nc/4.0/'
AUTHOR = 'Torsion Audio; RAVE by Antoine Caillon and Philippe Esling / ACIDS IRCAM'


def verify_source(path):
    """Read-only verification. A file with a different size/hash is rejected."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size != SOURCE_BYTES:
        raise ValueError('Djembe source byte size differs from the pinned upstream file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != SOURCE_SHA256:
        raise ValueError('Djembe source SHA-256 differs from the pinned upstream file')
    return {'bytes': SOURCE_BYTES, 'sha256': SOURCE_SHA256}


class PinnedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        # The raw.githubusercontent.com fixed-revision URL needs no redirects.
        # A redirect must not turn this pinned downloader into an arbitrary fetch.
        raise ValueError('Unexpected redirect while fetching pinned Djembe source')


def prepare_source(out):
    out = Path(out)
    if out.is_symlink():
        raise ValueError('Refusing to replace a symbolic-link output')
    if out.exists():
        verify_source(out)
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        opener = urllib.request.build_opener(PinnedRedirects())
        request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'rave-web-pinned-djembe-source/1'})
        with opener.open(request, timeout=60) as response:
            if response.status != 200 or response.geturl() != SOURCE_URL:
                raise ValueError('Unexpected response for pinned Djembe source')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != SOURCE_BYTES:
                raise ValueError('Upstream Djembe Content-Length differs from the pinned size')
            digest = hashlib.sha256()
            count = 0
            with tempfile.NamedTemporaryFile(dir=out.parent, prefix='.djembe-download-', delete=False) as stream:
                temporary = Path(stream.name)
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > SOURCE_BYTES:
                        raise ValueError('Djembe source exceeds the pinned byte size')
                    digest.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
        if count != SOURCE_BYTES or digest.hexdigest() != SOURCE_SHA256:
            raise ValueError('Djembe source byte size or SHA-256 verification failed')
        # No partially downloaded or unchecked file appears at the final path.
        os.replace(temporary, out)
        temporary = None
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--out', dest='output', type=Path, default=Path('work/djembe-source/djembe.onnx'))
    args = parser.parse_args()
    downloaded = prepare_source(args.output)
    print(json.dumps({'downloaded': downloaded, 'upstreamCommit': UPSTREAM_COMMIT,
                      'sourceURL': SOURCE_URL, **verify_source(args.output),
                      'author': AUTHOR, 'license': 'CC BY-NC 4.0',
                      'licenseSourceURL': LICENSE_SOURCE_URL, 'licenseURL': LICENSE_URL}, indent=2))


if __name__ == '__main__':
    main()
