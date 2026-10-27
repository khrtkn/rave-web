#!/usr/bin/env python3
"""Serve RAVE Morph static files on loopback. There is no inference/upload API."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import functools
from urllib.parse import unquote, urlsplit

WEB_ROOT = Path(__file__).resolve().parent / 'web'
BASELINE_ROOT = Path(__file__).resolve().parents[1] / 'rave-morph' / 'web'
ALIASES = {
    '/legacy': BASELINE_ROOT,
    '/vendor': BASELINE_ROOT / 'vendor',
    '/models/comparison-water': BASELINE_ROOT / 'models',
    '/models/comparison-djembe': BASELINE_ROOT / 'models' / 'djembe-candidate',
}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream'}

    def translate_path(self, path):
        clean = unquote(urlsplit(path).path)
        root, relative = WEB_ROOT, clean
        for prefix, destination in ALIASES.items():
            if clean == prefix or clean.startswith(prefix + '/'):
                root, relative = destination, clean[len(prefix):]
                break
        target = (root / relative.lstrip('/')).resolve()
        if not target.is_relative_to(root.resolve()):
            return str(root / '__not_found__')
        return str(target)

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; worker-src 'self' blob:; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; object-src 'none'; base-uri 'self'; form-action 'none'")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8873)
    args = parser.parse_args()
    root = WEB_ROOT
    with ThreadingHTTPServer(('127.0.0.1', args.port), functools.partial(Handler, directory=str(root))) as server:
        print(f'RAVE Morph Low Latency: http://127.0.0.1:{args.port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
