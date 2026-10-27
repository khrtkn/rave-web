#!/usr/bin/env python3
"""Fetch only pinned ONNX Runtime Web files from the official npm registry.

Models are never downloaded. The two checked-in license files are also verified.
No npm installation, pip packages, or GitHub Release archives are used.
"""
import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_URL = 'https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.30.0.tgz'
RUNTIME_DIRECTORY = 'outputs/rave-morph/web/vendor/ort/'
RUNTIME_NAMES = {
    'ort.wasm.bundle.min.mjs', 'ort.webgpu.bundle.min.mjs',
    'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
    'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm',
}
LICENSE_NAMES = {'LICENSE', 'ThirdPartyNotices.txt'}
EXPECTED_LINKS = {
    'outputs/rave-morph-lowlatency/web/legacy': '../../rave-morph/web',
    'outputs/rave-morph-lowlatency/web/vendor': '../../rave-morph/web/vendor',
    'outputs/rave-morph-lowlatency/web/models/comparison-water': '../../../rave-morph/web/models',
    'outputs/rave-morph-lowlatency/web/models/comparison-djembe': '../../../rave-morph/web/models/djembe-candidate',
}


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.digest()


def sha(path):
    return digest(path).hex()


def safe_relative(relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Unsafe relative path')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in ('', '.', '..') for p in relative.split('/')):
        raise ValueError('Unsafe relative path: ' + relative)
    return parts


def inside(relative, allow_final_link=False):
    parts = safe_relative(relative)
    target = ROOT.joinpath(*parts.parts)
    if not target.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Asset escapes repository')
    checked = target.parents if allow_final_link else (target, *target.parents)
    for part in checked:
        if part == ROOT:
            break
        if part.is_symlink():
            raise ValueError('Unexpected symlink in asset path: ' + relative)
    return target


def load_manifest():
    manifest = json.loads((ROOT / 'scripts' / 'runtime-manifest.json').read_text())
    package = manifest['package']
    if (manifest['schemaVersion'] != 1 or package['name'] != 'onnxruntime-web'
            or package['version'] != '1.30.0' or package['url'] != PACKAGE_URL):
        raise ValueError('Expected the fixed official onnxruntime-web@1.30.0 package')
    if not isinstance(package['bytes'], int) or not 0 < package['bytes'] < 100_000_000:
        raise ValueError('Invalid package size')
    if not re.fullmatch(r'[0-9a-f]{64}', package['sha256']):
        raise ValueError('Invalid package SHA-256')
    algorithm, separator, integrity = package['integrity'].partition('-')
    if algorithm != 'sha512' or not separator or len(base64.b64decode(integrity, validate=True)) != 64:
        raise ValueError('Expected npm SHA-512 integrity')
    names = set()
    for entry in manifest['files']:
        name = entry['path'].removeprefix(RUNTIME_DIRECTORY)
        if (entry['path'] != RUNTIME_DIRECTORY + name or name in names
                or name not in RUNTIME_NAMES | LICENSE_NAMES):
            raise ValueError('Manifest must contain only the allowlisted runtime and license files')
        inside(entry['path'])
        if not isinstance(entry['bytes'], int) or not 0 < entry['bytes'] < 100_000_000:
            raise ValueError('Invalid runtime size')
        if not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Invalid runtime SHA-256')
        if name in RUNTIME_NAMES:
            if entry.get('tarPath') != 'package/dist/' + name or entry.get('bundled'):
                raise ValueError('Unexpected npm runtime path')
        elif entry.get('bundled') is not True or 'tarPath' in entry:
            raise ValueError('License files must be checked-in copies')
        names.add(name)
    if names != RUNTIME_NAMES | LICENSE_NAMES:
        raise ValueError('Manifest is missing runtime or license entries')
    if manifest.get('links') != EXPECTED_LINKS:
        raise ValueError('Manifest must retain only the four known local server aliases')
    return manifest


def matches(path, entry):
    return path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']


def verify(manifest):
    return [entry['path'] for entry in manifest['files'] if not matches(inside(entry['path']), entry)]


def restore_links(manifest):
    for relative, link in manifest['links'].items():
        path = inside(relative, allow_final_link=True)
        destination = (path.parent / link).resolve()
        if not destination.is_relative_to(ROOT.resolve()):
            raise ValueError('Link escapes repository')
        if path.is_symlink():
            if path.resolve() != destination:
                raise ValueError('Unexpected link: ' + relative)
            continue
        if path.is_file():
            if path.read_text().strip() != link:
                raise ValueError('Unexpected file instead of link: ' + relative)
            path.unlink()
        if path.is_dir():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(link, target_is_directory=True)
        except OSError:
            # serve.py resolves these four aliases on Windows without symlink permission.
            print('Using the local server alias for ' + relative)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise ValueError('The fixed official npm package must not redirect')


def open_url(request, timeout):
    return urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout)


def download(url, destination, total):
    """Resume bounded GET requests; the complete SHA/integrity is checked before use."""
    if url != PACKAGE_URL:
        raise ValueError('Only the fixed official npm tarball can be downloaded')
    if destination.is_symlink():
        raise ValueError('Unexpected symlink at the download cache')
    if destination.exists() and destination.stat().st_size > total:
        destination.unlink()
    last = -1
    last_report = time.monotonic()
    for attempt in range(5):
        done = destination.stat().st_size if destination.exists() else 0
        if done == total:
            return
        headers = {'User-Agent': 'rave-web-runtime/0.1', 'Accept-Encoding': 'identity'}
        if done:
            headers['Range'] = f'bytes={done}-'
            print(f'Resuming at {done / 1_000_000:.1f} MB.', flush=True)
        request = urllib.request.Request(url, headers=headers)
        try:
            with open_url(request, timeout=60) as response:
                if response.status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if content_range != f'bytes {done}-{total - 1}/{total}':
                        raise ValueError('Unexpected download range: ' + content_range)
                elif response.status == 200:
                    if done:
                        print('Server cannot resume; restarting the download.', flush=True)
                    done = 0
                else:
                    raise ValueError('Unexpected download response: ' + str(response.status))
                with destination.open('ab' if done else 'wb') as stream:
                    while True:
                        block = response.read(64 * 1024)
                        if not block:
                            break
                        if done + len(block) > total:
                            raise ValueError('Download exceeds the pinned npm tarball size')
                        stream.write(block)
                        done += len(block)
                        percent = int(done / total * 100)
                        if percent >= last + 5 or done == total or time.monotonic() - last_report >= 5:
                            print(f'Downloaded {done / 1_000_000:.1f} / {total / 1_000_000:.1f} MB ({percent}%)', flush=True)
                            last = percent
                            last_report = time.monotonic()
            if done != total:
                raise OSError('The download ended before the npm tarball was complete')
            return
        except (OSError, http.client.HTTPException) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise SystemExit(f'Official npm download failed (HTTP {error.code}).') from error
            print(f'Download interrupted ({attempt + 1}/5): {error}', flush=True)
            if attempt == 4:
                raise SystemExit('Download is incomplete. Run the launcher again to resume the saved data.') from error
            time.sleep(min(2 ** attempt, 8))


def verify_archive(archive, package):
    if not archive.is_file() or archive.stat().st_size != package['bytes'] or sha(archive) != package['sha256']:
        raise ValueError('Official npm tarball size/SHA-256 mismatch. Nothing was extracted.')
    integrity = 'sha512-' + base64.b64encode(digest(archive, 'sha512')).decode('ascii')
    if integrity != package['integrity']:
        raise ValueError('Official npm tarball integrity mismatch. Nothing was extracted.')


def extract_runtime(archive, manifest):
    """Stage and verify all selected files; never extract package code or model files."""
    verify_archive(archive, manifest['package'])
    bundled = [entry for entry in manifest['files'] if entry.get('bundled')]
    if any(not matches(inside(entry['path']), entry) for entry in bundled):
        raise ValueError('Checked-in runtime license files are missing or changed. Restore them from the repository.')
    entries = {entry['tarPath']: entry for entry in manifest['files'] if 'tarPath' in entry}
    seen = set()
    with tempfile.TemporaryDirectory(prefix='rave-ort-extract-') as directory:
        staged = {}
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                safe_relative(member.name.rstrip('/'))
                if not member.name.startswith('package/'):
                    raise ValueError('Unexpected npm archive root')
                if member.name not in entries:
                    continue
                if not member.isfile() or member.name in seen:
                    raise ValueError('Unexpected or duplicate runtime member: ' + member.name)
                entry = entries[member.name]
                if member.size != entry['bytes']:
                    raise ValueError('Runtime size mismatch: ' + member.name)
                temporary = Path(directory) / str(len(staged))
                with tar.extractfile(member) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if not matches(temporary, entry):
                    raise ValueError('Runtime hash mismatch: ' + member.name)
                staged[entry['path']] = temporary
                seen.add(member.name)
        if seen != set(entries):
            raise ValueError('Official npm tarball is missing expected runtime files')
        for relative, staged_path in staged.items():
            destination = inside(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Stage in the destination filesystem so the final replacement is atomic.
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                with staged_path.open('rb') as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
            try:
                os.chmod(temporary, 0o644)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--archive', type=Path, help='Use the exact official onnxruntime-web-1.30.0.tgz (not a GitHub Release)')
    mode.add_argument('--verify', action='store_true', help='Verify only the 8 runtime files and 2 bundled license files; no download')
    args = parser.parse_args()
    manifest = load_manifest()
    if args.verify:
        problems = verify(manifest)
        if problems:
            raise SystemExit('Missing or mismatched runtime files:\n' + '\n'.join(problems))
        restore_links(manifest)
        print('All 8 runtime files and 2 bundled license files match the pinned SHA-256 manifest.')
        return
    if not args.archive and not verify(manifest):
        restore_links(manifest)
        print('ONNX Runtime Web is already verified. Model files are not checked or downloaded.')
        return
    if any(entry.get('bundled') and not matches(inside(entry['path']), entry) for entry in manifest['files']):
        raise SystemExit('Checked-in runtime license files are missing or changed. Restore them from the repository.')
    package = manifest['package']
    archive = args.archive or inside('.cache/rave-runtime/onnxruntime-web-1.30.0.tgz.partial')
    if not args.archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading the pinned official npm runtime: ' + package['url'], flush=True)
        download(package['url'], archive, package['bytes'])
    try:
        extract_runtime(archive, manifest)
    except ValueError:
        if not args.archive and archive.exists():
            try:
                verify_archive(archive, package)
            except ValueError:
                archive.unlink(missing_ok=True)
        raise
    restore_links(manifest)
    if verify(manifest):
        raise SystemExit('Runtime verification failed')
    if not args.archive:
        archive.unlink(missing_ok=True)
    print('Restored and verified 8 runtime files; 2 bundled license files verified. No models were downloaded.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, tarfile.TarError) as error:
        raise SystemExit(str(error)) from error
