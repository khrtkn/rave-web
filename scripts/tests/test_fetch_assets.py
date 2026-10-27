"""Pinned npm runtime recovery and safe-extraction tests; no network or models required."""
import base64
import contextlib
import hashlib
import json
import tarfile
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('fetch_assets',Path(__file__).resolve().parents[1]/'fetch-assets.py')
FETCH=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)
DATA=bytes(range(256))*2048


class Response(io.BytesIO):
    def __init__(self,data,status=200,content_range=None,interrupt=False):
        super().__init__(data)
        self.status=status
        self.headers={'Content-Range':content_range} if content_range else {}
        self.interrupt=interrupt

    def read(self,size=-1):
        if self.interrupt and self.tell()>=65536:raise OSError('Connection interrupted')
        return super().read(size)


class DownloadRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination=Path(self.temporary.name)/'archive.partial'
        self.output=contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__,None,None,None)

    def test_interrupted_response_resumes_from_saved_bytes(self):
        requests=[]
        def open_response(request,timeout):
            requests.append(request)
            if len(requests)==1:return Response(DATA,interrupt=True)
            offset=65536
            self.assertEqual(request.get_header('Range'),f'bytes={offset}-')
            return Response(DATA[offset:],206,f'bytes {offset}-{len(DATA)-1}/{len(DATA)}')
        with patch.object(FETCH,'open_url',side_effect=open_response),patch.object(FETCH.time,'sleep'):
            FETCH.download(FETCH.PACKAGE_URL,self.destination,len(DATA))
        self.assertEqual(self.destination.read_bytes(),DATA)
        self.assertEqual(len(requests),2)

    def test_failed_launch_keeps_data_for_the_next_launch(self):
        def interrupted(request,timeout):
            offset=int((request.get_header('Range') or 'bytes=0-').split('=')[1].split('-')[0])
            return Response(DATA[offset:],206,f'bytes {offset}-{len(DATA)-1}/{len(DATA)}',interrupt=True)
        with patch.object(FETCH,'open_url',side_effect=interrupted),patch.object(FETCH.time,'sleep'):
            with self.assertRaisesRegex(SystemExit,'launcher again'):
                FETCH.download(FETCH.PACKAGE_URL,self.destination,len(DATA))
        offset=self.destination.stat().st_size
        self.assertEqual(offset,5*65536)
        with patch.object(FETCH,'open_url',return_value=Response(DATA[offset:],206,f'bytes {offset}-{len(DATA)-1}/{len(DATA)}')) as opener:
            FETCH.download(FETCH.PACKAGE_URL,self.destination,len(DATA))
        self.assertEqual(opener.call_args.args[0].get_header('Range'),f'bytes={offset}-')
        self.assertEqual(self.destination.read_bytes(),DATA)

    def test_server_without_ranges_restarts_instead_of_appending(self):
        self.destination.write_bytes(DATA[:65536])
        with patch.object(FETCH,'open_url',return_value=Response(DATA)):
            FETCH.download(FETCH.PACKAGE_URL,self.destination,len(DATA))
        self.assertEqual(self.destination.read_bytes(),DATA)

    def test_unexpected_range_does_not_append_unrelated_bytes(self):
        prefix=DATA[:65536]
        self.destination.write_bytes(prefix)
        with patch.object(FETCH,'open_url',return_value=Response(DATA,206,f'bytes 0-{len(DATA)-1}/{len(DATA)}')):
            with self.assertRaisesRegex(ValueError,'Unexpected download range'):
                FETCH.download(FETCH.PACKAGE_URL,self.destination,len(DATA))
        self.assertEqual(self.destination.read_bytes(),prefix)

    def test_complete_cache_does_not_download_again(self):
        self.destination.write_bytes(DATA)
        with patch.object(FETCH,'open_url') as opener:
            FETCH.download(FETCH.PACKAGE_URL,self.destination,len(DATA))
        opener.assert_not_called()




class RuntimeExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patch = patch.object(FETCH, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.relative = FETCH.RUNTIME_DIRECTORY + 'ort.wasm.bundle.min.mjs'
        self.tar_path = 'package/dist/ort.wasm.bundle.min.mjs'
        self.data = b'pinned browser runtime'
        self.license_path = self.root / (FETCH.RUNTIME_DIRECTORY + 'LICENSE')
        self.license_path.parent.mkdir(parents=True)
        self.license_path.write_bytes(b'bundled license text')

    @staticmethod
    def entry(path, data, **fields):
        return {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), **fields}

    def archive(self, members, runtime_data=None):
        archive = self.root / 'official-npm-fixture.tgz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name, data, kind in members:
                member = tarfile.TarInfo(name)
                if kind == 'symlink':
                    member.type = tarfile.SYMTYPE
                    member.linkname = '../../outside'
                    tar.addfile(member)
                else:
                    member.size = len(data)
                    tar.addfile(member, io.BytesIO(data))
        body = archive.read_bytes()
        manifest = {
            'package': {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
                        'integrity': 'sha512-' + base64.b64encode(hashlib.sha512(body).digest()).decode()},
            'files': [self.entry(self.relative, self.data if runtime_data is None else runtime_data, tarPath=self.tar_path),
                      self.entry(FETCH.RUNTIME_DIRECTORY + 'LICENSE', self.license_path.read_bytes(), bundled=True)],
            'links': {},
        }
        return archive, manifest

    def test_extracts_only_allowlisted_runtime_and_keeps_bundled_license(self):
        archive, manifest = self.archive([(self.tar_path, self.data, 'file'),
                                         ('package/install.js', b'not executed or extracted', 'file'),
                                         ('package/models/unwanted.onnx', b'not extracted', 'file')])
        FETCH.extract_runtime(archive, manifest)
        self.assertEqual((self.root / self.relative).read_bytes(), self.data)
        self.assertEqual(self.license_path.read_bytes(), b'bundled license text')
        self.assertFalse((self.root / 'package').exists())
        self.assertEqual(FETCH.verify(manifest), [])
        # Model availability is not part of runtime verification.
        self.assertFalse(list(self.root.rglob('*.onnx')))

    def test_wrong_tarball_sha_is_rejected_before_any_write(self):
        archive, manifest = self.archive([(self.tar_path, self.data, 'file')])
        archive.write_bytes(archive.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'size/SHA-256 mismatch'):
            FETCH.extract_runtime(archive, manifest)
        self.assertFalse((self.root / self.relative).exists())

    def test_npm_integrity_is_checked_separately_from_sha256(self):
        archive, manifest = self.archive([(self.tar_path, self.data, 'file')])
        manifest['package']['integrity'] = 'sha512-' + base64.b64encode(bytes(64)).decode()
        with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
            FETCH.extract_runtime(archive, manifest)
        self.assertFalse((self.root / self.relative).exists())

    def test_member_hash_failure_does_not_install_staged_file(self):
        archive, manifest = self.archive([(self.tar_path, self.data, 'file')], runtime_data=b'x' * len(self.data))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            FETCH.extract_runtime(archive, manifest)
        self.assertFalse((self.root / self.relative).exists())

    def test_missing_member_does_not_replace_existing_runtime(self):
        current = self.root / self.relative
        current.write_bytes(b'keep existing bytes')
        archive, manifest = self.archive([('package/README.md', b'ignored', 'file')])
        with self.assertRaisesRegex(ValueError, 'missing expected runtime'):
            FETCH.extract_runtime(archive, manifest)
        self.assertEqual(current.read_bytes(), b'keep existing bytes')

    def test_path_traversal_rejected_even_for_unselected_member(self):
        archive, manifest = self.archive([(self.tar_path, self.data, 'file'),
                                         ('package/../../outside', b'unsafe', 'file')])
        with self.assertRaisesRegex(ValueError, 'Unsafe relative path'):
            FETCH.extract_runtime(archive, manifest)
        self.assertFalse((self.root / self.relative).exists())
        self.assertFalse((self.root.parent / 'outside').exists())

    def test_selected_symlink_and_duplicate_members_are_rejected(self):
        for members in [[(self.tar_path, b'', 'symlink')],
                        [(self.tar_path, self.data, 'file'), (self.tar_path, self.data, 'file')]]:
            with self.subTest(members=len(members)):
                archive, manifest = self.archive(members)
                with self.assertRaisesRegex(ValueError, 'Unexpected or duplicate'):
                    FETCH.extract_runtime(archive, manifest)
                self.assertFalse((self.root / self.relative).exists())

    def test_bundled_license_tampering_blocks_restoration(self):
        archive, manifest = self.archive([(self.tar_path, self.data, 'file')])
        self.license_path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'license files are missing or changed'):
            FETCH.extract_runtime(archive, manifest)
        self.assertFalse((self.root / self.relative).exists())

    def test_destination_parent_symlink_is_rejected(self):
        directory = self.root / 'other'
        directory.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Unexpected symlink'):
            FETCH.inside('alias/runtime.wasm')
        self.assertFalse((directory / 'runtime.wasm').exists())

    def test_nonofficial_download_url_is_rejected_without_network(self):
        with patch.object(FETCH, 'open_url') as opener:
            with self.assertRaisesRegex(ValueError, 'fixed official npm tarball'):
                FETCH.download('https://example.invalid/model.onnx', self.root / 'partial', 20)
        opener.assert_not_called()

    def test_redirects_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'must not redirect'):
            FETCH.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.invalid/asset')


class ManifestTests(unittest.TestCase):
    def test_manifest_is_runtime_only_and_pins_official_tarball(self):
        manifest = FETCH.load_manifest()
        self.assertEqual(len(manifest['files']), 10)
        self.assertEqual(sum('tarPath' in entry for entry in manifest['files']), 8)
        self.assertEqual(sum(entry.get('bundled', False) for entry in manifest['files']), 2)
        self.assertEqual(manifest['package']['url'], FETCH.PACKAGE_URL)
        self.assertEqual(len(manifest['links']), 4)
        self.assertFalse(any('.onnx' in entry['path'] for entry in manifest['files']))

    def test_model_entry_cannot_be_added_to_the_manifest(self):
        original = json.loads((FETCH.ROOT / 'scripts/runtime-manifest.json').read_text())
        original['files'][0]['path'] = FETCH.RUNTIME_DIRECTORY + 'model.onnx'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            (root / 'scripts/runtime-manifest.json').write_text(json.dumps(original))
            with patch.object(FETCH, 'ROOT', root), self.assertRaisesRegex(ValueError, 'only the allowlisted'):
                FETCH.load_manifest()


if __name__ == '__main__':
    unittest.main()
