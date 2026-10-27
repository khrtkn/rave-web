"""Registration must not expose missing, altered or failed local builds."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_catalog as catalog

class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / 'djembe'
        self.base.mkdir()
        model_id, filename, frame_size, source_sha = catalog.SPECS['djembe']
        data = b'test-only-stand-in'
        digest = hashlib.sha256(data).hexdigest()
        (self.base / filename).write_bytes(data)
        self.contract = {'id': model_id, 'frameSize': frame_size, 'sampleRate': 48000,
                         'validatedNative': True, 'sourceSha256': source_sha,
                         'bytes': len(data), 'sha256': digest}
        self.write('contract.json', self.contract)
        descriptor = {'file': 'fixture.f32', 'bytes': 4, 'sha256': hashlib.sha256(b'\0' * 4).hexdigest()}
        (self.base / 'fixture.f32').write_bytes(b'\0' * 4)
        self.fixture = {'model': model_id, 'modelSha256': digest, 'allNoiseInputs': 0,
                        'latentGain': 1, 'latentBias': 0, 'input': descriptor, 'expectedOutput': descriptor}
        self.write('parity-fixture.json', self.fixture)
        self.report = {'passed': True, 'sourceSha256': source_sha,
                       'allOriginalInitializersByteIdentical': True,
                       'resetOutputAndAllStateTensorsBitwiseIdentical': True,
                       'everyAudioAndStateOutputFinite': True}
        self.write('candidate-validation.json', self.report)
        self.write('noise-source-audit.json', {'candidateSha256': digest,
                   'sourceSha256': source_sha, 'nativeFixtureReloadParity': {'passed': True}})

    def write(self, name, value):
        (self.base / name).write_text(json.dumps(value))

    def test_djembe_only_registers_and_verifies(self):
        result = catalog.register(self.root)
        self.assertEqual(result['defaultModel'], self.contract['id'])
        self.assertEqual(result['models'][0]['url'], 'djembe/djembe-stateful-2048.onnx')
        self.assertEqual(catalog.verify(self.root), result)

    def test_model_tamper_does_not_replace_working_catalog(self):
        catalog.register(self.root)
        previous = (self.root / 'catalog.json').read_bytes()
        (self.base / 'djembe-stateful-2048.onnx').write_bytes(b'changed')
        with self.assertRaises(ValueError): catalog.register(self.root)
        self.assertEqual((self.root / 'catalog.json').read_bytes(), previous)

    def test_missing_fixture_rejects(self):
        (self.base / 'fixture.f32').unlink()
        with self.assertRaises(ValueError): catalog.register(self.root)

    def test_failed_numerical_report_rejects(self):
        self.report['passed'] = False
        self.write('candidate-validation.json', self.report)
        with self.assertRaises(ValueError): catalog.register(self.root)

    def test_different_source_rejects(self):
        self.contract['sourceSha256'] = '0' * 64
        self.write('contract.json', self.contract)
        with self.assertRaises(ValueError): catalog.register(self.root)

    def test_fixture_traversal_rejects(self):
        self.fixture['input']['file'] = '../outside.f32'
        self.write('parity-fixture.json', self.fixture)
        with self.assertRaises(ValueError): catalog.register(self.root)

    def test_fixture_symlink_escape_rejects(self):
        (self.root / 'outside.f32').write_bytes(b'\0' * 4)
        (self.base / 'fixture.f32').unlink()
        (self.base / 'fixture.f32').symlink_to(self.root / 'outside.f32')
        with self.assertRaises(ValueError): catalog.register(self.root)

    def test_catalog_url_tamper_rejects(self):
        result = catalog.register(self.root)
        result['models'][0]['url'] = 'https://unrelated.invalid/model.onnx'
        (self.root / 'catalog.json').write_text(json.dumps(result))
        with self.assertRaises(ValueError): catalog.verify(self.root)

    def test_no_models_returns_setup_instruction(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ValueError, 'MODEL_SOURCES.md'):
                catalog.verify(empty)

if __name__ == '__main__':
    unittest.main()
