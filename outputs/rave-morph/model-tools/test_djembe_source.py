"""Small offline checks for pinned download integrity and atomic publication."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_djembe_source as source


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, length=None):
        super().__init__(body)
        self.headers = {} if length is None else {'Content-Length': str(length)}

    def geturl(self):
        return source.SOURCE_URL


class Opener:
    def __init__(self, body, length=None):
        self.body, self.length, self.calls = body, length, 0

    def open(self, request, timeout):
        assert request.full_url == source.SOURCE_URL and timeout == 60
        self.calls += 1
        return Response(self.body, self.length)


class DjembeSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.target = self.folder/'djembe.onnx'
        self.good = b'pinned-test-model'
        self.stack = [patch.object(source, 'SOURCE_BYTES', len(self.good)),
                      patch.object(source, 'SOURCE_SHA256', hashlib.sha256(self.good).hexdigest())]
        for entry in self.stack:
            entry.start()
            self.addCleanup(entry.stop)

    def download(self, body, length=None):
        opener = Opener(body, length)
        with patch.object(source.urllib.request, 'build_opener', return_value=opener):
            result = source.prepare_source(self.target)
        return result, opener

    def test_success_verifies_then_atomically_creates_and_reuses(self):
        result, opener = self.download(self.good, len(self.good))
        self.assertTrue(result)
        self.assertEqual(self.target.read_bytes(), self.good)
        self.assertEqual(opener.calls, 1)
        result, unused = self.download(b'never fetched')
        self.assertFalse(result)
        self.assertEqual(unused.calls, 0)
        self.assertEqual(list(self.folder.iterdir()), [self.target])

    def test_bad_hash_truncated_and_oversized_downloads_leave_no_final_or_partial_file(self):
        for body in [b'x'*len(self.good), self.good[:-1], self.good+b'x']:
            with self.subTest(length=len(body)):
                with self.assertRaises(ValueError):
                    self.download(body)
                self.assertFalse(self.target.exists())
                self.assertEqual(list(self.folder.iterdir()), [])

    def test_wrong_content_length_is_rejected(self):
        with self.assertRaises(ValueError):
            self.download(self.good, len(self.good)+1)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_existing_corrupt_file_and_symlink_are_not_replaced(self):
        self.target.write_bytes(b'keep existing data')
        with self.assertRaises(ValueError):
            self.download(self.good)
        self.assertEqual(self.target.read_bytes(), b'keep existing data')
        self.target.unlink()
        original = self.folder/'original.onnx'
        original.write_bytes(self.good)
        self.target.symlink_to(original.name)
        with self.assertRaises(ValueError):
            self.download(self.good)
        self.assertTrue(self.target.is_symlink())
        self.assertEqual(original.read_bytes(), self.good)

    def test_redirects_are_rejected(self):
        with self.assertRaises(ValueError):
            source.PinnedRedirects().redirect_request(None, None, 302, 'Moved', {}, 'https://other.test/model.onnx')


if __name__ == '__main__':
    unittest.main()
