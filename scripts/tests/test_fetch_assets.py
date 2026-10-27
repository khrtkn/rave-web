"""Download recovery tests; no network, model files, or extra packages required."""
import contextlib
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
        with patch.object(FETCH.urllib.request,'urlopen',side_effect=open_response),patch.object(FETCH.time,'sleep'):
            FETCH.download('https://example.invalid/archive',self.destination,len(DATA))
        self.assertEqual(self.destination.read_bytes(),DATA)
        self.assertEqual(len(requests),2)

    def test_failed_launch_keeps_data_for_the_next_launch(self):
        def interrupted(request,timeout):
            offset=int((request.get_header('Range') or 'bytes=0-').split('=')[1].split('-')[0])
            return Response(DATA[offset:],206,f'bytes {offset}-{len(DATA)-1}/{len(DATA)}',interrupt=True)
        with patch.object(FETCH.urllib.request,'urlopen',side_effect=interrupted),patch.object(FETCH.time,'sleep'):
            with self.assertRaisesRegex(SystemExit,'launcher again'):
                FETCH.download('https://example.invalid/archive',self.destination,len(DATA))
        offset=self.destination.stat().st_size
        self.assertEqual(offset,5*65536)
        with patch.object(FETCH.urllib.request,'urlopen',return_value=Response(DATA[offset:],206,f'bytes {offset}-{len(DATA)-1}/{len(DATA)}')) as opener:
            FETCH.download('https://example.invalid/archive',self.destination,len(DATA))
        self.assertEqual(opener.call_args.args[0].get_header('Range'),f'bytes={offset}-')
        self.assertEqual(self.destination.read_bytes(),DATA)

    def test_server_without_ranges_restarts_instead_of_appending(self):
        self.destination.write_bytes(DATA[:65536])
        with patch.object(FETCH.urllib.request,'urlopen',return_value=Response(DATA)):
            FETCH.download('https://example.invalid/archive',self.destination,len(DATA))
        self.assertEqual(self.destination.read_bytes(),DATA)

    def test_unexpected_range_does_not_append_unrelated_bytes(self):
        prefix=DATA[:65536]
        self.destination.write_bytes(prefix)
        with patch.object(FETCH.urllib.request,'urlopen',return_value=Response(DATA,206,f'bytes 0-{len(DATA)-1}/{len(DATA)}')):
            with self.assertRaisesRegex(ValueError,'Unexpected download range'):
                FETCH.download('https://example.invalid/archive',self.destination,len(DATA))
        self.assertEqual(self.destination.read_bytes(),prefix)

    def test_complete_cache_does_not_download_again(self):
        self.destination.write_bytes(DATA)
        with patch.object(FETCH.urllib.request,'urlopen') as opener:
            FETCH.download('https://example.invalid/archive',self.destination,len(DATA))
        opener.assert_not_called()


if __name__=='__main__':unittest.main()
