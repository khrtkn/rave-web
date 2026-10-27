#!/usr/bin/env python3
"""Download the verified runtime if needed, start locally, and open the browser."""
import sys
if sys.version_info < (3, 9):
    raise SystemExit('Python 3.9 or later is required. Install Python from https://www.python.org/downloads/ .')

import argparse
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import webbrowser
sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from model_catalog import verify

ROOT=Path(__file__).resolve().parent

def available_port(preferred):
    for port in range(preferred, preferred+20):
        try:
            with socket.socket() as probe:probe.bind(('127.0.0.1',port))
            return port
        except OSError:continue
    raise RuntimeError('No free local port. Close the previous demo and try again.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8873)
    parser.add_argument('--no-browser',action='store_true',help='Print the URL without opening a browser')
    args=parser.parse_args()
    if not 1024<=args.port<=65000:parser.error('Choose a port between 1024 and 65000.')
    print('RAVE Web Research — noncommercial research prototype',flush=True)
    catalog = verify()
    print('Local models: ' + ', '.join(model['id'] for model in catalog['models']), flush=True)
    print('First launch obtains ONNX Runtime Web from official npm (about 33 MB).',flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/fetch-assets.py')],cwd=ROOT,check=True)
    port=available_port(args.port)
    url='http://127.0.0.1:'+str(port)+'/'
    process=subprocess.Popen([sys.executable,str(ROOT/'serve.py'),'--port',str(port)],cwd=ROOT)
    try:
        for attempt in range(100):
            if process.poll() is not None:raise RuntimeError('The local server could not start.')
            try:
                with urllib.request.urlopen(url,timeout=.5) as response:
                    if response.status==200:break
            except OSError:time.sleep(.1)
        else:raise RuntimeError('The local server did not become ready.')
        print('\nOpen in Chrome or Edge: '+url,flush=True)
        print('Keep this window open. Press Ctrl+C to stop.',flush=True)
        if not args.no_browser:webbrowser.open(url)
        process.wait()
    except KeyboardInterrupt:pass
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as error:
        print('\nCould not start: '+str(error),file=sys.stderr)
        raise SystemExit(1)
