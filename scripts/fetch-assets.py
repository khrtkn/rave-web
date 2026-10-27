#!/usr/bin/env python3
"""Restore runtime/model assets from a checksummed GitHub Release, without pip packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import http.client
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def inside(relative):
    part=PurePosixPath(relative)
    if part.is_absolute() or '..' in part.parts or '\\' in relative:
        raise ValueError('Unsafe asset path: '+relative)
    target=ROOT.joinpath(*part.parts)
    if not target.resolve().is_relative_to(ROOT):raise ValueError('Asset escapes repository')
    return target

def verify(manifest):
    problems=[]
    for entry in manifest['files']:
        path=inside(entry['path'])
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            problems.append(entry['path'])
    return problems

def restore_links(manifest):
    for relative,link in manifest['links'].items():
        path=inside(relative)
        destination=(path.parent/link).resolve()
        if not destination.is_relative_to(ROOT):raise ValueError('Link escapes repository')
        if path.is_symlink():
            if path.resolve()!=destination:raise ValueError('Unexpected link: '+relative)
            continue
        if path.is_file():
            if path.read_text().strip()!=link:raise ValueError('Unexpected file instead of link: '+relative)
            path.unlink()
        if path.is_dir():continue
        try:path.symlink_to(link,target_is_directory=True)
        except OSError:
            # The supplied HTTP server resolves these aliases without filesystem links.
            # Avoid copying hundreds of MB when Windows has no symlink permission.
            print('Using the local server alias for '+relative)

def download(url,destination,total):
    """Keep partial data across retries and launches; verify the full hash before use."""
    if destination.exists() and destination.stat().st_size>total:destination.unlink()
    last=-1;last_report=time.monotonic()
    for attempt in range(5):
        done=destination.stat().st_size if destination.exists() else 0
        if done==total:return
        headers={'User-Agent':'rave-web-research-assets/0.1','Accept-Encoding':'identity'}
        if done:
            headers['Range']=f'bytes={done}-'
            print(f'Resuming at {done/1_000_000:.1f} MB.',flush=True)
        request=urllib.request.Request(url,headers=headers)
        try:
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status==206:
                    content_range=response.headers.get('Content-Range','')
                    if content_range!=f'bytes {done}-{total-1}/{total}':
                        raise ValueError('Unexpected download range: '+content_range)
                elif response.status==200:
                    if done:print('Server cannot resume; restarting the download.',flush=True)
                    done=0
                else:raise ValueError('Unexpected download response: '+str(response.status))
                with destination.open('ab' if done else 'wb') as stream:
                    while True:
                        block=response.read(64*1024)
                        if not block:break
                        if done+len(block)>total:raise ValueError('Download exceeds the published archive size')
                        stream.write(block);done+=len(block)
                        percent=int(done/total*100)
                        if percent>=last+5 or done==total or time.monotonic()-last_report>=5:
                            print(f'Downloaded {done/1_000_000:.1f} / {total/1_000_000:.1f} MB ({percent}%)',flush=True)
                            last=percent;last_report=time.monotonic()
            if done!=total:raise OSError('The download ended before the archive was complete')
            return
        except (OSError,http.client.HTTPException) as error:
            print(f'Download interrupted ({attempt+1}/5): {error}',flush=True)
            if attempt==4:
                raise SystemExit('Download is incomplete. Run the launcher again to resume the saved data.') from error
            time.sleep(min(2**attempt,8))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,help='Use an already downloaded release archive')
    parser.add_argument('--verify',action='store_true',help='Verify local assets without downloading')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'assets-manifest.json').read_text())
    if args.verify:
        problems=verify(manifest)
        if problems:raise SystemExit('Missing or mismatched assets:\n'+'\n'.join(problems))
        restore_links(manifest);print('All runtime assets match the published SHA-256 manifest.');return
    if not verify(manifest):
        restore_links(manifest);print('Runtime assets are already verified.');return
    expected=manifest['archive']
    archive=args.archive or ROOT/'.cache'/'rave-assets'/(expected['name']+'.partial')
    if not args.archive:
        archive.parent.mkdir(parents=True,exist_ok=True)
        print('Downloading '+expected['url'],flush=True)
        download(expected['url'],archive,expected['bytes'])
    try:
        if archive.stat().st_size!=expected['bytes'] or sha(archive)!=expected['sha256']:
            if not args.archive:archive.unlink(missing_ok=True)
            raise SystemExit('Archive size/SHA-256 mismatch. Nothing was extracted.')
        entries={entry['path']:entry for entry in manifest['files']}
        seen=set()
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar:
                if not member.isfile() or member.name not in entries or member.name in seen:
                    raise ValueError('Unexpected archive member: '+member.name)
                entry=entries[member.name]
                if member.size!=entry['bytes']:raise ValueError('Asset size mismatch')
                destination=inside(member.name);destination.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source:
                    with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as output:
                        temporary_file=Path(output.name);shutil.copyfileobj(source,output,1024*1024)
                try:
                    if sha(temporary_file)!=entry['sha256']:raise ValueError('Asset hash mismatch: '+member.name)
                    os.replace(temporary_file,destination)
                finally:temporary_file.unlink(missing_ok=True)
                seen.add(member.name)
        if seen!=set(entries):raise ValueError('Archive is missing expected assets')
    finally:
        # Delete the cache only after all extracted assets verify successfully.
        if not args.archive and not verify(manifest):archive.unlink(missing_ok=True)
    restore_links(manifest)
    if verify(manifest):raise SystemExit('Asset verification failed')
    print('Restored and verified '+str(len(entries))+' runtime assets.')

if __name__=='__main__':main()
