#!/usr/bin/env python3
"""Fetch only pinned public RAVE source and the official public organ checkpoint."""
import hashlib
import json
import subprocess
import time
import http.client
import urllib.error
import urllib.request
from pathlib import Path
from organ_paths import options, verify_source_revisions

OPTIONS=options(__doc__)
WORK=OPTIONS.work_dir
CHECKPOINT_DIR=WORK/'organ-b512'
REV='c25a03d625840c40cd3a48779ed72f2a1947d7b4'
HF='https://huggingface.co/Intelligent-Instruments-Lab/rave-models/'
FILES=[
    ('config.gin','raw/'+REV+'/checkpoints/organ_archive_b512_r48000/config.gin',7087,'975e0e8615204af6c9f8aa901275ac634d70f60011c898dbc103e1e4fb09ccb0'),
    ('last.ckpt','resolve/'+REV+'/checkpoints/organ_archive_b512_r48000/version_1/checkpoints/last.ckpt',669961497,'218518cb0f7c2670a9cd3973c3e2972632893f599deacc851998b372a8f8a22e')]
REPOS=[('RAVE','https://github.com/victor-shepardson/RAVE.git','ab7d40d6740130f0268debbc4b6633318357ea7d'),
       ('cached_conv','https://github.com/victor-shepardson/cached_conv.git','d2da85992d030dcf1dd8ebf318c3d69753729295')]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fetch_verified(url,path,size,expected):
    """Resume a partial anonymous GET; never install an unverified final file."""
    partial=path.with_suffix(path.suffix+'.partial')
    for attempt in range(3):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset==size and sha(partial)==expected:
            partial.replace(path);return
        if offset>=size:
            raise ValueError('Invalid completed partial download; remove the .partial file and retry: '+path.name)
        headers={'User-Agent':'RAVE-Organ-source-preparation/1.0'}
        if offset:headers['Range']='bytes='+str(offset)+'-'
        try:
            request=urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(request,timeout=120) as response:
                append=offset>0 and response.status==206
                if append and not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'):
                    raise ValueError('Unexpected partial download range: '+path.name)
                with partial.open('ab' if append else 'wb') as f:
                    while True:
                        data=response.read(1024*1024)
                        if not data:break
                        f.write(data)
            if partial.stat().st_size!=size:
                raise OSError('Incomplete public download: '+path.name)
            if sha(partial)!=expected:
                raise ValueError('Public download SHA256 mismatch: '+path.name)
            partial.replace(path);return
        except (OSError,urllib.error.URLError,http.client.IncompleteRead):
            if attempt==2:raise
            time.sleep(attempt+1)


def main():
    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True);records=[]
    for name,relative,size,expected in FILES:
        path=CHECKPOINT_DIR/name
        if not path.exists():
            fetch_verified(HF+relative,path,size,expected)
        assert path.stat().st_size==size and sha(path)==expected,('Hash/size mismatch; existing file was not replaced',str(path))
        records.append({'file':name,'sourceURL':HF+relative,'bytes':size,'sha256':expected})
    for name,url,commit in REPOS:
        path=WORK/name
        if not path.exists():
            path.mkdir();subprocess.run(['git','init',str(path)],check=True)
            subprocess.run(['git','-C',str(path),'remote','add','origin',url],check=True)
            subprocess.run(['git','-C',str(path),'fetch','--depth','1','origin',commit],check=True)
            subprocess.run(['git','-C',str(path),'checkout','--detach',commit],check=True)
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        assert actual==commit,('Existing source has another revision; it was not reset',name,actual)
        records.append({'repository':url,'directory':name,'commit':commit})
    verify_source_revisions(WORK)
    destination=WORK/'source-manifest.json'
    destination.write_text(json.dumps({'filesAndRepositories':records,'license':'CC BY-NC 4.0',
        'modelLicenseSource':'https://huggingface.co/Intelligent-Instruments-Lab/rave-models/blob/'+REV+'/README.md',
        'scope':'Anonymous public GET and pinned public Git source only; no Neutone assets or authenticated endpoints.'},indent=2)+'\n')
    print('Pinned source verified:',destination)


if __name__=='__main__':main()
