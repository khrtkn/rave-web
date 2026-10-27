#!/usr/bin/env python3
"""Fetch only pinned public RAVE source and the official public organ checkpoint."""
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

BASE=Path(__file__).resolve().parents[3]
WORK=BASE/'work/rave-lowlatency'
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


def main():
    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True);records=[]
    for name,relative,size,expected in FILES:
        path=CHECKPOINT_DIR/name
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(HF+relative,timeout=120) as response,partial.open('wb') as f:
                while True:
                    data=response.read(1024*1024)
                    if not data:break
                    f.write(data)
            partial.rename(path)
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
    destination=Path(__file__).resolve().parent/'source-manifest.json'
    destination.write_text(json.dumps({'filesAndRepositories':records,'license':'CC BY-NC 4.0',
        'modelLicenseSource':'https://huggingface.co/Intelligent-Instruments-Lab/rave-models/blob/'+REV+'/README.md',
        'scope':'Anonymous public GET and pinned public Git source only; no Neutone assets or authenticated endpoints.'},indent=2)+'\n')
    print('Pinned source verified:',destination)


if __name__=='__main__':main()
