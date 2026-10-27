#!/usr/bin/env python3
"""Check unchanged RAVE.forward versus neutral controls with identical sampled noise."""
import json
from pathlib import Path
import torch
from build_organ import load_source, Controlled, OUT, err


def main():
    torch.set_num_threads(1)
    model,_=load_source();controlled=Controlled(model).eval()
    t=torch.arange(16384)/48000
    audio=(.13*torch.sin(2*torch.pi*220*t)+.05*torch.sin(2*torch.pi*733*t))[None,None]
    with torch.no_grad():
        torch.manual_seed(1324);original=model(audio)
        torch.manual_seed(1324);epsilon=torch.randn(1,128,32)
        neutral=controlled(audio,epsilon,torch.ones(1,6,1),torch.zeros(1,6,1))
    e=err(original.numpy(),neutral.numpy());assert e['maxAbs']==0,e
    report={'sourceForwardVsNeutralControlledError':e,'samples':16384,
        'sourceForwardTemperature':1,'epsilon':'torch.randn_like original distribution, fixed seed1324; identical explicit epsilon',
        'retainedPosteriorDimensions':128,'controlledPCADirections':6,'passed':True}
    (OUT/'source-forward-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
