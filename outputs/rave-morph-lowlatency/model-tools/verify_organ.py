#!/usr/bin/env python3
"""Independent native runtime fixture, individual PCA controls, reset, and audio demo."""
import hashlib
import json
from pathlib import Path
import time
from organ_paths import options
OPTIONS=options(__doc__)
import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT=OPTIONS.models_dir
OUT=ROOT/'organ-512'


def error(a,b):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    return {'maxAbs':float(np.max(np.abs(d))),'rms':float(np.sqrt(np.mean(d*d)))}


def stats(a):
    a=np.asarray(a,dtype=np.float64)
    return {'peak':float(np.max(np.abs(a))),'rms':float(np.sqrt(np.mean(a*a))),'finite':bool(np.isfinite(a).all())}


def main():
    c=json.loads((OUT/'contract.json').read_text());path=ROOT/c['url']
    o=ort.SessionOptions();o.intra_op_num_threads=o.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])
    def zeros():return {s['input']:np.zeros(s['shape'],np.float32) for s in c['states']}
    def run(audio,noise,initial=None,control=None,control_start=0):
        st=zeros() if initial is None else {k:v.copy() for k,v in initial.items()};out=[]
        for i in range(audio.shape[-1]//512):
            gain=np.ones((1,6,1),np.float32);bias=np.zeros((1,6,1),np.float32)
            if control and i>=control_start:(gain if control[0]=='gain' else bias)[0,control[1],0]=control[2]
            r=dict(zip([x.name for x in session.get_outputs()],session.run(None,{**st,
                'audio':np.ascontiguousarray(audio[:,:,i*512:(i+1)*512]),
                'encode_noise':np.ascontiguousarray(noise[:,:,i:i+1]),'latent_gain':gain,'latent_bias':bias})))
            assert np.isfinite(r['output']).all()
            st={s['input']:r[s['output']] for s in c['states']};out.append(r['output'])
        return np.concatenate(out,-1),st
    # Publish the browser fixture first, with no warmup, normalization or playback DSP.
    t=np.arange(4096)/48000
    fixture_input=(.2*np.sin(2*np.pi*220*t)+.08*np.sin(2*np.pi*733*t)).astype(np.float32)
    fixture_input[1000:1400]+=.2*np.exp(-np.arange(400)/80).astype(np.float32)
    fixture_input=fixture_input[None,None];fixture_noise=np.zeros((1,128,8),np.float32)
    expected,st=run(fixture_input,fixture_noise);repeated,st2=run(fixture_input,fixture_noise)
    assert np.array_equal(expected,repeated) and all(np.array_equal(st[k],st2[k]) for k in st)
    fixture={'version':1,'model':c['id'],'modelSha256':c['sha256'],'frameSize':512,'sampleRate':48000,
        'frames':8,'channels':1,'samples':4096,'format':'IEEE754 little-endian float32 mono, no header',
        'initialStates':'all zero','allNoiseInputs':0,'latentGain':1,'latentBias':0,'warmupFrames':0,
        'outputAlignment':'candidate first8 raw packets, no warmup/shift/crop/fade/limiter',
        'tolerance':{'maxAbs':max(1e-7,stats(expected)['peak']*1e-4),'rms':max(1e-8,stats(expected)['rms']*2e-4)},
        'nativeOutputStats':stats(expected),'resetOutputAndAllStatesBitwiseIdentical':True}
    for key,name,data in [('input','parity-input.f32',fixture_input),('expectedOutput','parity-expected.f32',expected)]:
        raw=np.asarray(data,dtype='<f4').tobytes();(OUT/name).write_bytes(raw)
        fixture[key]={'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    (OUT/'parity-fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    print('FIXTURE_READY',json.dumps(fixture),flush=True)
    d=np.load(OPTIONS.work_dir/'organ-b512/demo-arrays.npz')
    audio=d['input'];noise=d['noise'];neutral=d['neutral']
    # Same warm state, signal, and explicitly fixed random epsilon in all axis comparisons.
    start=128
    _,warm=run(audio[:,:,:start*512],noise[:,:,:start])
    segment=audio[:,:,start*512:(start+80)*512];eps=noise[:,:,start:start+80]
    base,_=run(segment,eps,warm);axes=[]
    for dim in range(6):
        axis={'dimension':dim+1,'coordinate':'checkpoint PCA direction '+str(dim+1)}
        for kind,val in [('gain',.5),('bias',.5)]:
            y,_=run(segment,eps,warm,(kind,dim,val));difference=error(base,y)
            assert difference['rms']>1e-6,(dim,kind,difference)
            axis[kind]={'value':val,'difference':difference,'outputStats':stats(y)}
        axes.append(axis)
    # A genuine live control change after three seconds; return to neutral for the last second.
    modified=[];state=zeros()
    for i in range(audio.shape[-1]//512):
        active=3<=i*512/48000<5
        y,state=run(audio[:,:,i*512:(i+1)*512],noise[:,:,i:i+1],state,('bias',0,.5) if active else None)
        modified.append(y)
    changed=np.concatenate(modified,-1)
    # Persist float WAV to retain raw levels; do not silently normalize neural output.
    files={}
    for name,array in [('input',audio),('neutral',neutral),('knob-change',changed)]:
        filename=f'native-demo-{name}.wav';sf.write(OUT/filename,array[0,0],48000,subtype='FLOAT')
        files[name]={'file':filename,'stats':stats(array),'sha256':hashlib.sha256((OUT/filename).read_bytes()).hexdigest()}
    report={'fixture':fixture,'independentSixAxes':axes,'neutralStats':stats(neutral),'changedStats':stats(changed),
        'demo':{'seconds':audio.shape[-1]/48000,'input':'Locally synthesized harmonic note/envelope sequence; first1second is silent initialization. Actual pretrained model performs all conversion.',
                'noise':'Identical explicit standard-normal posterior samples, seed20261026','control':'PCA1 bias+0.5 from3to5seconds, neutral before/after; current state is retained',
                'format':'WAV IEEE float32, raw model levels, no normalization/limiter','files':files},
        'scope':'Native runtime and numerical/control behavior. Browser timings/physical audio latency and subjective quality require separate validation.',
        'passed':True}
    (OUT/'control-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
