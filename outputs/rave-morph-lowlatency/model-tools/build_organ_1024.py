#!/usr/bin/env python3
"""Add a 1024-sample packet export of the verified 512-stride Organ checkpoint.

No retraining, quantization, source model edits, or default-model change.
"""
from __future__ import annotations
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from organ_paths import options, update_catalog
OPTIONS=options(__doc__)

import numpy as np
import onnx
import onnxruntime as ort
import torch
import build_organ as original

MODELS=OPTIONS.models_dir
SOURCE=MODELS/'organ-512'
OUT=MODELS/'organ-1024'
FRAME=1024

def stats(x):
    a=np.asarray(x,dtype=np.float64)
    return {'peak':float(np.max(np.abs(a))),'rms':float(np.sqrt(np.mean(a*a))),'finite':bool(np.isfinite(a).all())}

def native_stream(model,audio,noise,frame):
    state=model.zeros();out=[];times=[];latent_frames=frame//512
    gain=torch.ones(1,6,1);bias=torch.zeros_like(gain)
    with torch.no_grad():
        for i in range(audio.shape[-1]//frame):
            start=time.perf_counter()
            y,*state=model(audio[:,:,i*frame:(i+1)*frame],noise[:,:,i*latent_frames:(i+1)*latent_frames],gain,bias,*state)
            times.append((time.perf_counter()-start)*1000);out.append(y)
    return torch.cat(out,-1),state,times

def ort_stream(session,audio,noise,spec,frame,gain=None,bias=None):
    state={s['input']:np.zeros(s['shape'],np.float32) for s in spec};out=[];times=[]
    gain=np.ones((1,6,1),np.float32) if gain is None else gain
    bias=np.zeros((1,6,1),np.float32) if bias is None else bias
    names=[x.name for x in session.get_outputs()];latent_frames=frame//512
    for i in range(audio.shape[-1]//frame):
        feeds={**state,'audio':np.ascontiguousarray(audio[:,:,i*frame:(i+1)*frame]),
               'encode_noise':np.ascontiguousarray(noise[:,:,i*latent_frames:(i+1)*latent_frames]),
               'latent_gain':gain,'latent_bias':bias}
        start=time.perf_counter();r=dict(zip(names,session.run(None,feeds)))
        times.append((time.perf_counter()-start)*1000);out.append(r['output'])
        assert np.isfinite(r['output']).all()
        state={s['input']:r[s['output']] for s in spec}
    return np.concatenate(out,-1),state,times

def timing(values,warmup=48):
    values=np.asarray(values[warmup:],float)
    return {'median':float(np.median(values)),'p95':float(np.percentile(values,95)),
            'max':float(np.max(values)),'measuredFrames':len(values)}

def initializer_signatures(model):
    return {x.name:{'dtype':x.data_type,'shape':list(x.dims),
                    'sha256':hashlib.sha256(x.SerializeToString()).hexdigest()}
            for x in model.graph.initializer}

def write_fixture(session,spec,model_hash):
    # Same source signal definition as 512 fixture; independently computed 4-packet output.
    t=np.arange(4096)/48000
    audio=(.2*np.sin(2*np.pi*220*t)+.08*np.sin(2*np.pi*733*t)).astype(np.float32)
    audio[1000:1400]+=.2*np.exp(-np.arange(400)/80).astype(np.float32)
    audio=audio[None,None];noise=np.zeros((1,128,8),np.float32)
    expected,state,_=ort_stream(session,audio,noise,spec,FRAME)
    repeated,state2,_=ort_stream(session,audio,noise,spec,FRAME)
    assert np.array_equal(expected,repeated) and all(np.array_equal(state[k],state2[k]) for k in state)
    st=stats(expected)
    fixture={'version':1,'model':'organ-causal-1024','modelSha256':model_hash,'frameSize':FRAME,'sampleRate':48000,
             'frames':4,'channels':1,'samples':4096,'format':'IEEE754 little-endian float32 mono, no header',
             'initialStates':'all zero','allNoiseInputs':0,'latentGain':1,'latentBias':0,'warmupFrames':0,
             'outputAlignment':'candidate first4 raw1024 packets, no warmup/shift/crop/fade/limiter',
             'tolerance':{'maxAbs':max(1e-7,st['peak']*1e-4),'rms':max(1e-8,st['rms']*2e-4)},
             'nativeOutputStats':st,'resetOutputAndAllStatesBitwiseIdentical':True}
    for key,name,data in [('input','parity-input.f32',audio),('expectedOutput','parity-expected.f32',expected)]:
        raw=np.asarray(data,dtype='<f4').tobytes();(OUT/name).write_bytes(raw)
        fixture[key]={'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    (OUT/'parity-fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    return fixture

def write_contract(source_contract,meta,spec):
    c=copy.deepcopy(source_contract)
    c.update({'id':'organ-causal-1024','name':'Organ · RAVE 1024 packet / PCA6','family':'organ','familyName':'Organ / IIL',
              'url':'organ-1024/organ-causal-1024.onnx','frameSize':FRAME,'audioInputShape':[1,1,FRAME],
              'audioOutputShape':[1,1,FRAME],'noiseInputs':[{'name':'encode_noise','shape':[1,128,2],'distribution':'normal'}],
              'encoderStrideSamples':512,'latentFramesPerPacket':2,'states':spec,'warmupBlocks':64,'warmupFrames':64,
              'blockDurationMs':FRAME/48,'bytes':meta['onnxBytes'],'sha256':meta['onnxSha256'],
              'parityFixture':'organ-1024/parity-fixture.json','additionalDelaySamples':0,'dryOffsetSamples':0,
              'modelDelaySamples':None,'sourceModelDelaySamples':None,
              'latencyNote':'1024-sample packet batches two true512-stride latent frames. Reference output has no added shift. Packet availability is21.333ms; original absolute model/physical latency remains unmeasured.',
              'adaptation':'Same verified causal Organ checkpoint and learned weights as512 export;1024-sample packets,36 unchanged explicit past states,128x2 posterior Gaussian epsilon, six PCA delta controls broadcast across two latent frames. No retraining or quantization.',
              'validatedNative':True,'experimental':True})
    (OUT/'contract.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    update_catalog(c,OPTIONS)
    return c

def main():
    torch.set_num_threads(1);torch.manual_seed(20261026);OUT.mkdir(parents=True,exist_ok=True)
    source_onnx=SOURCE/'organ-causal-512.onnx'
    source_contract=json.loads((SOURCE/'contract.json').read_text())
    source_onnx_sha=original.sha(source_onnx)
    assert source_onnx_sha==source_contract['sha256']
    assert source_contract['sourceCheckpointSha256']==original.CHECKPOINT_SHA
    source_initializers=initializer_signatures(onnx.load(source_onnx))
    model,audit=original.load_source()
    reference=original.Controlled(model).eval();stream=original.Stateful(reference).eval()
    spec=stream.context.spec
    assert spec==source_contract['states'] and len(spec)==36
    d=np.load(OPTIONS.work_dir/'organ-b512/demo-arrays.npz')
    audio=torch.from_numpy(d['input']);noise=torch.from_numpy(d['noise']);n=audio.shape[-1]
    assert n%FRAME==0 and noise.shape==(1,128,n//512)
    gain=torch.ones(1,6,1);bias=torch.zeros_like(gain)
    with torch.no_grad():full=reference(audio,noise,gain,bias)
    out1024,state1024,t1024=native_stream(stream,audio,noise,FRAME)
    out512,state512,t512=native_stream(stream,audio,noise,512)
    full_error=original.err(full.numpy(),out1024.numpy());packet_error=original.err(out512.numpy(),out1024.numpy())
    assert full_error['maxAbs']<1e-4 and full_error['rms']<1e-5,full_error
    assert packet_error['maxAbs']<1e-4 and packet_error['rms']<1e-5,packet_error
    print('NATIVE_REFERENCE',json.dumps({'longVs1024':full_error,'512Vs1024':packet_error}),flush=True)
    path=OUT/'organ-causal-1024.onnx'
    x=torch.randn(1,1,FRAME)*.05;eps=torch.zeros(1,128,2);zero=stream.zeros()
    torch.onnx.export(stream,(x,eps,gain,bias,*zero),str(path),opset_version=17,
        input_names=['audio','encode_noise','latent_gain','latent_bias']+[s['input'] for s in spec],
        output_names=['output']+[s['output'] for s in spec],do_constant_folding=True)
    exported=onnx.load(path);onnx.checker.check_model(exported)
    assert initializer_signatures(exported)==source_initializers,'Learned initializer changed between packet sizes'
    session=original.ort_session(path);session512=original.ort_session(source_onnx)
    ort1024,os1024,ot1024=ort_stream(session,audio.numpy(),noise.numpy(),spec,FRAME)
    ort512,os512,ot512=ort_stream(session512,audio.numpy(),noise.numpy(),spec,512)
    onnx_error=original.err(out1024.numpy(),ort1024);onnx_packet_error=original.err(ort512,ort1024)
    assert onnx_error['maxAbs']<1e-3 and onnx_error['rms']<1e-4,onnx_error
    assert onnx_packet_error['maxAbs']<1e-3 and onnx_packet_error['rms']<1e-4,onnx_packet_error
    state_errors={s['input']:original.err(os512[s['input']],os1024[s['input']]) for s in spec}
    model_hash=original.sha(path);fixture=write_fixture(session,spec,model_hash)
    controls=[]
    # Six individual coordinates, same full source/noise/state reset, controls broadcast across both latent frames.
    for dim in range(6):
        cg=np.ones((1,6,1),np.float32);cb=np.zeros_like(cg);cb[0,dim,0]=.5
        changed,_,_=ort_stream(session,audio.numpy()[:,:,:8192],noise.numpy()[:,:,:16],spec,FRAME,cg,cb)
        neutral=ort1024[:,:,:8192];diff=original.err(neutral,changed)
        assert diff['rms']>1e-6,(dim,diff)
        controls.append({'dimension':dim+1,'bias':.5,'difference':diff,'finite':stats(changed)['finite']})
    meta={'environment':{'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__,
                         'platform':platform.platform(),'threads':1},
          'source512ModelSha256':source_onnx_sha,'checkpointSha256':original.CHECKPOINT_SHA,'configSha256':original.CONFIG_SHA,
          'sourceWeightAudit':'../organ-512/native-validation.json','sourceForwardAudit':'../organ-512/source-forward-audit.json',
          'sourceStateLoadMissingKeys':audit['sourceMissingNonInferenceBuffer'],'allLearnedStateLoaded':audit['allLearnedStateLoaded'],
          'all512InitializersBitwiseIdentical':True,'initializerCount':len(source_initializers),
          'sampleRate':48000,'frameSize':FRAME,'encoderStrideSamples':512,'encodedLatentShape':[1,128,2],
          'controlsShape':[1,6,1],'stateCount':len(spec),'samplesTested':n,
          'nativeLongVsStreamError':full_error,'native512Vs1024StreamError':packet_error,
          'onnxVsNativeStreamError':onnx_error,'onnx512Vs1024StreamError':onnx_packet_error,
          'onnx512Vs1024FinalStates':state_errors,'additionalStreamingShiftSamples':0,'sourceAbsoluteAlignmentSamples':None,
          'nativeStreamMs':timing(t1024),'ortStreamMs':timing(ot1024),
          'comparison512NativeMs':timing(t512,96),'comparison512OrtMs':timing(ot512,96),
          'outputStats':stats(ort1024),'individualPcaBiasChecks':controls,
          'onnxBytes':path.stat().st_size,'onnxSha256':model_hash,'fixture':fixture,'passed':True}
    (OUT/'native-validation.json').write_text(json.dumps(meta,indent=2)+'\n')
    (OUT/'state-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    (OUT/'ATTRIBUTION.md').write_text('''# IIL Organ 1024 packet model attribution

Same actual pretrained Organ checkpoint and learned weights as [the512 export](../organ-512/ATTRIBUTION.md), with the same CC BY-NC4.0 terms, authors, pinned sources and checkpoint/config hashes. This is independent of Neutone Morpho.

Adaptation2026-10-26: batch two true512-stride latent frames into each1024-sample audio packet. Preserve all128 posterior dimensions,36 explicit states and six PCA delta controls. No retraining, quantization or new model weights. Every ONNX initializer is bitwise identical to the512 export. Only fixed packet shapes and trace constants differ.

Source: [IIL checkpoint c25a03d](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/tree/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000). License: [CC BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/). See the full attribution linked above for RAVE/IRCAM, Victor Shepardson forks and runtime licenses.

1024 samples is the packet size, not a new encoder compression ratio or a measured physical latency. No additional shift is needed against the native reference; absolute dry/wet and physical round-trip delay remain unmeasured.
''')
    c=write_contract(source_contract,meta,spec)
    assert original.sha(source_onnx)==source_onnx_sha
    print('DONE',json.dumps({k:meta[k] for k in ['nativeLongVsStreamError','native512Vs1024StreamError','onnxVsNativeStreamError','onnx512Vs1024StreamError','nativeStreamMs','ortStreamMs','onnxBytes','onnxSha256','passed']}),flush=True)

if __name__=='__main__':main()
