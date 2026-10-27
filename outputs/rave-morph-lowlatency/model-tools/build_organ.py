#!/usr/bin/env python3
"""Export the pinned, pretrained causal IIL organ checkpoint at its real 512 hop."""
from __future__ import annotations
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
WORK = BASE/'work/rave-lowlatency'
OUT = Path(__file__).resolve().parents[1]/'web/models/organ-512'
sys.path.insert(0,str(WORK/'cached_conv'))
sys.path.insert(0,str(WORK/'RAVE'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import rave
import cached_conv as cc
import onnx
import onnxruntime as ort

CHECKPOINT_SHA='218518cb0f7c2670a9cd3973c3e2972632893f599deacc851998b372a8f8a22e'
CONFIG_SHA='975e0e8615204af6c9f8aa901275ac634d70f60011c898dbc103e1e4fb09ccb0'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def err(a,b):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    return {'maxAbs':float(np.max(np.abs(d))),'rms':float(np.sqrt(np.mean(d*d)))}


def load_source():
    checkpoint=WORK/'organ-b512/last.ckpt';config=WORK/'organ-b512/config.gin'
    assert sha(checkpoint)==CHECKPOINT_SHA and sha(config)==CONFIG_SHA
    cc.use_cached_conv(False)
    gin.clear_config();gin.parse_config_file(str(config))
    m=rave.RAVE()
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state=data['state_dict'];result=m.load_state_dict(state,strict=False)
    assert result.missing_keys==['latent_valid_crop'] and not result.unexpected_keys,result
    m.eval()
    adain=[]
    for name,mod in m.named_modules():
        if isinstance(mod,rave.blocks.AdaptiveInstanceNormalization):
            flags={k:float(getattr(mod,k).item()) for k in ['learn_x','learn_y','num_update_x','num_update_y']}
            assert flags['learn_x']==0 and flags['learn_y']==0
            adain.append({'name':name,**flags})
    # Training-only discriminator/loss state is never part of inference.
    m.discriminator=nn.Identity();m.audio_distance=nn.Identity();m.multiband_audio_distance=nn.Identity()
    x=torch.zeros(1,1,4096)
    with torch.no_grad():
        mean,std=m.encoder.params(m.encode(x));before=m.decode(mean).clone()
    folded=0
    for mod in m.modules():
        if hasattr(mod,'weight_g') and hasattr(mod,'weight_v'):
            nn.utils.remove_weight_norm(mod);folded+=1
    with torch.no_grad():
        mean,std=m.encoder.params(m.encode(x));after=m.decode(mean)
    fold_error=err(before.numpy(),after.numpy());assert fold_error['maxAbs']==0
    meta={'checkpointSha256':CHECKPOINT_SHA,'configSha256':CONFIG_SHA,'trainingGlobalStep':data['global_step'],
        'sourceMissingNonInferenceBuffer':['latent_valid_crop'],'allLearnedStateLoaded':True,
        'checkpointReceptiveField':state['receptive_field'].tolist(),'adain':adain,
        'foldedWeightNormModules':folded,'weightNormFoldError':fold_error,
        'pcaShape':list(m.latent_pca.shape),'fidelityFirstSix':m.fidelity[:6].tolist(),
        'pcaOrthonormalMaxError':float((m.latent_pca@m.latent_pca.T-torch.eye(128)).abs().max())}
    del state,data
    return m,meta


class Controlled(nn.Module):
    def __init__(self,m):
        super().__init__();self.model=m
        self.register_buffer('pca6',m.latent_pca[:6].unsqueeze(-1).clone())
        self.register_buffer('pca6_inverse',m.latent_pca[:6].T.unsqueeze(-1).contiguous())
        self.register_buffer('pca_mean',m.latent_mean[None,:,None].clone())
    def forward(self,audio,encode_noise,latent_gain,latent_bias):
        mean,std=self.model.encoder.params(self.model.encode(audio))
        z=mean+std*encode_noise
        projected=F.conv1d(z-self.pca_mean,self.pca6)
        delta=projected*(latent_gain-1)+latent_bias
        z=z+F.conv1d(delta,self.pca6_inverse)
        return self.model.decode(z)


class Context:
    def __init__(self):self.spec=[];self.inputs=[];self.outputs=[]
    def register(self,name,channels,length,kind):
        index=len(self.spec)
        self.spec.append({'input':f'state_{index}','output':f'next_state_{index}',
                         'shape':[1,channels,length],'sourceModule':name,'kind':kind})
        return index


class StateConv(nn.Module):
    def __init__(self,conv,ctx,name):
        super().__init__();self.conv=conv;self.ctx=ctx
        self.length=conv._pad[0]
        assert conv._pad[1]==0 and conv.padding==(0,),name
        self.index=ctx.register(name,conv.in_channels,self.length,'causal_conv')
    def forward(self,x):
        x=torch.cat([self.ctx.inputs[self.index],x],-1)
        self.ctx.outputs[self.index]=x[:,:,-self.length:]
        c=self.conv
        return F.conv1d(x,c.weight,c.bias,c.stride,0,c.dilation,c.groups)


class StateTranspose(nn.Module):
    def __init__(self,conv,ctx,name):
        super().__init__();self.conv=conv;self.ctx=ctx
        self.length=conv.pad.padding[0]
        assert conv.pad.padding[1]==0 and conv.padding[0]==conv.kernel_size[0]-conv.stride[0]
        self.index=ctx.register(name,conv.in_channels,self.length,'trained_causal_transpose')
    def forward(self,x):
        x=torch.cat([self.ctx.inputs[self.index],x],-1)
        self.ctx.outputs[self.index]=x[:,:,-self.length:]
        c=self.conv
        return F.conv_transpose1d(x,c.weight,c.bias,c.stride,c.padding,c.output_padding,c.groups,c.dilation)


def replace(module,ctx,prefix=''):
    for name,child in list(module.named_children()):
        path=prefix+'.'+name if prefix else name
        if isinstance(child,cc.convs.CausalConvTranspose1d):
            setattr(module,name,StateTranspose(child,ctx,path))
        elif isinstance(child,cc.convs.Conv1d) and child._pad[0]:
            setattr(module,name,StateConv(child,ctx,path))
        else:replace(child,ctx,path)


class Stateful(nn.Module):
    def __init__(self,reference):
        super().__init__();self.reference=copy.deepcopy(reference);self.context=Context()
        replace(self.reference.model,self.context)
    def forward(self,audio,encode_noise,latent_gain,latent_bias,*states):
        self.context.inputs=states;self.context.outputs=[None]*len(self.context.spec)
        y=self.reference(audio,encode_noise,latent_gain,latent_bias)
        return (y,*self.context.outputs)
    def zeros(self):return [torch.zeros(x['shape']) for x in self.context.spec]


def infer_stream(model,audio,noise,knob=False):
    states=model.zeros();outputs=[];times=[]
    with torch.no_grad():
        for k in range(audio.shape[-1]//512):
            gain=torch.ones(1,6,1);bias=torch.zeros(1,6,1)
            if knob and k>=audio.shape[-1]//1024:bias[:,0,:]=.5;gain[:,1,:]=.5
            t=time.perf_counter()
            y,*states=model(audio[:,:,k*512:(k+1)*512],noise[:,:,k:k+1],gain,bias,*states)
            times.append((time.perf_counter()-t)*1000);outputs.append(y)
    return torch.cat(outputs,-1),states,times


def ort_session(path):
    o=ort.SessionOptions();o.intra_op_num_threads=o.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])


def write_contract(meta,specs):
    c={'id':'organ-causal-512','name':'Organ · RAVE causal 512 / PCA6','family':'organ','familyName':'Organ / IIL',
        'defaultOutputGain':.35,'controlDescription':'学習済みPCA先頭6方向のgain/bias。全128潜在座標を保持し、neutralでは元モデルと同じ処理。',
        'url':'organ-512/organ-causal-512.onnx','sampleRate':48000,'frameSize':512,
        'audioInput':'audio','audioOutput':'output','audioInputShape':[1,1,512],'audioOutputShape':[1,1,512],'dtype':'float32',
        'latentGainInput':'latent_gain','latentBiasInput':'latent_bias','latentShape':[1,6,1],
        'latentDimensions':6,'exposedDimensions':6,'encodedLatentDimensions':128,'retainedLatentDimensions':128,
        'latentCoordinateSystem':'First six checkpoint PCA directions; additive delta controls preserve all128 posterior coordinates at neutral',
        'neutralGain':1,'neutralBias':0,'noiseInputs':[{'name':'encode_noise','shape':[1,128,1],'distribution':'normal'}],
        'states':specs,'stateReset':'zeros','streaming':True,'overlapSamples':0,'warmupBlocks':128,'warmupFrames':128,
        'additionalDelaySamples':0,'dryOffsetSamples':0,'modelDelaySamples':None,'sourceModelDelaySamples':None,
        'latencyNote':'True encoder stride512. Native long-reference alignment requires no added shift; original absolute dry/wet/physical latency unmeasured. checkpoint receptive_field right470 is not a delay.',
        'blockDurationMs':512/48,'opset':17,'irVersion':8,'bytes':meta['onnxBytes'],'sha256':meta['onnxSha256'],
        'sourceCheckpointSha256':CHECKPOINT_SHA,
        'sourceURL':'https://huggingface.co/Intelligent-Instruments-Lab/rave-models/tree/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000',
        'author':'Intelligent Instruments Lab; RAVE by ACIDS / IRCAM; source/export fork by Victor Shepardson',
        'license':'CC BY-NC 4.0','licenseURL':'https://creativecommons.org/licenses/by-nc/4.0/',
        'adaptation':'Causal learned model, 36 explicit past states, posterior Gaussian epsilon input, six PCA delta gain/bias controls. No retraining; original learned state loaded and weight norm folded.',
        'validatedNative':True,'experimental':True,'parityFixture':'organ-512/parity-fixture.json'}
    (OUT/'contract.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    catpath=OUT.parent/'catalog.json'
    catalog=json.loads(catpath.read_text()) if catpath.exists() else {'version':1,'models':[]}
    catalog['models']=[c]+[x for x in catalog['models'] if x['id']!=c['id']]
    catalog['defaultModel']=c['id']
    catpath.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    return c


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(20261026)
    model,meta=load_source();reference=Controlled(model).eval();stream=Stateful(reference).eval()
    specs=stream.context.spec;zero=stream.zeros()
    x=torch.randn(1,1,512)*.05;noise=torch.zeros(1,128,1);gain=torch.ones(1,6,1);bias=torch.zeros(1,6,1)
    with torch.no_grad():
        direct=reference(x,noise,gain,bias);candidate=stream(x,noise,gain,bias,*zero)[0]
    print('FIRST_PACKET',err(direct.numpy(),candidate.numpy()),'states',len(specs),flush=True)
    # Six seconds; first second zero, followed by a locally synthesized note sequence.
    n=512*564;t=torch.arange(n)/48000
    frequencies=torch.tensor([220.,261.6256,329.6276,293.6648,196.,246.9417])
    note=(t.long()).clamp(max=5);freq=frequencies[note]
    envelope=(1-t%1).clamp(0,1)*torch.minimum((t%1)*30,torch.ones_like(t))
    audio=(.12*torch.sin(2*torch.pi*freq*t)+.04*torch.sin(4*torch.pi*freq*t))*envelope
    audio[:48000]=0;audio=audio[None,None];eps=torch.randn(1,128,n//512)
    with torch.no_grad():full=reference(audio,eps,gain,bias)
    emitted,states,times=infer_stream(stream,audio,eps)
    native_error=err(full.numpy(),emitted.numpy());print('FULL_NATIVE',native_error,flush=True)
    assert native_error['maxAbs']<1e-4 and native_error['rms']<1e-5,native_error
    path=OUT/'organ-causal-512.onnx'
    torch.onnx.export(stream,(x,noise,gain,bias,*zero),str(path),opset_version=17,
        input_names=['audio','encode_noise','latent_gain','latent_bias']+[s['input'] for s in specs],
        output_names=['output']+[s['output'] for s in specs],do_constant_folding=True)
    exported=onnx.load(path);onnx.checker.check_model(exported)
    session=ort_session(path);ortstates={s['input']:np.zeros(s['shape'],np.float32) for s in specs}
    out=[];orttimes=[]
    for k in range(n//512):
        feeds={**ortstates,'audio':audio[:,:,k*512:(k+1)*512].numpy(),'encode_noise':eps[:,:,k:k+1].numpy(),
               'latent_gain':gain.numpy(),'latent_bias':bias.numpy()}
        started=time.perf_counter();r=dict(zip([x.name for x in session.get_outputs()],session.run(None,feeds)))
        orttimes.append((time.perf_counter()-started)*1000);out.append(r['output'])
        ortstates={s['input']:r[s['output']] for s in specs}
    ortout=np.concatenate(out,-1);ort_error=err(emitted.numpy(),ortout)
    print('ORT_ERROR',ort_error,flush=True)
    assert ort_error['maxAbs']<.001 and ort_error['rms']<.0001,ort_error
    meta.update({'environment':{'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__,
                'platform':platform.platform(),'threads':1},'sampleRate':48000,'frameSize':512,'encodedLatentShape':[1,128,1],
        'controlsShape':[1,6,1],'stateCount':len(specs),'nativeLongVsStreamError':native_error,
        'onnxVsNativeStreamError':ort_error,'additionalStreamingShiftSamples':0,'sourceAbsoluteAlignmentSamples':None,
        'nativeStreamMs':{'median':float(np.median(times[96:])),'p95':float(np.percentile(times[96:],95))},
        'ortStreamMs':{'median':float(np.median(orttimes[96:])),'p95':float(np.percentile(orttimes[96:],95))},
        'onnxBytes':path.stat().st_size,'onnxSha256':sha(path),'passed':True})
    (OUT/'native-validation.json').write_text(json.dumps(meta,indent=2)+'\n')
    (OUT/'state-spec.json').write_text(json.dumps(specs,indent=2)+'\n')
    write_contract(meta,specs)
    np.savez(WORK/'organ-b512/demo-arrays.npz',input=audio.numpy(),noise=eps.numpy(),neutral=ortout)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
