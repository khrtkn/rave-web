#!/usr/bin/env python3
"""Separate experimental stateful specialization of the actual Scyclone RAVE graph."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from build_water_latent import errors, session, states_for, step

SHA = "ea94557a8d9b40f22a1dadbc93ab98f46144a68ce0bed1403a1558a2d1007f16"
EXTRA_DELAY = 1360


def value(name, dims):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)


def explicit_noise(data, frame=None, controls=False):
    m = onnx.load_from_string(data)
    if frame:
        m.graph.input[0].type.tensor_type.shape.dim[2].dim_value = frame
    length = frame//2048 if frame else "latent_time"
    for i, n in enumerate(m.graph.node):
        if n.op_type == "RandomNormalLike":
            name = "encode_noise" if n.name == "/RandomNormalLike" else "decode_noise"
            channels = 16 if name == "encode_noise" else 12
            m.graph.input.append(value(name, [1, channels, length]))
            m.graph.node[i].CopyFrom(helper.make_node("Identity", [name], list(n.output), name=n.name+"/ExplicitNoise"))
    if controls:
        edge = "/Slice_2_output_0"
        for n in m.graph.node:
            for i, inp in enumerate(n.input):
                if inp == edge:
                    n.input[i] = "latent_controlled"
        nodes = list(m.graph.node)
        j = next(i for i,n in enumerate(nodes) if edge in n.output)+1
        nodes[j:j] = [helper.make_node("Mul", [edge,"latent_gain"],["latent_scaled"],name="Candidate/LatentGain"),
                      helper.make_node("Add",["latent_scaled","latent_bias"],["latent_controlled"],name="Candidate/LatentBias")]
        del m.graph.node[:];m.graph.node.extend(nodes)
        m.graph.input.extend([value("latent_gain",[1,4,1]),value("latent_bias",[1,4,1])])
    return m


def prune(m):
    needed = {o.name for o in m.graph.output}
    keep = []
    for n in reversed(list(m.graph.node)):
        if any(x in needed for x in n.output):
            keep.append(n);needed.update(n.input)
    del m.graph.node[:];m.graph.node.extend(reversed(keep))
    del m.graph.value_info[:]
    return m


def streaming(data, frame=2048):
    m = explicit_noise(data, frame, True)
    initializers = {x.name:numpy_helper.to_array(x) for x in m.graph.initializer}
    consumers = {name:n for n in m.graph.node for name in n.input}
    nodes=[];states=[];counter=0
    def constant(arr):
        nonlocal counter
        name=f"Candidate/Constant_{counter}";counter+=1
        m.graph.initializer.append(numpy_helper.from_array(np.asarray(arr,dtype=np.int64),name))
        return name
    axis=constant([2]);one=constant([1]);end=constant([np.iinfo(np.int64).max])
    def history(activation,channels,length):
        name=f"state_{len(states)}";out="next_"+name;cat="Candidate/Concat_"+name
        states.append({"input":name,"output":out,"shape":[1,channels,length]})
        m.graph.input.append(value(name,[1,channels,length]));m.graph.output.append(value(out,[1,channels,length]))
        result=[helper.make_node("Concat",[name,activation],[cat],name="Candidate/History/"+name,axis=2),
                helper.make_node("Slice",[cat,constant([-length]),end,axis,one],[out],name="Candidate/Next/"+name)]
        return cat,result
    for n in list(m.graph.node):
        if n.op_type == "Pad":
            conv = consumers.get(n.output[0])
            assert conv and conv.op_type=="Conv",n.name
            a={x.name:helper.get_attribute_value(x) for x in conv.attribute}
            length=(a['kernel_shape'][0]-1)*a.get('dilations',[1])[0]
            w=initializers[conv.input[1]];channels=w.shape[1]*a.get('group',1)
            cat,extra=history(n.input[0],channels,length)
            # Keep original Pad output name as the state-plus-current activation.
            extra[0].output[0]=n.output[0];extra[1].input[0]=n.output[0]
            nodes.extend(extra)
        elif n.op_type == "ConvTranspose":
            a={x.name:helper.get_attribute_value(x) for x in n.attribute}
            stride=a['strides'][0];pad=a['pads'][0]
            assert a['kernel_shape']==[2*stride] and a['pads']==[pad,pad]
            w=initializers[n.input[1]]
            cat,extra=history(n.input[0],w.shape[0],2)
            nodes.extend(extra)
            changed=copy.deepcopy(n);changed.input[0]=cat
            original_output=n.output[0];changed.output[0]=original_output+"/with_history"
            nodes.append(changed)
            nodes.append(helper.make_node("Slice",[changed.output[0],constant([2*stride-pad]),constant([-pad]),axis,one],
                                           [original_output],name=n.name+"/CausalShift"))
        elif n.name == "/Mul_6":
            # The decoder's extra delay is 85 subband samples: odd modulation phase.
            sign=np.ones((1,16,1),np.float32);sign[:,1::2,:]=-1
            m.graph.initializer.append(numpy_helper.from_array(sign,"Candidate/SynthesisPhase"))
            nodes.append(helper.make_node("Mul",[n.input[0],"Candidate/SynthesisPhase"],["Candidate/PhaseAdjusted"],name="Candidate/PhaseAdjust"))
            changed=copy.deepcopy(n);changed.input[0]="Candidate/PhaseAdjusted";nodes.append(changed)
        else:nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes)
    m=prune(m)
    # Pin the audio output; original anonymous dynamic dimensions are not contracts.
    m.graph.output[0].CopyFrom(value("audio_out",[1,1,frame]))
    m=onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    return m,states


def run_stream(s,audio,knobs=None,noise=None):
    st=states_for(s);out=[];timings=[]
    for start in range(0,audio.shape[-1],2048):
        feeds={**st,"audio_in":audio[:,:,start:start+2048],"encode_noise":np.zeros((1,16,1),np.float32),"decode_noise":np.zeros((1,12,1),np.float32),
               "latent_gain":np.ones((1,4,1),np.float32),"latent_bias":np.zeros((1,4,1),np.float32)}
        if knobs:
            if start>=2048*48:feeds[knobs[0]][0,knobs[1],0]=knobs[2]
        if noise:
            j=start//2048
            for name,arr in noise.items():feeds[name]=arr[:,:,j:j+1]
        t=time.perf_counter();r=dict(zip([o.name for o in s.get_outputs()],s.run(None,feeds)));timings.append((time.perf_counter()-t)*1000)
        out.append(r['audio_out']);st={k:r['next_'+k] for k in st}
    return np.concatenate(out,axis=2),timings,st


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True,help='Original Scyclone djembe.onnx; the pinned SHA-256 is verified')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'web/models/djembe-candidate')
    a=p.parse_args();data=a.source.read_bytes();assert hashlib.sha256(data).hexdigest()==SHA
    model,states=streaming(data);encoded=model.SerializeToString()
    reference=session(explicit_noise(data).SerializeToString());s=session(encoded)
    rng=np.random.default_rng(20261026);length=2048*128
    t=np.arange(length)/48000
    x=(0.12*np.sin(2*np.pi*(170+32*t)*t)+0.03*rng.standard_normal(length)).astype(np.float32)
    x[:2048*32]=0
    audio=x[None,None]
    ref=reference.run(None,{'audio_in':audio,'encode_noise':np.zeros((1,16,length//2048),np.float32),'decode_noise':np.zeros((1,12,length//2048),np.float32)})[0]
    y,timings,final_state=run_stream(s,audio)
    warmup=2048*48
    e=errors(ref[:,:,warmup:-EXTRA_DELAY],y[:,:,warmup+EXTRA_DELAY:])
    print('ALIGNED_ERROR',e,flush=True)
    a.out.mkdir(parents=True,exist_ok=True)
    report={'sourceSha256':SHA,'totalEncoderStride':2048,'frameSize':2048,'additionalShiftSamples':EXTRA_DELAY,
            'noise':'Both explicit noise inputs zero','warmupExcludedSamples':warmup,'comparedSamples':length-warmup-EXTRA_DELAY,
            'alignedError':e,'medianNativeMs':float(np.median(timings[32:])),'p95NativeMs':float(np.percentile(timings[32:],95)),
            'stateCount':len(states),'retainedLatentDimensions':4,'passed':e['max_abs']<2e-4 and e['rms']<2e-5}
    report['referenceOutputRmsAfterWarmup']=float(np.sqrt(np.mean(ref[:,:,warmup:].astype(np.float64)**2)))
    report['streamOutputPeakAfterWarmup']=float(np.max(np.abs(y[:,:,warmup:])))
    (a.out/'candidate-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise RuntimeError('Candidate rejected: reference alignment failed; no model is published')
    random_noise={'encode_noise':rng.standard_normal((1,16,length//2048)).astype(np.float32),
                  'decode_noise':rng.standard_normal((1,12,length//2048)).astype(np.float32)}
    ref_random=reference.run(None,{'audio_in':audio,**random_noise})[0]
    y_random,_,_=run_stream(s,audio,noise=random_noise)
    random_error=errors(ref_random[:,:,warmup:-EXTRA_DELAY],y_random[:,:,warmup+EXTRA_DELAY:])
    report['alignedSameExplicitRandomNoiseError']=random_error
    report['randomNoiseParityNote']='Measured float32 long-signal versus short-block difference; exceeds the zero-noise strict threshold. Retained as experimental evidence, not reported as exact parity.'
    report['randomNoiseReferenceRms']=float(np.sqrt(np.mean(ref_random[:,:,warmup:-EXTRA_DELAY].astype(np.float64)**2)))
    axis_tests=[]
    for dim in range(4):
        axis={'dimension':dim+1}
        for kind in ['gain','bias']:
            changed,_,_=run_stream(s,audio,('latent_'+kind,dim,0.5),noise=random_noise)
            diff=errors(y_random[:,:,warmup+EXTRA_DELAY:],changed[:,:,warmup+EXTRA_DELAY:])
            assert diff['rms']>1e-6 and np.isfinite(changed).all(),(dim,kind,diff)
            axis[kind]={'value':0.5,'difference':diff,'finite':True}
        axis_tests.append(axis)
    report['independentFourAxisTests']=axis_tests
    report['independentAxisNoise']='All comparisons use identical explicit normal-noise tensors and identical warm state; exactly one gain or bias changes after frame 48.'
    repeated,_,repeated_state=run_stream(s,audio)
    assert np.array_equal(y,repeated) and all(np.array_equal(v,repeated_state[k]) for k,v in final_state.items())
    report['resetOutputAndAllStateTensorsBitwiseIdentical']=True
    orig_inits={x.name:x.SerializeToString() for x in onnx.load_from_string(data).graph.initializer}
    adapted_inits={x.name:x.SerializeToString() for x in model.graph.initializer}
    assert all(adapted_inits[k]==v for k,v in orig_inits.items())
    report['allOriginalInitializersByteIdentical']=True
    report['originalInitializerCount']=len(orig_inits)
    report['projectionInverseTransposeMaxError']=float(np.max(np.abs(numpy_helper.to_array(next(x for x in model.graph.initializer if x.name=='onnx::Conv_1259'))[:,:,0].T-
        numpy_helper.to_array(next(x for x in model.graph.initializer if x.name=='onnx::Conv_1262'))[:,:,0])))
    (a.out/'djembe-stateful-2048.onnx').write_bytes(encoded)
    (a.out/'candidate-states.json').write_text(json.dumps(states,indent=2)+'\n')
    contract={'id':'djembe-stateful-2048','name':'Djembe · RAVE 4 latent','url':'djembe-stateful-2048.onnx',
        'sampleRate':48000,'frameSize':2048,'audioInput':'audio_in','audioOutput':'audio_out',
        'audioInputShape':[1,1,2048],'audioOutputShape':[1,1,2048],'dtype':'float32',
        'latentGainInput':'latent_gain','latentBiasInput':'latent_bias','latentShape':[1,4,1],
        'latentDimensions':4,'exposedDimensions':4,'latentCoordinateSystem':'Four retained projected latent coordinates; statistical PCA provenance not established by graph alone',
        'neutralGain':1,'neutralBias':0,'noiseInputs':[{'name':'encode_noise','shape':[1,16,1],'distribution':'normal'},
            {'name':'decode_noise','shape':[1,12,1],'distribution':'normal'}],
        'states':states,'stateReset':'zeros','streaming':True,'overlapSamples':0,'warmupFrames':64,'warmupBlocks':64,
        'additionalDelaySamples':EXTRA_DELAY,'dryOffsetSamples':EXTRA_DELAY,'modelDelaySamples':EXTRA_DELAY,'sourceModelDelaySamples':None,
        'latencyNote':'1360 samples is the validated added causalization delay relative to this original ONNX; original timbre model alignment delay is unknown. Dry compensation of 1360 addresses only the known added delay.',
        'blockDurationMs':2048/48,'opset':12,'irVersion':7,'bytes':len(encoded),'sha256':hashlib.sha256(encoded).hexdigest(),
        'sourceSha256':SHA,'sourceURL':'https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/assets/models/djembe.onnx',
        'author':'Torsion Audio; RAVE by Antoine Caillon and Philippe Esling / ACIDS IRCAM',
        'license':'CC BY-NC 4.0','licenseURL':'https://creativecommons.org/licenses/by-nc/4.0/',
        'adaptation':'35 past Conv caches, four ConvTranspose cache/shift stages and synthesis modulation phase adjustment, explicit noise inputs, four latent gain/bias controls; all original initializers unchanged',
        'validatedNative':True,'experimental':True,'parityFixture':'djembe-candidate/parity-fixture.json'}
    (a.out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (a.out/'catalog.json').write_text(json.dumps({'version':1,'defaultModel':contract['id'],'models':[contract]},indent=2)+'\n')
    fixture_audio=audio[:,:,:16384].copy()
    fixture_audio[0,0,:]=(0.12*np.sin(2*np.pi*220*np.arange(16384)/48000)+0.03*np.sin(2*np.pi*733*np.arange(16384)/48000)).astype(np.float32)
    fixture_audio[0,0,5000:5400]+=np.float32(0.15)*np.exp(-np.arange(400)/90).astype(np.float32)
    expected,_,_=run_stream(s,fixture_audio)
    fixture={
        'version':1,'model':contract['id'],'modelSha256':contract['sha256'],'sampleRate':48000,'frameSize':2048,
        'frames':8,'channels':1,'samples':16384,'format':'IEEE754 little-endian float32 contiguous mono, no header',
        'latentGain':1,'latentBias':0,'allNoiseInputs':0,'initialStates':'all zero','warmupFrames':0,
        'outputAlignment':'raw stream output; includes causalization delay; no output shift/crop/limiter/fade',
        'tolerance':{'maxAbs':0.00002,'rms':0.000002},'nativeOutputPeak':float(np.max(np.abs(expected)))}
    for key,filename,array in [('input','parity-input.f32',fixture_audio),('expectedOutput','parity-expected.f32',expected)]:
        raw=np.asarray(array,dtype='<f4').tobytes();(a.out/filename).write_bytes(raw)
        fixture[key]={'file':filename,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    (a.out/'parity-fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    (a.out/'candidate-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
