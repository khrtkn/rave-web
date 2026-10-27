"""Experimental static Conv quantization; preserves source file and recurrent contract."""
from pathlib import Path
import collections, hashlib, json, time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'outputs/rave-morph-lowlatency/web/models/organ-512'
OUT=ROOT/'work/rave-lowlatency/quant-candidates'
SOURCE_SHA='01099b4120d451f66300c91b4c02030622688bb75b9c20e6ca0e616348c7d342'
MODEL=SOURCE/'organ-causal-512.onnx'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def session(p,profile=False):
    o=ort.SessionOptions();o.intra_op_num_threads=o.inter_op_num_threads=1;o.enable_profiling=profile
    o.log_severity_level=3
    if profile:o.profile_file_prefix=str(OUT/'ort-profile')
    return ort.InferenceSession(str(p),o,providers=['CPUExecutionProvider'])
CONTRACT=json.loads((SOURCE/'contract.json').read_text());SPECS=CONTRACT['states']
def states():return {s['input']:np.zeros(s['shape'],np.float32) for s in SPECS}
def feeds(audio,eps,gain=None,bias=None):
    return {'audio':np.ascontiguousarray(audio,np.float32).reshape(1,1,512),
            'encode_noise':np.ascontiguousarray(eps,np.float32).reshape(1,128,1),
            'latent_gain':np.ones((1,6,1),np.float32) if gain is None else gain,
            'latent_bias':np.zeros((1,6,1),np.float32) if bias is None else bias}
def stimuli(blocks,seed):
    rng=np.random.default_rng(seed);t=np.arange(blocks*512,dtype=np.float64)/48000
    segment=(t//.4).astype(int);frequency=80*2.**((segment%9)/2)
    amp=np.array([0.,.003,.02,.1,.3,.03,.2,.08,.4])[segment%9]
    x=amp*(.7*np.sin(2*np.pi*frequency*t)+.2*np.sin(2*np.pi*frequency*2.013*t)+.1*rng.normal(size=t.size))
    return x.astype(np.float32).reshape(blocks,512),rng.normal(size=(blocks,128)).astype(np.float32)
def collect(blocks=384):
    x,eps=stimuli(blocks,20261026);s=session(MODEL);state=states();outnames=[z.name for z in s.get_outputs()];rows=[]
    for i in range(blocks):
        gain=np.ones((1,6,1),np.float32);bias=np.zeros_like(gain)
        if i>=blocks//2:gain[0,i%6,0]=.7+((i//48)%3)*.3;bias[0,(i+2)%6,0]=.25*np.sin(i*.1)
        f={**state,**feeds(x[i],eps[i],gain,bias)};rows.append(f)
        r=dict(zip(outnames,s.run(None,f)));state={v['input']:r[v['output']] for v in SPECS}
    return rows
class Reader(CalibrationDataReader):
    def __init__(self,rows):self.rows=rows;self.rewind()
    def get_next(self):return next(self.it,None)
    def rewind(self):self.it=iter(self.rows)
def evaluate(path,blocks=384):
    x,eps=stimuli(blocks,20261027);s=session(path);state=states();outs=[];times=[]
    names=[v.name for v in s.get_outputs()]
    for i in range(blocks):
        gain=np.ones((1,6,1),np.float32);bias=np.zeros_like(gain)
        if i>=blocks//2:gain[0,0,0]=.7;bias[0,1,0]=.35
        f={**state,**feeds(x[i],eps[i],gain,bias)};t=time.perf_counter();r=dict(zip(names,s.run(None,f)));times.append((time.perf_counter()-t)*1000)
        state={v['input']:r[v['output']] for v in SPECS};outs.append(r['output'])
    out=np.concatenate(outs,-1).reshape(-1)
    return out,{'medianMs':float(np.median(times[64:])), 'p95Ms':float(np.percentile(times[64:],95)),
                'rms':float(np.sqrt(np.mean(out.astype(float)**2))),'peak':float(np.max(np.abs(out))),
                'finite':bool(np.isfinite(out).all())}
def make_toys():
    from onnx import helper as h,TensorProto as T,numpy_helper as nh
    for dtype,npdtype,name in [(T.UINT8,np.uint8,'u8'),(T.INT8,np.int8,'s8')]:
        init=[nh.from_array(np.array(1.,np.float32),name='scale'),nh.from_array(np.array(0,npdtype),name='zero'),nh.from_array(np.array([[[1]]],np.int8),name='weight'),nh.from_array(np.array(0,np.int8),name='wzero')]
        node=h.make_node('QLinearConv',['x','scale','zero','weight','scale','wzero','scale','zero'],['y'])
        m=h.make_model(h.make_graph([node],f'qlinearconv-{name}',[h.make_tensor_value_info('x',dtype,[1,1,4])],[h.make_tensor_value_info('y',dtype,[1,1,4])],init),opset_imports=[h.make_opsetid('',17)])
        m.ir_version=8;onnx.save(m,OUT/f'toy-{name}.onnx')
def main():
    assert sha(MODEL)==SOURCE_SHA;OUT.mkdir(exist_ok=True)
    make_toys()
    m=onnx.load(MODEL);weight={v.name:v for v in m.graph.initializer}
    selected=[];inventory=[]
    for n in m.graph.node:
        if n.op_type not in ('Conv','ConvTranspose'):continue
        w=weight[n.input[1]];size=int(np.prod(w.dims))*4
        choose=n.op_type=='Conv' and '/encoder/' in n.name or n.op_type=='Conv' and '/decoder/' in n.name
        choose=choose and not any(k in n.name for k in ['/encoder/encoder/net/net.0/','/encoder/encoder/net/net.32/','/decoder/net/net.32/'])
        if choose:selected.append(n.name)
        inventory.append({'name':n.name,'op':n.op_type,'weightShape':list(w.dims),'weightBytesF32':size,'selected':choose})
    report={'source':str(MODEL),'sourceSha256':sha(MODEL),'ortVersion':ort.__version__,'onnxVersion':onnx.__version__,
            'nodes':dict(collections.Counter(n.op_type for n in m.graph.node)), 'weightedNodeInventory':inventory,
            'selectedConvCount':len(selected),'calibration':'384 consecutive source-F32-state frames with seeded multi-tone/noise, amplitudes 0..0.4 and 6-axis controls; synthetic only',
            'validation':'Different seed; each candidate recursively carries its own 36 FP32 states; same source input/noise/control; no re-training',
            'nativeThreads':1,'candidates':[]}
    rows=collect();baseline,bstat=evaluate(MODEL);report['baseline']=bstat
    np.save(OUT/'baseline-validation.npy',baseline)
    for ident,fmt,act in [('qdq-s8s8',QuantFormat.QDQ,QuantType.QInt8),('qop-u8s8',QuantFormat.QOperator,QuantType.QUInt8)]:
        path=OUT/f'organ-512-{ident}.onnx'
        quantize_static(MODEL,path,Reader(rows),quant_format=fmt,activation_type=act,weight_type=QuantType.QInt8,
                       per_channel=True,reduce_range=False,nodes_to_quantize=selected,op_types_to_quantize=['Conv'],
                       extra_options={'WeightSymmetric':True,'ActivationSymmetric':act==QuantType.QInt8})
        qm=onnx.load(path);onnx.checker.check_model(qm);output,stat=evaluate(path);delta=output.astype(float)-baseline
        signal=float(np.sum(baseline.astype(float)**2));noise=float(np.sum(delta**2));stat.update({
            'name':ident,'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'ops':dict(collections.Counter(n.op_type for n in qm.graph.node)),
            'errorMaxAbs':float(np.max(np.abs(delta))),'errorRms':float(np.sqrt(np.mean(delta**2))),
            'relativeRms':float(np.sqrt(noise/max(signal,1e-30))),'snrDb':float(10*np.log10(max(signal,1e-30)/max(noise,1e-30))),
            'speedupMedian':bstat['medianMs']/stat['medianMs'], 'qualityAccepted':False,
            'note':'A numeric candidate only; listening and broad calibration/validation are required. Not bitwise source parity.'})
        np.save(OUT/f'{ident}-validation.npy',output);report['candidates'].append(stat)
        (OUT/'quantization-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(stat),flush=True)
    print('DONE',flush=True)
if __name__=='__main__':main()
