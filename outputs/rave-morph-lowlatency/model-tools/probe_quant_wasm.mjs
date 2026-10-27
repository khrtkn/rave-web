// Same official ORT Web WASM build as the app, under Node; no browser/UI automation.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const scriptDir=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(scriptDir,'../../..');
const here=path.join(root,'work/rave-lowlatency/quant-candidates');
const vendor=path.join(root,'outputs/rave-morph-lowlatency/web/vendor/ort');
const ort=await import(pathToFileURL(path.join(vendor,'ort.wasm.bundle.min.mjs')).href);
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths={mjs:pathToFileURL(path.join(vendor,'ort-wasm-simd-threaded.mjs')).href,wasm:pathToFileURL(path.join(vendor,'ort-wasm-simd-threaded.wasm')).href};
const report={environment:'Node.js with actual bundled ORT Web WASM; not a browser measurement',node:process.version,wasmThreads:1,toys:[],models:[]};
for(const kind of ['u8','s8']){
  try{
    const s=await ort.InferenceSession.create(fs.readFileSync(path.join(here,`toy-${kind}.onnx`)),{executionProviders:['wasm']});
    const data=kind==='u8'?Uint8Array.from([1,2,3,4]):Int8Array.from([1,2,3,4]);
    const r=await s.run({x:new ort.Tensor(kind==='u8'?'uint8':'int8',data,[1,1,4])});
    report.toys.push({kind,output:Array.from(r.y.data),passed:Array.from(r.y.data).join(',')==='1,2,3,4'});await s.release();
  }catch(e){report.toys.push({kind,error:String(e),passed:false});}
}
const contract=JSON.parse(fs.readFileSync(path.join(root,'outputs/rave-morph-lowlatency/web/models/organ-512/contract.json')));
const models=[['f32',path.join(root,'outputs/rave-morph-lowlatency/web/models/organ-512/organ-causal-512.onnx')],['qdq-s8s8',path.join(here,'organ-512-qdq-s8s8.onnx')],['qop-u8s8',path.join(here,'organ-512-qop-u8s8.onnx')]];
let baseline;
for(const [name,file] of models){
  if(!fs.existsSync(file))continue;
  try{
    const s=await ort.InferenceSession.create(fs.readFileSync(file),{executionProviders:['wasm']});
    let states=Object.fromEntries(contract.states.map(v=>[v.input,new ort.Tensor('float32',new Float32Array(v.shape.reduce((a,b)=>a*b,1)),v.shape)]));
    const gain=new ort.Tensor('float32',new Float32Array(6).fill(1),[1,6,1]);const bias=new ort.Tensor('float32',new Float32Array(6),[1,6,1]);
    const output=[],times=[];let seed=20261027;
    const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return(seed+.5)/4294967296;};
    for(let k=0;k<192;k++){
      const input=Float32Array.from({length:512},(_,j)=>.1*Math.sin(2*Math.PI*(220+Math.floor(k/32)*55)*(k*512+j)/48000));
      const noise=Float32Array.from({length:128},()=>Math.sqrt(-2*Math.log(rand()))*Math.cos(2*Math.PI*rand()));
      const f={...states,audio:new ort.Tensor('float32',input,[1,1,512]),encode_noise:new ort.Tensor('float32',noise,[1,128,1]),latent_gain:gain,latent_bias:bias};
      const t=performance.now();const r=await s.run(f);times.push(performance.now()-t);output.push(...r.output.data);
      states=Object.fromEntries(contract.states.map(v=>[v.input,r[v.output]]));
    }
    const sorted=times.slice(32).sort((a,b)=>a-b);let err=0,power=0,maxAbs=0,peak=0;
    if(name==='f32')baseline=output;
    for(let i=0;i<output.length;i++){const d=output[i]-baseline[i];err+=d*d;power+=baseline[i]*baseline[i];maxAbs=Math.max(maxAbs,Math.abs(d));peak=Math.max(peak,Math.abs(output[i]));}
    report.models.push({name,passed:true,medianMs:sorted[Math.floor(sorted.length*.5)],p95Ms:sorted[Math.floor(sorted.length*.95)],relativeRms:Math.sqrt(err/power),errorMaxAbs:maxAbs,peak});
    await s.release();
  }catch(e){report.models.push({name,passed:false,error:String(e)});}
}
fs.writeFileSync(path.join(here,'wasm-results.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
