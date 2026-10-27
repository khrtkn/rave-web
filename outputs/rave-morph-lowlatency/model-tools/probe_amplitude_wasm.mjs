// Actual app ORT Web WASM + unchanged Organ F32 model; no native ORT bindings.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../..');
const args=process.argv.slice(2),get=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const threads=Number(get('--threads','4')),probe=args.includes('--probe');
const measuredBlocks=Number(get('--blocks','192')),warmBlocks=Number(get('--warmup','128')),repeats=Number(get('--repeats','2'));
const outputFile=path.resolve(root,get('--output','outputs/rave-morph-lowlatency/research/amplitude-wasm-results.json'));
const vendor=path.join(root,'outputs/rave-morph-lowlatency/web/vendor/ort');
const modelDir=path.join(root,'outputs/rave-morph-lowlatency/web/models/organ-512');
const contract=JSON.parse(fs.readFileSync(path.join(modelDir,'contract.json')));
const model=fs.readFileSync(path.join(modelDir,'organ-causal-512.onnx'));
// The browser bundle compares origins before importing its pthread factory.
// file: has the standard opaque origin "null"; Node has no location global.
globalThis.location={origin:'null'};
// data: import of unmodified bundle bytes avoids its browser-specific file:
// base-URL resolution. The separate original pthread .mjs retains its file URL.
const ort=await import('data:text/javascript;base64,'+fs.readFileSync(path.join(vendor,'ort.wasm.bundle.min.mjs')).toString('base64'));
ort.env.wasm.numThreads=threads;ort.env.wasm.proxy=false;
ort.env.wasm.wasmBinary=new Uint8Array(fs.readFileSync(path.join(vendor,'ort-wasm-simd-threaded.wasm')));
// Bundle embeds the same Emscripten factory. Supplying wasmBinary avoids Node's
// unsupported fetch(file:), while preserving the app's exact WASM bytes.
ort.env.wasm.wasmPaths={mjs:pathToFileURL(path.join(vendor,'ort-wasm-simd-threaded.mjs')).href};
const report={createdAt:new Date().toISOString(),environment:'Node.js using exact app ORT Web WASM; no AudioWorklet, browser, microphone, or OS audio scheduling',node:process.version,architecture:process.arch,platform:process.platform,availableParallelism:os.availableParallelism(),requestedWasmThreads:threads,runtimeVersion:'1.30.0',modelSha256:crypto.createHash('sha256').update(model).digest('hex'),wasmSha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(vendor,'ort-wasm-simd-threaded.wasm'))).digest('hex'),nodeAdapter:'Unmodified browser bundle bytes imported from a data URL; original pthread mjs imported from its file URL; exact WASM bytes supplied via wasmBinary. No native ORT bindings.',timingScope:'session.run + finite output copy + finite next-state checks, matching app retain mode; excludes input/noise generation, tensor preparation, disposal and subnormal scan',stateMode:'retain',measuredBlocks,warmBlocks,repeats,float32MinNormal:2**-126,noiseModes:['random','zeros'],results:[]};
const session=await ort.InferenceSession.create(model,{executionProviders:['wasm'],executionMode:'sequential',graphOptimizationLevel:'all'});
report.configuredWasmThreads=ort.env.wasm.numThreads;
const size=shape=>shape.reduce((a,b)=>a*b,1),dispose=obj=>{for(const v of Object.values(obj))v.dispose?.();};
const gain=new ort.Tensor('float32',new Float32Array(size(contract.latentShape)).fill(1),contract.latentShape);
const bias=new ort.Tensor('float32',new Float32Array(size(contract.latentShape)),contract.latentShape);
function makePrng(seed=1){let spare=null;const uniform=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return((seed>>>0)+.5)/4294967296;};return{uniform,normal:()=>{if(spare!==null){const n=spare;spare=null;return n;}const r=Math.sqrt(-2*Math.log(uniform())),t=2*Math.PI*uniform();spare=r*Math.sin(t);return r*Math.cos(t);}};}
function stats(values){const s=[...values].sort((a,b)=>a-b);return{count:s.length,meanMs:values.reduce((a,b)=>a+b,0)/values.length,p50Ms:s[Math.floor(s.length*.5)],p95Ms:s[Math.ceil(s.length*.95)-1],p99Ms:s[Math.ceil(s.length*.99)-1],maxMs:s.at(-1),overBudgetBlocks:values.filter(n=>n>512/48).length};}
function magnitude(data){let subnormal=0,zero=0,nonfinite=0,minNonzero=Infinity,peak=0,power=0;for(const v of data){const a=Math.abs(v);if(!Number.isFinite(v))nonfinite++;else{if(a===0)zero++;else{minNonzero=Math.min(minNonzero,a);if(a<2**-126)subnormal++;}peak=Math.max(peak,a);power+=v*v;}}return{elements:data.length,subnormal,zero,nonfinite,minNonzero:minNonzero===Infinity?null:minNonzero,peak,rms:Math.sqrt(power/data.length)};}
const {demoSamples}=await import(pathToFileURL(path.join(root,'outputs/rave-morph-lowlatency/web/audio-utils.mjs')).href);
const demo=demoSamples(48000,8);let peak=0;for(const v of demo)peak=Math.max(peak,Math.abs(v));
const conditions=[['demo_peak_0.05',.05],['zero',0],['white_1e-3',1e-3],['white_1e-6',1e-6],['white_1e-12',1e-12]];
async function run(kind,amplitude,noiseMode,repeat,blocks,warmup){
  const startedAt=new Date().toISOString();
  let states=Object.fromEntries(contract.states.map(s=>[s.input,new ort.Tensor('float32',new Float32Array(size(s.shape)),s.shape)]));
  const audioRandom=makePrng(20261027),posteriorRandom=makePrng(1),times=[],runTimes=[];let lastOutput;
  for(let k=0;k<warmup+blocks;k++){
    const audio=new Float32Array(512);for(let j=0;j<audio.length;j++)audio[j]=kind.startsWith('demo')?demo[(k*512+j)%demo.length]*amplitude/peak:kind==='zero'?0:(audioRandom.uniform()*2-1)*amplitude;
    const noise=new Float32Array(128);if(noiseMode==='random')for(let j=0;j<128;j++)noise[j]=posteriorRandom.normal();
    const temporary={audio:new ort.Tensor('float32',audio,contract.audioInputShape),encode_noise:new ort.Tensor('float32',noise,[1,128,1])};
    const start=performance.now(),outputs=await session.run({...states,...temporary,latent_gain:gain,latent_bias:bias}),runEnd=performance.now();
    const raw=outputs.output.data;if(!raw.every(Number.isFinite))throw Error('Non-finite audio');lastOutput=new Float32Array(raw);
    const next={};for(const s of contract.states){const value=outputs[s.output];if(!value.data.every(Number.isFinite))throw Error('Non-finite state');next[s.input]=value;delete outputs[s.output];}
    const end=performance.now();if(k>=warmup){times.push(end-start);runTimes.push(runEnd-start);}
    dispose(states);states=next;dispose(temporary);dispose(outputs);
  }
  // Scan only after timed loop: scanning cannot inflate any reported packet time.
  const flat=new Float32Array(contract.states.reduce((n,s)=>n+size(s.shape),0));let at=0;for(const s of contract.states){flat.set(states[s.input].data,at);at+=size(s.shape);}
  const row={condition:kind,amplitude,noiseMode,repeat,startedAt,finishedAt:new Date().toISOString(),...stats(times),sessionRun:stats(runTimes),finalAudio:magnitude(lastOutput),finalStates:magnitude(flat)};dispose(states);return row;
}
if(probe){report.probe=await run('demo_peak_0.05',.05,'random',0,1,0);console.log(JSON.stringify(report,null,2));}
else{for(let repeat=0;repeat<repeats;repeat++){const order=repeat%2?[...conditions].reverse():conditions;for(const noiseMode of ['random','zeros'])for(const [kind,amplitude] of order){const row=await run(kind,amplitude,noiseMode,repeat,measuredBlocks,warmBlocks);report.results.push(row);console.log(JSON.stringify({condition:kind,noiseMode,repeat,meanMs:row.meanMs,p95Ms:row.p95Ms,maxMs:row.maxMs,subnormalStates:row.finalStates.subnormal}));}}
  fs.mkdirSync(path.dirname(outputFile),{recursive:true});fs.writeFileSync(outputFile,JSON.stringify(report,null,2)+'\n');console.log('Saved '+outputFile);
}
gain.dispose();bias.dispose();await session.release();
