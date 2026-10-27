import {normalizeContract,normalizeParameters,validateMetadata,shapeSize,chooseInferenceThreads} from './stream-engine.mjs';

let ort,session,contract,backend,threads,states={},parameters,current,port,queued=null,busy=false,generation=0,lastSequence=null;
let seed=1,noiseMode='random',normalSpare=null,total=0,maxMs=0,timings=[],inferredBlocks=0,stateResets=0;
let commands=Promise.resolve();
const dispose=tensors=>{for(const t of Object.values(tensors||{}))try{t.dispose?.();}catch{}};
function uniform(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return ((seed>>>0)+0.5)/4294967296;}
function normal(){if(normalSpare!==null){const n=normalSpare;normalSpare=null;return n;}const r=Math.sqrt(-2*Math.log(uniform())),t=2*Math.PI*uniform();normalSpare=r*Math.sin(t);return r*Math.cos(t);}
function clearStates(){dispose(states);states={};if(contract&&ort)for(const s of contract.states)states[s.input]=new ort.Tensor('float32',new Float32Array(shapeSize(s.shape)),s.shape);lastSequence=null;stateResets++;}
function reset(args={}){
  noiseMode=args.noiseMode??'random';if(!['random','zeros'].includes(noiseMode))throw Error('noiseMode must be random or zeros');
  seed=(Number.isInteger(args.seed)?args.seed:1)>>>0 || 1;normalSpare=null;
  parameters=normalizeParameters(args.parameters||{},contract.latentShape[1],parameters);
  current=structuredClone(parameters);clearStates();total=0;maxMs=0;timings=[];inferredBlocks=0;stateResets=0;
}
function metrics(){const sorted=[...timings].sort((a,b)=>a-b),mean=inferredBlocks?total/inferredBlocks:0,p95=sorted.length?sorted[Math.ceil(sorted.length*.95)-1]:0;return {inferredBlocks,meanInferenceMs:mean,p95InferenceMs:p95,maxInferenceMs:maxMs,inferenceMeanMs:mean,inferenceP95Ms:p95,inferenceMaxMs:maxMs,lastInferenceMs:timings.at(-1)??0,inferenceBudgetPercent:mean/(contract.frameSize/contract.sampleRate*1000)*100,stateResets,threads,configuredThreads:threads,requestedBackend:backend,inferenceTimingScope:'session.run plus all output/state CPU readback; excludes noise preparation, packet transport and rendering',p95WindowBlocks:timings.length};}
async function readTensor(t,shape,name){
  if(!t||t.type!=='float32'||t.dims.length!==shape.length||t.dims.some((n,i)=>n!==shape[i]))throw Error('Runtime tensor shape/type mismatch: '+name);
  const data=typeof t.getData==='function'?await t.getData():t.data;
  if(!(data instanceof Float32Array)||data.length!==shapeSize(shape)||!data.every(Number.isFinite))throw Error('Non-finite or invalid runtime tensor: '+name);
  return new Float32Array(data);
}
async function infer(audio){
  if(!session)throw Error('Model not loaded');
  if(!(audio instanceof Float32Array)||audio.length!==contract.frameSize||!audio.every(Number.isFinite))throw Error('Audio block must contain frameSize finite float32 samples');
  const feeds={...states},temporary={};let outputs,nextStates={};
  try {
    temporary[contract.audioInput]=new ort.Tensor('float32',audio,contract.audioInputShape);
    const alpha=1-Math.exp(-contract.frameSize/contract.sampleRate/0.05);
    for(const [key,name] of [['latentGain',contract.latentGainInput],['latentBias',contract.latentBiasInput]]){
      const data=new Float32Array(current[key].length);for(let i=0;i<data.length;i++){current[key][i]+=alpha*(parameters[key][i]-current[key][i]);data[i]=current[key][i];}
      temporary[name]=new ort.Tensor('float32',data,contract.latentShape);
    }
    for(const n of contract.noiseInputs){const data=new Float32Array(shapeSize(n.shape));if(noiseMode!=='zeros')for(let i=0;i<data.length;i++)data[i]=n.distribution==='normal'?normal():uniform()*2-1;temporary[n.name]=new ort.Tensor('float32',data,n.shape);}
    Object.assign(feeds,temporary);const start=performance.now();outputs=await session.run(feeds);
    const audioOut=await readTensor(outputs[contract.audioOutput],contract.audioOutputShape,contract.audioOutput);
    for(const s of contract.states){const data=await readTensor(outputs[s.output],s.shape,s.output);nextStates[s.input]=new ort.Tensor('float32',data,s.shape);}
    const elapsed=performance.now()-start;dispose(states);states=nextStates;nextStates={};
    total+=elapsed;maxMs=Math.max(maxMs,elapsed);timings.push(elapsed);if(timings.length>128)timings.shift();inferredBlocks++;
    const n=contract.frameSize,left=audioOut.slice(0,n),right=contract.audioOutputShape[1]===2?audioOut.slice(n,n*2):new Float32Array(left);
    return {wet:[left,right],inferenceMs:elapsed};
  } finally {dispose(temporary);dispose(outputs);dispose(nextStates);}
}
function disconnect(){generation++;queued=null;port?.close();port=null;}
function pump(){
  if(busy||!queued)return;
  const packet=queued;queued=null;busy=true;const token=generation,replyPort=port;
  commands=commands.then(async()=>{
    if(token!==generation||!session)return;
    try {
      if(lastSequence!==null&&packet.sequence!==lastSequence+1)clearStates();
      const result=await infer(packet.audio);lastSequence=packet.sequence;
      if(token===generation){replyPort.postMessage({type:'audio',sequence:packet.sequence,frameStart:packet.frameStart,dry:packet.audio,...result},[packet.audio.buffer,...result.wet.map(c=>c.buffer)]);self.postMessage({type:'metrics',metrics:metrics()});}
    } catch(error){clearStates();if(token===generation){replyPort.postMessage({type:'drop',sequence:packet.sequence});self.postMessage({type:'streamError',error:error.message||String(error)});}}
  }).finally(()=>{busy=false;pump();});
}
async function handle(command,args={}) {
  if(command==='release'){disconnect();dispose(states);states={};const old=session;session=null;contract=null;await old?.release();return {released:true};}
  if(command==='loadModel'){
    disconnect();dispose(states);states={};const old=session;session=null;await old?.release();
    contract=normalizeContract(args.contract);backend=args.backend??'wasm';if(!['wasm','webgpu'].includes(backend))throw Error('Unknown backend');
    const imported=await import(backend==='webgpu'?'./vendor/ort/ort.webgpu.bundle.min.mjs':'./vendor/ort/ort.wasm.bundle.min.mjs');ort=imported.default||imported;
    const requestedThreads=args.threads??1;threads=chooseInferenceThreads(requestedThreads,backend,!!self.crossOriginIsolated);
    ort.env.wasm.wasmPaths=new URL('./vendor/ort/',self.location.href).href;ort.env.wasm.numThreads=threads;ort.env.wasm.proxy=false;
    const bytes=new Uint8Array(args.bytes);const sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');
    if(contract.sha256&&sha256!==contract.sha256)throw Error('Model SHA-256 differs from contract');
    try {session=await ort.InferenceSession.create(bytes,{executionProviders:[backend],executionMode:'sequential',graphOptimizationLevel:'all'});validateMetadata(session,contract);parameters=undefined;reset();}
    catch(error){const bad=session;session=null;await bad?.release();throw error;}
    return {sha256,modelBytes:bytes.length,threads,configuredThreads:threads,requestedThreads,requestedBackend:backend,runtime:'onnxruntime-web 1.30.0',backendNote:'Requested provider; WebGPU may use internal CPU fallback. Thread count is configured WASM threads, not CPU utilization.',morphoVerified:false};
  }
  if(!session)throw Error('Load model first');
  if(command==='reset'){reset(args);return {reset:true};}
  if(command==='warmup'){
    const blocks=args.blocks??32;if(!Number.isInteger(blocks)||blocks<0||blocks>128||port)throw Error('Warmup requires 0–128 blocks and a disconnected worker');
    const start=performance.now();for(let i=0;i<blocks;i++)await infer(new Float32Array(contract.frameSize));
    // Keep the warmed causal state, excluding startup work from live counters.
    total=0;maxMs=0;timings=[];inferredBlocks=0;stateResets=0;
    return {warmupBlocks:blocks,warmupMs:performance.now()-start};
  }
  if(command==='parameters'){parameters=normalizeParameters(args.parameters,contract.latentShape[1],parameters);return {parameters};}
  if(command==='connect'){
    disconnect();port=args.port;const token=generation;
    port.onmessage=({data})=>{if(token!==generation||data.type!=='audio')return;if(queued)port.postMessage({type:'drop',sequence:queued.sequence});queued=data;pump();};port.start();return {connected:true};
  }
  if(command==='disconnect'){disconnect();return {disconnected:true};}
  // Separate worker instance recommended for offline operations, to avoid
  // interrupting a live stream's state. This is the same real graph and noise.
  if(command==='infer'){return {...await infer(args.audio),metrics:metrics()};}
  if(command==='render'){
    if(port)throw Error('Offline render requires a separate, disconnected worker');
    const audio=args.audio;if(!(audio instanceof Float32Array)||audio.length<1||audio.length>contract.sampleRate*30||!audio.every(Number.isFinite))throw Error('Offline audio must contain 1 sample to 30 seconds, finite float32');
    reset(args);const channels=contract.audioOutputShape[1],output=Array.from({length:channels},()=>new Float32Array(audio.length));
    for(let offset=0;offset<audio.length;offset+=contract.frameSize){const block=new Float32Array(contract.frameSize);block.set(audio.subarray(offset,offset+contract.frameSize));const r=await infer(block);for(let c=0;c<channels;c++)output[c].set(r.wet[c].subarray(0,Math.min(contract.frameSize,audio.length-offset)),offset);}
    return {audio:output[0],channels:output,sampleRate:contract.sampleRate,metrics:metrics(),seed:args.seed??1,noiseMode,morphoVerified:false};
  }
  throw Error('Unknown worker command: '+command);
}
self.onmessage=({data})=>{
  commands=commands.then(async()=>{try{const result=await handle(data.command,data.args);self.postMessage({id:data.id,ok:true,result});}catch(error){self.postMessage({id:data.id,ok:false,error:error.message||String(error)});}}).catch(error=>self.postMessage({id:data.id,ok:false,error:String(error)}));
};
