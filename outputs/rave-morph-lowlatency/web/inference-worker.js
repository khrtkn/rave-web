import {normalizeContract,normalizeParameters,validateMetadata,shapeSize,chooseInferenceThreads,SHARED_LAYOUT,sharedViews,sharedClock,writeFrame,readFrame,cancelSharedTransport,readSharedParameters,waitSharedSignal} from './stream-engine.mjs';

let ort,session,contract,backend,threads,states={},parameters,current,port,queued=null,busy=false,generation=0,lastSequence=null;
let seed=1,noiseMode='random',normalSpare=null,total=0,maxMs=0,timings=[],inferredBlocks=0,stateResets=0;
let commands=Promise.resolve();
let shared=null,stateMode='retain',lastMetricReport=0,stateTensorCopies=0,meanPrep=0,meanRun=0,meanReadback=0,meanState=0,sharedParameterUpdates=0,gpuStateUploadPending=false,gpuStateUploadBatches=0;
function publishMetrics(){const now=performance.now();if(now-lastMetricReport>=250){lastMetricReport=now;self.postMessage({type:'metrics',metrics:metrics()});}}
const dispose=tensors=>{for(const t of Object.values(tensors||{}))try{t.dispose?.();}catch{}};
function uniform(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return ((seed>>>0)+0.5)/4294967296;}
function normal(){if(normalSpare!==null){const n=normalSpare;normalSpare=null;return n;}const r=Math.sqrt(-2*Math.log(uniform())),t=2*Math.PI*uniform();normalSpare=r*Math.sin(t);return r*Math.cos(t);}
function clearStates(){dispose(states);states={};if(contract&&ort)for(const s of contract.states)states[s.input]=new ort.Tensor('float32',new Float32Array(shapeSize(s.shape)),s.shape);gpuStateUploadPending=stateMode==='gpu-retain';lastSequence=null;stateResets++;}
function reset(args={}){
  noiseMode=args.noiseMode??'random';if(!['random','zeros'].includes(noiseMode))throw Error('noiseMode must be random or zeros');
  seed=(Number.isInteger(args.seed)?args.seed:1)>>>0 || 1;normalSpare=null;
  parameters=normalizeParameters(args.parameters||{},contract.latentShape[1],parameters);
  current=structuredClone(parameters);clearStates();total=0;maxMs=0;timings=[];inferredBlocks=0;stateResets=0;stateTensorCopies=0;gpuStateUploadBatches=0;sharedParameterUpdates=0;meanPrep=meanRun=meanReadback=meanState=0;
}
function metrics(){const sorted=[...timings].sort((a,b)=>a-b),mean=inferredBlocks?total/inferredBlocks:0,p95=sorted.length?sorted[Math.ceil(sorted.length*.95)-1]:0;return {inferredBlocks,meanInferenceMs:mean,p95InferenceMs:p95,maxInferenceMs:maxMs,inferenceMeanMs:mean,inferenceP95Ms:p95,inferenceMaxMs:maxMs,lastInferenceMs:timings.at(-1)??0,inferenceBudgetPercent:mean/(contract.frameSize/contract.sampleRate*1000)*100,stateResets,threads,configuredThreads:threads,requestedBackend:backend,stateMode,stateTensorCopies,gpuStateUploadBatches,cpuFallbackDisabled:stateMode==='gpu-retain',stateDataLocation:stateMode==='gpu-retain'?(gpuStateUploadPending?'cpu: initialization or explicit snapshot; next run uploads once':'gpu-buffer'):'cpu',stateFiniteValidation:stateMode==='gpu-retain'?'offline snapshotStates only; live state checks shape/type/location, finite audio checked every block':'all states and audio checked every block',sharedParameterVersion:shared?.parameterVersion??null,sharedParameterRevision:shared?.parameterVersion??null,sharedParameterUpdates,meanInputPreparationMs:inferredBlocks?meanPrep/inferredBlocks:0,meanSessionRunMs:inferredBlocks?meanRun/inferredBlocks:0,meanAudioReadbackMs:inferredBlocks?meanReadback/inferredBlocks:0,meanStateHandlingMs:inferredBlocks?meanState/inferredBlocks:0,inferenceTimingScope:stateMode==='gpu-retain'?'session.run + finite CPU audio copy/readback + GPU state shape/type/location checks; GPU next-state tensors retained without live state readback or finite scan. Excludes input/noise preparation and transport.':stateMode==='retain'?'session.run + finite audio copy/readback + state shape/finite checks; WASM next-state tensors retained, no JS state copy. Excludes input/noise preparation and transport.':'session.run + all state/audio CPU readback/copy, excludes input/noise preparation and transport',p95WindowBlocks:timings.length};}
function checkShape(t,shape,name){if(!t||t.type!=='float32'||t.dims.length!==shape.length||t.dims.some((n,i)=>n!==shape[i]))throw Error('Runtime tensor shape/type mismatch: '+name);}
async function readTensor(t,shape,name){
  checkShape(t,shape,name);
  const data=typeof t.getData==='function'?await t.getData():t.data;
  if(!(data instanceof Float32Array)||data.length!==shapeSize(shape)||!data.every(Number.isFinite))throw Error('Non-finite or invalid runtime tensor: '+name);
  return new Float32Array(data);
}
async function infer(audio){
  if(!session)throw Error('Model not loaded');
  if(!(audio instanceof Float32Array)||audio.length!==contract.frameSize||!audio.every(Number.isFinite))throw Error('Audio block must contain frameSize finite float32 samples');
  const feeds={...states},temporary={};let outputs,nextStates={};const preparationStart=performance.now();
  try {
    if(stateMode==='gpu-retain')for(const s of contract.states){const value=states[s.input];checkShape(value,s.shape,s.input);const expected=gpuStateUploadPending?'cpu':'gpu-buffer';if(value.location!==expected)throw Error('GPU retained state input location mismatch: '+s.input+' (expected '+expected+', got '+value.location+')');}
    temporary[contract.audioInput]=new ort.Tensor('float32',audio,contract.audioInputShape);
    const alpha=1-Math.exp(-contract.frameSize/contract.sampleRate/0.05);
    for(const [key,name] of [['latentGain',contract.latentGainInput],['latentBias',contract.latentBiasInput]]){
      const data=new Float32Array(current[key].length);for(let i=0;i<data.length;i++){current[key][i]+=alpha*(parameters[key][i]-current[key][i]);data[i]=current[key][i];}
      temporary[name]=new ort.Tensor('float32',data,contract.latentShape);
    }
    for(const n of contract.noiseInputs){const data=new Float32Array(shapeSize(n.shape));if(noiseMode!=='zeros')for(let i=0;i<data.length;i++)data[i]=n.distribution==='normal'?normal():uniform()*2-1;temporary[n.name]=new ort.Tensor('float32',data,n.shape);}
    Object.assign(feeds,temporary);const start=performance.now();outputs=await session.run(feeds);const runEnd=performance.now();
    if(stateMode==='gpu-retain'&&outputs[contract.audioOutput]?.location!=='cpu')throw Error('GPU mode requires CPU audio output: '+contract.audioOutput);
    const audioOut=await readTensor(outputs[contract.audioOutput],contract.audioOutputShape,contract.audioOutput);
    const readbackEnd=performance.now();
    for(const s of contract.states){
      if(stateMode==='gpu-retain'){
        const value=outputs[s.output];checkShape(value,s.shape,s.output);
        if(value.location!=='gpu-buffer')throw Error('GPU retained state output is not GPU-resident: '+s.output+' ('+value.location+')');
        // Accessing .data or .getData() here would download every state tensor.
        nextStates[s.input]=value;delete outputs[s.output];
      } else if(stateMode==='retain'){
        const value=outputs[s.output];checkShape(value,s.shape,s.output);const data=value.data;
        if(!(data instanceof Float32Array)||data.length!==shapeSize(s.shape)||!data.every(Number.isFinite))throw Error('Non-finite CPU state: '+s.output);
        nextStates[s.input]=value;delete outputs[s.output];
      } else {const data=await readTensor(outputs[s.output],s.shape,s.output);nextStates[s.input]=new ort.Tensor('float32',data,s.shape);stateTensorCopies++;}
    }
    const stateEnd=performance.now(),elapsed=stateEnd-start;dispose(states);states=nextStates;nextStates={};if(stateMode==='gpu-retain'){if(gpuStateUploadPending)gpuStateUploadBatches++;gpuStateUploadPending=false;}
    meanPrep+=start-preparationStart;meanRun+=runEnd-start;meanReadback+=readbackEnd-runEnd;meanState+=stateEnd-readbackEnd;
    total+=elapsed;maxMs=Math.max(maxMs,elapsed);timings.push(elapsed);if(timings.length>128)timings.shift();inferredBlocks++;
    const n=contract.frameSize,left=audioOut.slice(0,n),right=contract.audioOutputShape[1]===2?audioOut.slice(n,n*2):new Float32Array(left);
    return {wet:[left,right],inferenceMs:elapsed};
  } finally {dispose(temporary);dispose(outputs);dispose(nextStates);}
}
function disconnect(){generation++;queued=null;if(shared)cancelSharedTransport(shared.bundle);shared=null;port?.close();port=null;}
function pump(){
  if(busy||!queued)return;
  const packet=queued;queued=null;busy=true;const token=generation,replyPort=port;
  commands=commands.then(async()=>{
    if(token!==generation||!session)return;
    try {
      if(lastSequence!==null&&packet.sequence!==lastSequence+1)clearStates();
      const result=await infer(packet.audio);lastSequence=packet.sequence;
      if(token===generation){replyPort.postMessage({type:'audio',sequence:packet.sequence,frameStart:packet.frameStart,dry:packet.audio,...result},[packet.audio.buffer,...result.wet.map(c=>c.buffer)]);publishMetrics();}
    } catch(error){clearStates();if(token===generation){replyPort.postMessage({type:'drop',sequence:packet.sequence});self.postMessage({type:'streamError',error:error.message||String(error)});}}
  }).finally(()=>{busy=false;pump();});
}
function claimInput(s){
  const l=SHARED_LAYOUT;let chosen=-1,sequence=Infinity;
  for(let slot=0;slot<l.slots;slot++){const at=slot*l.stride;if(Atomics.load(s.inputMeta,at+l.status)===l.ready&&s.inputMeta[at+l.sequence]<sequence){chosen=slot;sequence=s.inputMeta[at+l.sequence];}}
  if(chosen<0)return null;const at=chosen*l.stride;
  if(Atomics.compareExchange(s.inputMeta,at+l.status,l.ready,l.reading)!==l.ready)return null;
  const packet={sequence,frameStart:readFrame(s.inputMeta,at+l.startLo,at+l.startHi),captureEnd:readFrame(s.inputMeta,at+l.endLo,at+l.endHi),audio:new Float32Array(s.inputAudio.subarray(chosen*contract.frameSize,(chosen+1)*contract.frameSize))};
  Atomics.store(s.inputMeta,at+l.status,l.empty);return packet;
}
function publishOutput(s,packet,result,processingFrame){
  const l=SHARED_LAYOUT;let chosen=-1;
  for(let slot=0;slot<l.slots;slot++){if(Atomics.compareExchange(s.outputMeta,slot*l.stride+l.status,l.empty,l.writing)===l.empty){chosen=slot;break;}}
  if(chosen<0){let oldest=Infinity,index=-1;for(let slot=0;slot<l.slots;slot++){const at=slot*l.stride;if(Atomics.load(s.outputMeta,at+l.status)===l.ready&&s.outputMeta[at+l.sequence]<oldest){oldest=s.outputMeta[at+l.sequence];index=slot;}}
    if(index>=0&&Atomics.compareExchange(s.outputMeta,index*l.stride+l.status,l.ready,l.writing)===l.ready){chosen=index;Atomics.add(s.control,l.outputDrops,1);}}
  if(chosen<0){Atomics.add(s.control,l.outputDrops,1);return;}
  const at=chosen*l.stride,n=contract.frameSize,offset=chosen*n*3;
  s.outputAudio.set(packet.audio,offset);s.outputAudio.set(result.wet[0],offset+n);s.outputAudio.set(result.wet[1],offset+n*2);
  s.outputMeta[at+l.sequence]=packet.sequence;writeFrame(s.outputMeta,at+l.startLo,at+l.startHi,packet.frameStart);writeFrame(s.outputMeta,at+l.endLo,at+l.endHi,packet.captureEnd);writeFrame(s.outputMeta,at+l.runLo,at+l.runHi,processingFrame);
  const published=sharedClock(s.control);writeFrame(s.outputMeta,at+l.doneLo,at+l.doneHi,published);writeFrame(s.outputMeta,at+l.publishLo,at+l.publishHi,published);Atomics.store(s.outputMeta,at+l.status,l.ready);
}
function launchShared(s,token){
  if(s.running)return;s.running=true;
  void (async()=>{
    while(token===generation&&Atomics.load(s.control,SHARED_LAYOUT.active)){
      const observed=Atomics.load(s.control,SHARED_LAYOUT.wake),packet=claimInput(s);
      if(!packet){if(s.wakeMode==='message-port')break;await waitSharedSignal(s.control,observed,s.wakeMode);continue;}
      const task=commands.then(async()=>{
        if(token!==generation||!Atomics.load(s.control,SHARED_LAYOUT.active))return;
        const update=readSharedParameters(s.bundle,contract.latentShape[1],s.parameterVersion);if(update){parameters=update.parameters;s.parameterVersion=update.version;sharedParameterUpdates++;}
        const processingFrame=sharedClock(s.control);
        if(lastSequence!==null&&packet.sequence!==lastSequence+1)clearStates();
        const result=await infer(packet.audio);lastSequence=packet.sequence;
        if(token===generation&&Atomics.load(s.control,SHARED_LAYOUT.active)){publishOutput(s,packet,result,processingFrame);publishMetrics();}
      });
      commands=task.catch(()=>{});await task;
    }
  })().catch(error=>{if(token===generation){Atomics.add(s.control,SHARED_LAYOUT.errors,1);cancelSharedTransport(s.bundle);self.postMessage({type:'streamError',error:error.message||String(error)});}}).finally(()=>{s.running=false;});
}
async function handle(command,args={}) {
  if(command==='release'){disconnect();dispose(states);states={};const old=session;session=null;contract=null;await old?.release();return {released:true};}
  if(command==='loadModel'){
    disconnect();dispose(states);states={};const old=session;session=null;await old?.release();
    contract=normalizeContract(args.contract);backend=args.backend??'wasm';if(!['wasm','webgpu'].includes(backend))throw Error('Unknown backend');
    stateMode=args.stateMode??(backend==='wasm'?'retain':'copy');if(!['retain','copy','gpu-retain'].includes(stateMode)||backend!=='wasm'&&stateMode==='retain'||backend!=='webgpu'&&stateMode==='gpu-retain')throw Error('State mode must be copy, WASM retain, or WebGPU gpu-retain');
    const imported=await import(backend==='webgpu'?'./vendor/ort/ort.webgpu.bundle.min.mjs':'./vendor/ort/ort.wasm.bundle.min.mjs');ort=imported.default||imported;
    const requestedThreads=args.threads??1;threads=chooseInferenceThreads(requestedThreads,backend,!!self.crossOriginIsolated);
    ort.env.wasm.wasmPaths=new URL('./vendor/ort/',self.location.href).href;ort.env.wasm.numThreads=threads;ort.env.wasm.proxy=false;
    const bytes=new Uint8Array(args.bytes);const sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');
    if(contract.sha256&&sha256!==contract.sha256)throw Error('Model SHA-256 differs from contract');
    const sessionOptions={executionProviders:[backend],executionMode:'sequential',graphOptimizationLevel:'all'};
    if(stateMode==='gpu-retain'){
      // ORT rejects session creation if any graph node needs CPU EP fallback.
      // No retry with different providers or output locations is performed.
      sessionOptions.extra={session:{disable_cpu_ep_fallback:'1'}};
      sessionOptions.preferredOutputLocation=Object.fromEntries([[contract.audioOutput,'cpu'],...contract.states.map(s=>[s.output,'gpu-buffer'])]);
    }
    try {session=await ort.InferenceSession.create(bytes,sessionOptions);validateMetadata(session,contract);parameters=undefined;reset();}
    catch(error){const bad=session;session=null;await bad?.release();throw error;}
    return {sha256,modelBytes:bytes.length,threads,configuredThreads:threads,requestedThreads,requestedBackend:backend,stateMode,cpuFallbackDisabled:stateMode==='gpu-retain',preferredStateOutputLocation:stateMode==='gpu-retain'?'gpu-buffer':'cpu',preferredAudioOutputLocation:'cpu',runtime:'onnxruntime-web 1.30.0',backendNote:stateMode==='gpu-retain'?'WebGPU required; CPU EP fallback disabled. All state outputs must remain GPU-resident. Thread count is configured WASM threads, not CPU utilization.':'Requested provider; WebGPU may use internal CPU fallback. Thread count is configured WASM threads, not CPU utilization.',morphoVerified:false};
  }
  if(!session)throw Error('Load model first');
  if(command==='reset'){reset(args);return {reset:true};}
  if(command==='warmup'){
    const blocks=args.blocks??32;if(!Number.isInteger(blocks)||blocks<0||blocks>2048||port||shared)throw Error('Warmup requires 0–2048 blocks and a disconnected worker');
    const start=performance.now();for(let i=0;i<blocks;i++)await infer(new Float32Array(contract.frameSize));
    const warmed=metrics();
    const tail=timings.slice(-64).sort((a,b)=>a-b),tailP95=tail.length?tail[Math.ceil(tail.length*.95)-1]:0;
    // Keep the warmed causal state, excluding startup work from live counters.
    total=0;maxMs=0;timings=[];inferredBlocks=0;stateResets=0;stateTensorCopies=0;gpuStateUploadBatches=0;sharedParameterUpdates=0;meanPrep=meanRun=meanReadback=meanState=0;lastMetricReport=0;
    return {warmupBlocks:blocks,warmupMs:performance.now()-start,warmupP95Ms:tailP95,warmupMeanInferenceMs:warmed.meanInferenceMs,warmupP95WindowBlocks:tail.length,warmupTimingScope:warmed.inferenceTimingScope,warmupInput:'zero audio with configured noise and latent controls; last <=64 warm blocks for target calibration; synchronous offline cadence, excludes worker wake/render jitter'};
  }
  if(command==='parameters'){parameters=normalizeParameters(args.parameters,contract.latentShape[1],parameters);return {parameters};}
  if(command==='connect'){
    disconnect();port=args.port;const token=generation;
    port.onmessage=({data})=>{if(token!==generation||data.type!=='audio')return;if(queued)port.postMessage({type:'drop',sequence:queued.sequence});queued=data;pump();};port.start();return {connected:true,transportMode:'message-channel',wakeMode:'message-port'};
  }
  if(command==='connectShared'){
    disconnect();if(!self.crossOriginIsolated||typeof SharedArrayBuffer!=='function')throw Error('Shared transport unavailable');
    const bundle=args.shared;if(bundle?.frameSize!==contract.frameSize)throw Error('Shared frame size differs from model');
    const requested=args.wakeMode??'auto';if(!['auto','blocking','async'].includes(requested))throw Error('wakeMode must be auto, blocking or async');
    if(requested==='blocking'&&typeof Atomics.wait!=='function')throw Error('Blocking Atomics.wait unavailable');
    const wakeMode=requested!=='async'&&typeof Atomics.wait==='function'?'atomics-wait':typeof Atomics.waitAsync==='function'?'atomics-wait-async':'message-port';
    readSharedParameters(bundle,contract.latentShape[1]);
    const s={...sharedViews(bundle),bundle,wakeMode,running:false,parameterVersion:-1};shared=s;port=args.port;const token=generation;
    port.onmessage=({data})=>{if(token===generation&&data.type==='wake'&&s.wakeMode==='message-port')launchShared(s,token);};port.start();if(wakeMode!=='atomics-wait')queueMicrotask(()=>launchShared(s,token));
    return {connected:true,transportMode:'shared-ring',wakeMode:s.wakeMode,ringSlots:SHARED_LAYOUT.slots,inputRingSlots:SHARED_LAYOUT.slots,outputRingSlots:SHARED_LAYOUT.slots,inputQueuePolicy:'bounded FIFO; output fixed deadline'};
  }
  if(command==='beginShared'){if(!shared||shared.wakeMode!=='atomics-wait')throw Error('Blocking shared transport is not configured');return {armed:true};}
  if(command==='disconnect'){disconnect();return {disconnected:true};}
  // Separate worker instance recommended for offline operations, to avoid
  // interrupting a live stream's state. This is the same real graph and noise.
  if(command==='infer'){return {...await infer(args.audio),metrics:metrics()};}
  if(command==='render'){
    if(port||shared)throw Error('Offline render requires a separate, disconnected worker');
    const audio=args.audio;if(!(audio instanceof Float32Array)||audio.length<1||audio.length>contract.sampleRate*30||!audio.every(Number.isFinite))throw Error('Offline audio must contain 1 sample to 30 seconds, finite float32');
    reset(args);const channels=contract.audioOutputShape[1],output=Array.from({length:channels},()=>new Float32Array(audio.length));
    for(let offset=0;offset<audio.length;offset+=contract.frameSize){const block=new Float32Array(contract.frameSize);block.set(audio.subarray(offset,offset+contract.frameSize));const r=await infer(block);for(let c=0;c<channels;c++)output[c].set(r.wet[c].subarray(0,Math.min(contract.frameSize,audio.length-offset)),offset);}
    return {audio:output[0],channels:output,sampleRate:contract.sampleRate,metrics:metrics(),seed:args.seed??1,noiseMode,morphoVerified:false};
  }
  if(command==='snapshotStates'){
    if(stateMode==='gpu-retain'&&(port||shared))throw Error('GPU state snapshots require a disconnected worker');
    const snapshot={};
    try {for(const s of contract.states)snapshot[s.input]={dims:s.shape,data:await readTensor(states[s.input],s.shape,s.input)};}
    finally {if(stateMode==='gpu-retain')gpuStateUploadPending=true;}
    return {states:snapshot,stateMode,stateFiniteValidated:true,snapshotScope:stateMode==='gpu-retain'?'Explicit offline GPU state readback and finite check; states are now CPU-resident and the next run uploads once.':'CPU state readback and finite check'};
  }
  throw Error('Unknown worker command: '+command);
}
self.onmessage=({data})=>{
  commands=commands.then(async()=>{try{const result=await handle(data.command,data.args);self.postMessage({id:data.id,ok:true,result});if(data.command==='beginShared'){const s=shared,token=generation;setTimeout(()=>{if(shared===s&&token===generation)launchShared(s,token);},0);}}catch(error){self.postMessage({id:data.id,ok:false,error:error.message||String(error)});}}).catch(error=>self.postMessage({id:data.id,ok:false,error:String(error)}));
};
