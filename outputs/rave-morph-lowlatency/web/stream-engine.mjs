// Browser-local RAVE streaming. No Neutone Morpho engine is used here.
// Capacity retains early packets until their fixed deadlines (including48ms).
// It is never prefilled as an extra playback delay.
export const SHARED_LAYOUT=Object.freeze({slots:8,stride:12,status:0,sequence:1,startLo:2,startHi:3,endLo:4,endHi:5,publishLo:6,publishHi:7,runLo:8,runHi:9,doneLo:10,doneHi:11,empty:0,ready:1,reading:2,writing:3,active:0,wake:1,clockLo:2,clockHi:3,inputDrops:4,outputDrops:5,errors:6,clockVersion:7,controlWords:8});
export function createSharedTransport(frameSize,dimensions=16,parameters={}) {
  if(!Number.isInteger(frameSize)||frameSize<128||frameSize>32768)throw Error('Invalid shared frameSize');
  if(typeof SharedArrayBuffer!=='function')throw Error('SharedArrayBuffer unavailable');
  if(!Number.isInteger(dimensions)||dimensions<1||dimensions>256)throw Error('Invalid shared latent dimensions');
  const l=SHARED_LAYOUT,bundle={frameSize,dimensions,layout:l,control:new SharedArrayBuffer(l.controlWords*4),inputMeta:new SharedArrayBuffer(l.slots*l.stride*4),outputMeta:new SharedArrayBuffer(l.slots*l.stride*4),inputAudio:new SharedArrayBuffer(l.slots*frameSize*4),outputAudio:new SharedArrayBuffer(l.slots*frameSize*3*4),parameterControl:new SharedArrayBuffer(8),parameterData:new SharedArrayBuffer((dimensions*2+2)*4)};
  Atomics.store(new Int32Array(bundle.control),l.active,1);Atomics.store(new Int32Array(bundle.parameterControl),1,dimensions);writeSharedParameters(bundle,normalizeParameters(parameters,dimensions));return bundle;
}
export function sharedViews(bundle) {return {control:new Int32Array(bundle.control),inputMeta:new Int32Array(bundle.inputMeta),outputMeta:new Int32Array(bundle.outputMeta),inputAudio:new Float32Array(bundle.inputAudio),outputAudio:new Float32Array(bundle.outputAudio)};}
export function writeSharedParameters(bundle,parameters){
  const header=new Int32Array(bundle.parameterControl),data=new Float32Array(bundle.parameterData),dimensions=Atomics.load(header,1);
  const checked=normalizeParameters(parameters,dimensions);Atomics.add(header,0,1);
  data.set(checked.latentGain,0);data.set(checked.latentBias,dimensions);data[dimensions*2]=checked.wet;data[dimensions*2+1]=checked.outputGain;
  Atomics.add(header,0,1);const control=new Int32Array(bundle.control);Atomics.add(control,SHARED_LAYOUT.wake,1);Atomics.notify(control,SHARED_LAYOUT.wake,1);
}
export function readSharedParameters(bundle,dimensions,previousVersion=-1){
  const header=new Int32Array(bundle.parameterControl),data=new Float32Array(bundle.parameterData);
  if(Atomics.load(header,1)!==dimensions||data.length!==dimensions*2+2)throw Error('Shared parameter dimensions differ from model');
  for(let attempt=0;attempt<3;attempt++){
    const version=Atomics.load(header,0);if(version===previousVersion)return null;if(version&1)continue;
    const parameters={latentGain:Array.from(data.subarray(0,dimensions)),latentBias:Array.from(data.subarray(dimensions,dimensions*2)),wet:data[dimensions*2],outputGain:data[dimensions*2+1]};
    if(version===Atomics.load(header,0))return {version,parameters:normalizeParameters(parameters,dimensions)};
  }
  return null;
}
export function waitSharedSignal(control,observed,mode){
  if(mode==='atomics-wait')return Atomics.wait(control,SHARED_LAYOUT.wake,observed,100);
  const waiter=Atomics.waitAsync(control,SHARED_LAYOUT.wake,observed,100);return waiter.async?waiter.value:waiter.value;
}
export function writeFrame(array,lo,hi,frame){array[lo]=frame>>>0;array[hi]=Math.floor(frame/4294967296);}
export function readFrame(array,lo,hi){return (array[hi]>>>0)*4294967296+(array[lo]>>>0);}
export function sharedClock(control) {
  const l=SHARED_LAYOUT;
  for(let i=0;i<4;i++){const before=Atomics.load(control,l.clockVersion);if(before&1)continue;const lo=Atomics.load(control,l.clockLo),hi=Atomics.load(control,l.clockHi);if(before===Atomics.load(control,l.clockVersion))return (hi>>>0)*4294967296+(lo>>>0);}
  return (Atomics.load(control,l.clockHi)>>>0)*4294967296+(Atomics.load(control,l.clockLo)>>>0);
}
export function cancelSharedTransport(bundle){if(!bundle)return;const c=new Int32Array(bundle.control),l=SHARED_LAYOUT;Atomics.store(c,l.active,0);Atomics.add(c,l.wake,1);Atomics.notify(c,l.wake);}
export function shapeSize(shape) {
  if(!Array.isArray(shape)||shape.some(n=>!Number.isSafeInteger(n)||n<1))throw Error('Explicit positive tensor shape required');
  const n=shape.reduce((a,b)=>a*b,1);if(n>16000000)throw Error('Tensor too large');return n;
}
export function chooseInferenceThreads(requested=1,backend='wasm',isolated=false) {
  if(![1,2,4].includes(requested))throw Error('threads must be 1, 2 or 4');
  return backend==='wasm'&&isolated?requested:1;
}
export function normalizeContract(c) {
  if(!c||c.dtype!=='float32'||c.streaming!==true||c.stateReset!=='zeros')throw Error('Explicit float32 streaming / zero-state contract required');
  if(!Number.isSafeInteger(c.sampleRate)||c.sampleRate<8000||c.sampleRate>192000||!Number.isSafeInteger(c.frameSize)||c.frameSize<128||c.frameSize>32768)throw Error('Invalid sample rate or frame size');
  if(JSON.stringify(c.audioInputShape)!==JSON.stringify([1,1,c.frameSize]))throw Error('Mono input shape must match frameSize');
  if(![1,2].includes(c.audioOutputShape?.[1])||c.audioOutputShape?.[0]!==1||c.audioOutputShape?.[2]!==c.frameSize||c.audioOutputShape.length!==3)throw Error('Output must be mono or stereo with the same frame size');
  if(!Array.isArray(c.states)||!Array.isArray(c.noiseInputs)||c.overlapSamples!==0)throw Error('Explicit state/noise mappings and overlapSamples:0 required');
  if(c.modelDelaySamples!==null&&(!Number.isInteger(c.modelDelaySamples)||c.modelDelaySamples<0||c.modelDelaySamples>c.sampleRate*2))throw Error('modelDelaySamples must be null or a measured nonnegative offset');
  if(c.warmupBlocks!==undefined&&(!Number.isInteger(c.warmupBlocks)||c.warmupBlocks<0||c.warmupBlocks>2048))throw Error('warmupBlocks must be 0–2048');
  const ins=[c.audioInput,c.latentGainInput,c.latentBiasInput,...c.states.map(s=>s.input),...c.noiseInputs.map(s=>s.name)];
  const outs=[c.audioOutput,...c.states.map(s=>s.output)];
  for(const names of [ins,outs])if(names.some(n=>typeof n!=='string'||!n)||new Set(names).size!==names.length)throw Error('Missing or duplicate tensor mapping');
  shapeSize(c.audioOutputShape);shapeSize(c.latentShape);
  if(c.latentShape.length!==3||c.latentShape[0]!==1||c.latentShape[2]!==1)throw Error('Latent controls must have shape [1, dimensions, 1]');
  for(const s of [...c.states,...c.noiseInputs])shapeSize(s.shape);
  for(const n of c.noiseInputs)if(!['normal','uniform'].includes(n.distribution))throw Error('Unknown noise distribution');
  return structuredClone(c);
}
export function normalizeParameters(value={},dimensions=16,previous={}) {
  const result={...previous};
  for(const [key,low,high,defaultValue] of [['wet',0,1,1],['outputGain',0,1,0.7]]){
    const n=value[key]??result[key]??defaultValue;if(!Number.isFinite(n)||n<low||n>high)throw Error(key+' out of range');result[key]=n;
  }
  for(const [key,neutral,low,high] of [['latentGain',1,0,3],['latentBias',0,-3,3]]) {
    const supplied=value[key]??result[key]??neutral;
    const array=typeof supplied==='number'?Array(dimensions).fill(supplied):Array.from(supplied);
    if(array.length>dimensions||!array.length||array.some(n=>!Number.isFinite(n)||n<low||n>high))throw Error(key+' out of range or wrong dimension');
    result[key]=Array.from({length:dimensions},(_,i)=>array[i]??neutral);
  }
  return result;
}
export function validateMetadata(session,c) {
  const expectedInputs=[{name:c.audioInput,shape:c.audioInputShape},{name:c.latentGainInput,shape:c.latentShape},{name:c.latentBiasInput,shape:c.latentShape},...c.noiseInputs,...c.states.map(s=>({name:s.input,shape:s.shape}))];
  const expectedOutputs=[{name:c.audioOutput,shape:c.audioOutputShape},...c.states.map(s=>({name:s.output,shape:s.shape}))];
  for(const [names,metadata,expected] of [[session.inputNames,session.inputMetadata,expectedInputs],[session.outputNames,session.outputMetadata,expectedOutputs]]) {
    if(names.length!==expected.length||expected.some(e=>!names.includes(e.name)))throw Error('Graph tensor names differ from contract');
    for(const e of expected){const m=Array.isArray(metadata)?metadata.find(m=>m.name===e.name):metadata?.[e.name];
      if(!m||m.type!=='float32'||!Array.isArray(m.shape)||m.shape.length!==e.shape.length||m.shape.some((n,i)=>typeof n==='number'&&n>0&&n!==e.shape[i]))throw Error('Graph metadata differs from contract: '+e.name);
    }
  }
}
function concatenate(chunks) {const out=new Float32Array(chunks.reduce((n,c)=>n+c.length,0));let at=0;for(const c of chunks){out.set(c,at);at+=c.length;}return out;}
export function connectPlaybackCeiling(context,source) {
  const ceiling=context.createWaveShaper();
  // WaveShaper clamps inputs outside [-1,1] to the curve endpoints. Within
  // that domain this two-point curve adds 0.98 gain; no oversampling delay.
  ceiling.curve=new Float32Array([-0.98,0.98]);ceiling.oversample='none';
  source.connect(ceiling);ceiling.connect(context.destination);return ceiling;
}
export function requestMicrophone({deviceId,timeoutMs=30000,getUserMedia}={}) {
  if(!Number.isFinite(timeoutMs)||timeoutMs<=0)throw Error('Invalid microphone permission timeout');
  const constraints={audio:{deviceId:deviceId?{exact:deviceId}:undefined,channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false};
  return new Promise((resolve,reject)=>{
    let expired=false;
    const timer=setTimeout(()=>{expired=true;const error=Error(`マイクの許可待ちが${timeoutMs/1000}秒でタイムアウトしました。許可を確認して再度開始してください。`);error.name='TimeoutError';reject(error);},timeoutMs);
    Promise.resolve().then(()=>getUserMedia?getUserMedia(constraints):navigator.mediaDevices.getUserMedia(constraints)).then(stream=>{
      clearTimeout(timer);
      if(expired){for(const track of stream?.getTracks?.()||[])try{track.stop();}catch{}return;}
      resolve(stream);
    },error=>{clearTimeout(timer);if(!expired)reject(error);});
  });
}
export async function configureOutputDevice(context,outputDeviceId) {
  const supported=typeof context.setSinkId==='function';
  if(outputDeviceId===undefined||outputDeviceId===null||outputDeviceId===''||outputDeviceId==='default')return {outputDeviceSelected:false,sinkIdConfigured:false,outputSinkSupported:supported};
  if(typeof outputDeviceId!=='string')throw Error('outputDeviceId must be a device ID string');
  if(!supported)throw Error('このブラウザはAudioContext.setSinkIdに対応していません。出力先の指定を外して開始してください。');
  await context.setSinkId(outputDeviceId);
  // Report configuration status only; do not publish device identifiers.
  return {outputDeviceSelected:true,sinkIdConfigured:true,outputSinkSupported:true};
}
export class RaveStreamEngine {
  constructor({onMetrics=()=>{},onError=()=>{},onState=()=>{}}={}) {
    this.onMetrics=onMetrics;this.onError=onError;this.onState=onState;this.state='empty';this.runtimeInfo=null;this.metrics={state:this.state};this.parameters=normalizeParameters();this.requests=new Map();this.requestId=0;this.chain=Promise.resolve();this.recording=null;this.pendingParameters=null;this.parameterSending=false;this.closingWorker=false;
  }
  _state(state){this.state=state;this.metrics.state=state;this.onState(state);this._metrics();}
  _metrics(values={}){Object.assign(this.metrics,values,{state:this.state});this.onMetrics(this.getMetrics());}
  getMetrics(){return {...this.metrics,parameters:structuredClone(this.parameters),modelId:this.contract?.id??null};}
  _beginRun(source){
    // Whitelist immutable load information; never merge a previous run's
    // device settings, waveform buffers, counters or recording completion flag.
    this.metrics={...this.runtimeInfo,source,renderCallbacks:0,capturedBlocks:0,inferredBlocks:0,renderedBlocks:0,droppedInputBlocks:0,droppedOutputBlocks:0,underrunSamples:0,startupSilentSamples:0,pendingBlocks:0,queuedBlocks:0,queueSamples:0,meanInferenceMs:0,p95InferenceMs:0,maxInferenceMs:0,inferenceMeanMs:0,inferenceP95Ms:0,inferenceMaxMs:0,lastInferenceMs:0,inferenceBudgetPercent:0,p95WindowBlocks:0,stateResets:0,transportMs:null,transportDelayMs:null,transportMeanMs:null,warmupBlocks:0,warmupMs:0,startupBufferSamples:0,startupBufferMs:0,inputPeak:0,outputPeak:0,wetPeak:0,wetRms:0,baseLatencyMs:null,outputLatencyMs:null,recording:false,recordingLimitReached:false};
    Object.assign(this.metrics,{receivedOutputBlocks:0,workletDroppedOutputBlocks:0,workerDroppedOutputBlocks:0,skippedOutputSamples:0,deadlineSkippedOutputSamples:0,startupUnderrunSamples:0,steadyUnderrunSamples:0,targetDelayMs:null,rawPacketAgeMs:null,outputDeviceSelected:false,sinkIdConfigured:false});
    this._state('starting');
  }
  _serialize(fn){const p=this.chain.then(fn);this.chain=p.catch(()=>{});return p;}
  _makeWorker() {
    const worker=new Worker(new URL('./inference-worker.js',import.meta.url),{type:'module'});
    worker.onmessage=({data})=>{
      if(this.worker!==worker)return;
      if(data.type==='metrics'){this._metrics(data.metrics);return;}
      if(data.type==='streamError'){this.onError(Error(data.error));this.stop().catch(()=>{});return;}
      const request=this.requests.get(data.id);if(!request)return;this.requests.delete(data.id);clearTimeout(request.timeout);data.ok?request.resolve(data.result):request.reject(Error(data.error));
    };
    worker.onerror=e=>{if(this.worker!==worker)return;const error=Error(e.message||'Inference worker failed');for(const r of this.requests.values()){clearTimeout(r.timeout);r.reject(error);}this.requests.clear();worker.terminate();this.worker=null;this.contract=null;this.pendingParameters=null;this.onError(error);this.stop().catch(()=>{});};
    return worker;
  }
  _rpc(command,args={},transfer=[],timeoutMs=120000) {
    if(!this.worker)return Promise.reject(Error('Model worker not loaded'));
    const id=++this.requestId;return new Promise((resolve,reject)=>{const timeout=setTimeout(()=>{this.requests.delete(id);reject(Error(command+' timed out'));},timeoutMs);this.requests.set(id,{resolve,reject,timeout});this.worker.postMessage({id,command,args},transfer);});
  }
  loadModel(options) {return this._serialize(async()=>{
    await this._stop();await this._releaseWorker();this.contract=null;this.runtimeInfo=null;this.metrics={};this._state('loading');
    try {
      const contract=normalizeContract(options.contract),backend=options.backend??'wasm';
      if(!['wasm','webgpu'].includes(backend))throw Error('Backend must be wasm or webgpu');
      const threads=options.threads??1;chooseInferenceThreads(threads,backend,false);
      let bytes;
      if(options.bytes instanceof ArrayBuffer)bytes=options.bytes.slice(0);
      else if(options.bytes instanceof Uint8Array)bytes=options.bytes.slice().buffer;
      else {const url=new URL(options.url,location.href);if(url.origin!==location.origin)throw Error('Model URL must be served from this app origin');const response=await fetch(url);if(!response.ok)throw Error('Model fetch failed: '+response.status);bytes=await response.arrayBuffer();}
      if(!bytes.byteLength||bytes.byteLength>256*1024*1024)throw Error('Model byte size invalid');
      this.worker=this._makeWorker();
      const info=await this._rpc('loadModel',{bytes,contract,backend,threads,stateMode:options.stateMode},[bytes]);this.contract=contract;
      this.parameters=normalizeParameters({latentGain:1,latentBias:0,wet:this.parameters.wet,outputGain:this.parameters.outputGain},contract.latentShape[1]);
      this.runtimeInfo=Object.freeze({...info,frameSize:contract.frameSize,sampleRate:contract.sampleRate,blockDurationMs:contract.frameSize/contract.sampleRate*1000,modelDelaySamples:contract.modelDelaySamples,dryAlignment:contract.modelDelaySamples===null?'transport only; model delay unmeasured':'transport + declared model delay'});
      this.metrics={...this.runtimeInfo};
      this._state('ready');return info;
    } catch(error){await this._releaseWorker();this._state('empty');this.onError(error);throw error;}
  });}
  start(options={}) {return this._serialize(async()=>{
    if(!this.contract||!this.worker)throw Error('Load a model first');await this._stop();this._beginRun(options.source);
    try {
      // Acquire permission before warmup or constructing any live audio graph.
      // A late grant after the timeout is cleaned up inside requestMicrophone.
      if(options.source==='microphone')this.mediaStream=await this._acquireMicrophone(options.deviceId);
      else if(options.source!=='buffer'||!options.buffer)throw Error('start requires source:microphone or source:buffer and AudioBuffer');
      await this._rpc('reset',{seed:options.seed??1,parameters:this.parameters});
      const warmup=await this._rpc('warmup',{blocks:this.contract.warmupBlocks??32});this._metrics(warmup);
      const AudioContextClass=globalThis.AudioContext||globalThis.webkitAudioContext;
      this.context=new AudioContextClass({sampleRate:this.contract.sampleRate,latencyHint:'interactive'});
      this._metrics(await configureOutputDevice(this.context,options.outputDeviceId));
      if(this.context.sampleRate!==this.contract.sampleRate)throw Error('AudioContext did not honor model sample rate');
      await this.context.audioWorklet.addModule(new URL('./transport-worklet.js',import.meta.url));
      const requestedTransport=options.transport??'auto';if(!['auto','shared','message-channel'].includes(requestedTransport))throw Error('Unknown transport');
      const sharedAvailable=globalThis.crossOriginIsolated&&typeof SharedArrayBuffer==='function';
      if(requestedTransport==='shared'&&!sharedAvailable)throw Error('Shared transport requires crossOriginIsolated and SharedArrayBuffer');
      this.sharedTransport=requestedTransport!=='message-channel'&&sharedAvailable?createSharedTransport(this.contract.frameSize,this.contract.latentShape[1],this.parameters):null;
      if(this.sharedTransport){this.pendingParameters=null;if(this.parameterPromise)await this.parameterPromise;writeSharedParameters(this.sharedTransport,this.parameters);}
      if(options.targetDelayMs!==undefined&&(!Number.isFinite(options.targetDelayMs)||options.targetDelayMs<this.contract.frameSize/this.contract.sampleRate*1000||options.targetDelayMs>1000))throw Error('targetDelayMs must be between one model hop and 1000ms');
      this.node=new AudioWorkletNode(this.context,'rave-stream-transport',{numberOfInputs:1,numberOfOutputs:2,outputChannelCount:[2,2],channelCount:1,channelCountMode:'explicit',processorOptions:{frameSize:this.contract.frameSize,dryOffsetSamples:this.contract.modelDelaySamples??0,shared:this.sharedTransport,startupGuardQuanta:options.startupGuardQuanta??1,warmupP95Ms:warmup.warmupP95Ms??0,targetDelayMs:options.targetDelayMs,fixedDeadline:options.fixedDeadline!==false}});
      const transportNode=this.node;this.node.port.onmessage=({data})=>{
        if(this.node!==transportNode)return;
        if(data.type==='stats')this._metrics(data);
        if(data.type==='record'&&this.recording){this.recording.dry.push(data.dry);this.recording.wet.push(data.wet);}
        if(data.type==='recordLimit')this._metrics({recording:false,recordingLimitReached:true});
        if(data.type==='recordStopped'){this.recordStopResolve?.();this.recordStopResolve=null;}
      };
      const channel=new MessageChannel();
      const transportInfo=await this._rpc(this.sharedTransport?'connectShared':'connect',{port:channel.port2,shared:this.sharedTransport,wakeMode:options.wakeMode??'auto'},[channel.port2]);
      this.node.port.postMessage({type:'connect',port:channel.port1,sharedWake:transportInfo.wakeMode==='message-port'},[channel.port1]);
      this._metrics({...transportInfo,transportFallbackReason:!this.sharedTransport&&requestedTransport==='auto'?'crossOriginIsolated / SharedArrayBuffer unavailable':null});
      this.wetGain=this.context.createGain();this.dryGain=this.context.createGain();this.outputGain=this.context.createGain();
      this.wetGain.gain.value=this.parameters.wet;this.dryGain.gain.value=1-this.parameters.wet;this.outputGain.gain.value=0;
      this.node.connect(this.wetGain,0);this.node.connect(this.dryGain,1);this.wetGain.connect(this.outputGain);this.dryGain.connect(this.outputGain);
      this.limiter=null;this.safetyCeiling=connectPlaybackCeiling(this.context,this.outputGain);
      if(options.source==='microphone'){
        this.source=this.context.createMediaStreamSource(this.mediaStream);this.source.connect(this.node);this._metrics({inputDeviceSettings:this.mediaStream.getAudioTracks()[0]?.getSettings()});
      } else if(options.source==='buffer'&&options.buffer){this.source=this.context.createBufferSource();this.source.buffer=options.buffer;this.source.loop=options.loop!==false;this.source.connect(this.node);this.source.start();}
      else throw Error('start requires source:microphone or source:buffer and AudioBuffer');
      await this.context.resume();if(this.sharedTransport&&transportInfo.wakeMode==='atomics-wait')await this._rpc('beginShared');this.outputGain.gain.setTargetAtTime(this.parameters.outputGain,this.context.currentTime,0.02);
      this._metrics({source:options.source,baseLatencyMs:this.context.baseLatency*1000,outputLatencyMs:Number.isFinite(this.context.outputLatency)?this.context.outputLatency*1000:null,inferredBlocks:0,underrunSamples:0,droppedInputBlocks:0,droppedOutputBlocks:0,recording:false});this._state('running');
    } catch(error){await this._stop();this.onError(error);throw error;}
  });}
  _acquireMicrophone(deviceId){return requestMicrophone({deviceId});}
  setParameters(value) {
    this.parameters=normalizeParameters(value,this.contract?.latentShape[1]??16,this.parameters);
    if(this.context){const t=this.context.currentTime;this.wetGain?.gain.setTargetAtTime(this.parameters.wet,t,0.03);this.dryGain?.gain.setTargetAtTime(1-this.parameters.wet,t,0.03);this.outputGain?.gain.setTargetAtTime(this.parameters.outputGain,t,0.03);}
    if(this.sharedTransport&&Atomics.load(new Int32Array(this.sharedTransport.control),SHARED_LAYOUT.active)){writeSharedParameters(this.sharedTransport,this.parameters);this.pendingParameters=null;}
    else if(this.worker&&!this.closingWorker){this.pendingParameters=structuredClone(this.parameters);this._sendParameters();}return structuredClone(this.parameters);
  }
  _sendParameters(){
    if(this.parameterSending||!this.pendingParameters||!this.worker||this.closingWorker)return;
    const worker=this.worker,parameters=this.pendingParameters;this.pendingParameters=null;this.parameterSending=true;
    const promise=this._rpc('parameters',{parameters}).catch(error=>{if(this.worker===worker&&!this.closingWorker)this.onError(error);}).finally(()=>{if(this.parameterPromise===promise)this.parameterPromise=null;this.parameterSending=false;this._sendParameters();});this.parameterPromise=promise;
  }
  reset(){return this._serialize(async()=>{await this._stop();if(this.worker)await this._rpc('reset',{seed:1,parameters:this.parameters});return this.getMetrics();});}
  stop(){return this._serialize(()=>this._stop());}
  async _stop(){
    if(this.recording&&this.node)await this._stopRecording();
    if(this.outputGain&&this.context&&this.context.state==='running'){const time=this.context.currentTime;this.outputGain.gain.cancelScheduledValues(time);this.outputGain.gain.setValueAtTime(this.outputGain.gain.value,time);this.outputGain.gain.linearRampToValueAtTime(0,time+0.02);await new Promise(resolve=>setTimeout(resolve,25));}
    try{this.source?.stop?.();}catch{}this.source?.disconnect();for(const track of this.mediaStream?.getTracks()||[])track.stop();this.mediaStream=null;this.source=null;
    cancelSharedTransport(this.sharedTransport);this.sharedTransport=null;
    if(this.node){this.node.port.postMessage({type:'stop'});this.node.disconnect();this.node.port.close();this.node=null;}
    for(const node of [this.wetGain,this.dryGain,this.outputGain,this.limiter,this.safetyCeiling])node?.disconnect();this.wetGain=this.dryGain=this.outputGain=this.limiter=this.safetyCeiling=null;
    if(this.context){await this.context.close();this.context=null;}
    if(this.worker)await this._rpc('disconnect',{},[],30000).catch(()=>{});
    if(this.state!=='empty'&&this.state!=='loading')this._state(this.contract?'ready':'empty');
  }
  startRecording({maxSeconds=30}={}) {
    if(this.state!=='running'||!this.node)throw Error('Start audio first');if(this.recording)throw Error('Recording already started');if(!Number.isFinite(maxSeconds)||maxSeconds<=0||maxSeconds>30)throw Error('Recording is limited to 30 seconds');
    this.recording={dry:[],wet:[]};this.node.port.postMessage({type:'recordStart',maxSamples:Math.floor(maxSeconds*this.contract.sampleRate)});this._metrics({recording:true,recordingLimitReached:false});
  }
  _stopRecording(){
    if(this.recordStopPromise)return this.recordStopPromise;
    if(!this.recording)return Promise.resolve(null);
    const promise=this._collectRecording();this.recordStopPromise=promise;
    const clear=()=>{if(this.recordStopPromise===promise)this.recordStopPromise=null;};promise.then(clear,clear);return promise;
  }
  async _collectRecording(){
    if(!this.recording)return null;
    if(this.node)await new Promise(resolve=>{let timer;const finish=()=>{clearTimeout(timer);if(this.recordStopResolve===finish)this.recordStopResolve=null;resolve();};this.recordStopResolve=finish;timer=setTimeout(finish,1000);this.node.port.postMessage({type:'recordStop',id:1});});
    const r=this.recording;this.recording=null;this._metrics({recording:false});return {dry:concatenate(r.dry),wet:concatenate(r.wet),sampleRate:this.contract.sampleRate,channels:1,scope:'Aligned transport output, wet stereo downmixed to mono; before Dry/Wet gain, output gain and playback ceiling. Raw model output may exceed full scale. No playback compressor in this low-latency version.'};
  }
  stopRecording(){return this._stopRecording();}
  async _releaseWorker(){if(!this.worker)return;this.closingWorker=true;this.pendingParameters=null;const worker=this.worker;try{await this._rpc('release',{},[],15000);}catch{}worker.terminate();if(this.worker===worker)this.worker=null;for(const r of this.requests.values()){clearTimeout(r.timeout);r.reject(Error('Worker released'));}this.requests.clear();this.parameterSending=false;this.closingWorker=false;}
  dispose(){return this._serialize(async()=>{await this._stop();await this._releaseWorker();this.contract=null;this._state('empty');});}
}
