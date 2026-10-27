import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {normalizeContract,normalizeParameters,validateMetadata,chooseInferenceThreads,connectPlaybackCeiling,RaveStreamEngine} from '../web/stream-engine.mjs';

// Pin the 16-dimensional Water contract regardless of which UI model is default.
const contract=JSON.parse(fs.readFileSync(new URL('../web/models/catalog.json',import.meta.url),'utf8')).models.find(c=>c.id==='water-latent-2048');
assert.ok(contract,'The Water 2048 contract must remain available for the 16/4 switch test');
test('thread selection defaults to one, admits only one/two, and requires isolation for WASM two',()=>{
  assert.equal(chooseInferenceThreads(),1);assert.equal(chooseInferenceThreads(2,'wasm',true),2);
  assert.equal(chooseInferenceThreads(2,'wasm',false),1);assert.equal(chooseInferenceThreads(2,'webgpu',true),1);
  for(const n of [0,3,1.5,'2',NaN])assert.throws(()=>chooseInferenceThreads(n,'wasm',true),/threads/);
});
test('real model contract covers every graph input and rejects state/control shape conflicts',()=>{
  const c=normalizeContract(contract);
  const meta=(name,shape)=>({name,shape,type:'float32',isTensor:true});
  const inputs=[meta(c.audioInput,c.audioInputShape),meta(c.latentGainInput,c.latentShape),meta(c.latentBiasInput,c.latentShape),...c.noiseInputs.map(x=>meta(x.name,x.shape)),...c.states.map(x=>meta(x.input,x.shape))];
  const outputs=[meta(c.audioOutput,c.audioOutputShape),...c.states.map(x=>meta(x.output,x.shape))];
  const session={inputNames:inputs.map(x=>x.name),outputNames:outputs.map(x=>x.name),inputMetadata:inputs,outputMetadata:outputs};
  validateMetadata(session,c);
  const bad=structuredClone(session);bad.inputMetadata.at(-1).shape=[1,99,99];assert.throws(()=>validateMetadata(bad,c),/metadata/);
  assert.throws(()=>normalizeContract({...c,audioOutputShape:[1,1,c.frameSize+1]}),/Output/);
  const duplicate=structuredClone(c);duplicate.states[0].input=c.audioInput;assert.throws(()=>normalizeContract(duplicate),/duplicate/);
});
test('six UI latent axes preserve the ten neutral dimensions and reject unsafe values',()=>{
  const p=normalizeParameters({latentGain:[0.5,1,1.5,2,1,0],latentBias:[1,0,-1,0.3,0,0]},16);
  assert.deepEqual(p.latentGain.slice(6),Array(10).fill(1));assert.deepEqual(p.latentBias.slice(6),Array(10).fill(0));
  assert.throws(()=>normalizeParameters({latentBias:[NaN]},16),/range/);
  assert.throws(()=>normalizeParameters({outputGain:2},16),/range/);
  assert.throws(()=>normalizeParameters({latentGain:Array(17).fill(1)},16),/dimension/);
});
test('model switching 16 → 4 → 16 resets latent controls, preserves mix and releases each previous worker',async()=>{
  const engine=new RaveStreamEngine(),released=[],commands=[];
  engine.contract=contract;engine.worker={terminate:()=>released.push('initial16')};engine.state='ready';
  engine.parameters=normalizeParameters({wet:.3,outputGain:.4,latentGain:2,latentBias:1},16);
  let created=0;engine._makeWorker=()=>{const index=++created;return {terminate:()=>released.push('created'+index)};};
  engine._rpc=async(command,args)=>{commands.push({command,args});return command==='loadModel'?{threads:1}:{};};
  const four={...structuredClone(contract),id:'dimension-lifecycle-test-4',latentShape:[1,4,1],latentDimensions:4,exposedDimensions:4,warmupBlocks:64};
  await engine.loadModel({bytes:new ArrayBuffer(1),contract:four});
  assert.deepEqual(engine.parameters.latentGain,[1,1,1,1]);assert.deepEqual(engine.parameters.latentBias,[0,0,0,0]);assert.equal(engine.parameters.wet,.3);assert.equal(engine.parameters.outputGain,.4);assert.deepEqual(released,['initial16']);
  engine.setParameters({latentGain:2,latentBias:-1});await new Promise(resolve=>setImmediate(resolve));
  await engine.loadModel({bytes:new ArrayBuffer(1),contract});
  assert.deepEqual(engine.parameters.latentGain,Array(16).fill(1));assert.deepEqual(engine.parameters.latentBias,Array(16).fill(0));assert.equal(engine.parameters.wet,.3);assert.equal(engine.parameters.outputGain,.4);assert.deepEqual(released,['initial16','created1']);
  await engine.dispose();assert.deepEqual(released,['initial16','created1','created2']);assert.equal(commands.filter(x=>x.command==='release').length,3);assert.equal(engine.state,'empty');
});
test('restarting from mic to buffer replaces all run metrics before warmup and keeps only immutable model information',async()=>{
  const snapshots=[],calls=[];const engine=new RaveStreamEngine({onMetrics:m=>snapshots.push(m)});
  engine.contract={...contract,warmupBlocks:64};engine.worker={};engine.state='ready';
  engine.runtimeInfo=Object.freeze({sha256:'model-hash',runtime:'ORT',requestedBackend:'wasm',threads:1,configuredThreads:1,frameSize:2048,sampleRate:48000});
  engine.metrics={...engine.runtimeInfo,source:'microphone',capturedBlocks:99,renderedBlocks:98,renderCallbacks:1234,recordingLimitReached:true,inputDeviceSettings:{deviceId:'old-device'},inputPreview:new Float32Array([1]),outputPreview:new Float32Array([2]),unknownPreviousRunField:'must disappear'};
  engine._rpc=async(command,args)=>{calls.push({command,args});if(command==='warmup')throw Error('stop before browser construction');return {};};
  await assert.rejects(engine.start({source:'buffer',buffer:{}}),/stop before browser construction/);
  const starting=snapshots.find(m=>m.state==='starting');assert.ok(starting);assert.equal(starting.source,'buffer');
  for(const key of ['capturedBlocks','renderedBlocks','renderCallbacks','inferredBlocks','underrunSamples','inputPeak','outputPeak'])assert.equal(starting[key],0,key);
  assert.equal(starting.recordingLimitReached,false);assert.equal(starting.transportDelayMs,null);assert.equal(starting.sha256,'model-hash');assert.equal(starting.configuredThreads,1);
  for(const key of ['inputDeviceSettings','inputPreview','outputPreview','unknownPreviousRunField'])assert.equal(Object.hasOwn(starting,key),false,key);
  assert.equal(calls.find(c=>c.command==='warmup').args.blocks,64);
});
function worklet({size=256,delay=0}={}) {
  let Processor;const control=[],packets=[];
  const sandbox={Float32Array,Set,Math,Number,Array,currentFrame:0,sampleRate:48000,registerProcessor:(name,value)=>{assert.equal(name,'rave-stream-transport');Processor=value;},AudioWorkletProcessor:class{constructor(){this.port={postMessage:p=>control.push(p)};}}};
  vm.runInNewContext(fs.readFileSync(new URL('../web/transport-worklet.js',import.meta.url),'utf8'),sandbox);
  const instance=new Processor({processorOptions:{frameSize:size,dryOffsetSamples:delay}});
  instance.audioPort={postMessage:p=>packets.push(p),close(){}};
  const render=(input=[new Float32Array(128)])=>{const out=[[new Float32Array(128),new Float32Array(128)],[new Float32Array(128),new Float32Array(128)]];const active=instance.process([input],out);sandbox.currentFrame+=128;return {out,active};};
  return {instance,packets,control,render,sandbox};
}
test('stereo capture is averaged to mono; pending input is bounded under a stalled worker',()=>{
  const w=worklet();for(let i=0;i<12;i++)w.render([new Float32Array(128).fill(1),new Float32Array(128).fill(3)]);
  assert.equal(w.packets.length,2);assert.equal(w.instance.pending.size,2);assert.equal(w.instance.stats.droppedInputBlocks,4);
  assert.ok(w.packets[0].audio.every(n=>n===2));
});
test('paired dry/wet packets render stereo exactly, apply declared dry offset, then silence on underrun',()=>{
  const w=worklet({delay:3});w.render();w.render();const p=w.packets[0];
  const dry=Float32Array.from({length:256},(_,i)=>i/256),left=new Float32Array(256).fill(.25),right=new Float32Array(256).fill(-.5);
  w.instance.receive({...p,dry,wet:[left,right]});
  const startup=w.render().out;assert.ok(startup.flat().every(c=>c.every(n=>n===0)));assert.equal(w.instance.stats.underrunSamples,0);
  const first=w.render().out;assert.equal(first[0][0][0],.25);assert.equal(first[0][1][127],-.5);
  assert.deepEqual(Array.from(first[1][0].slice(0,5)),[0,0,0,0,1/256]);
  w.render();const missing=w.render().out;assert.ok(missing.flat().every(c=>c.every(n=>n===0)));assert.equal(w.instance.stats.underrunSamples,128);
  assert.equal(w.instance.stats.transportMs,384/48000*1000);
});
test('one-quantum startup headroom absorbs a next-packet quantum of jitter without a model-block prefill',()=>{
  const w=worklet();w.render();w.render();
  const packet=p=>({...p,dry:new Float32Array(256),wet:[new Float32Array(256).fill(.25),new Float32Array(256).fill(.25)]});
  w.instance.receive(packet(w.packets[0]));
  w.render();assert.equal(w.instance.stats.startupBufferSamples,128);assert.equal(w.instance.stats.startupBufferMs,128/48000*1000);
  w.render();w.render();
  // Sequence 1 arrives one quantum after its earliest possible playback time.
  w.instance.receive(packet(w.packets[1]));const next=w.render().out;
  assert.ok(next[0][0].every(n=>n===.25));assert.equal(w.instance.stats.underrunSamples,0);assert.equal(w.instance.stats.transportMs,384/48000*1000);
});
test('waveform snapshots contain the last 64 real samples, remain bounded and do not alias prior reports',()=>{
  const w=worklet();const first=Float32Array.from({length:128},(_,i)=>i/256);
  w.render([first]);const before=w.control.find(p=>p.type==='stats');
  assert.equal(before.inputPreview.length,64);assert.equal(before.outputPreview.length,64);assert.deepEqual(Array.from(before.inputPreview),Array.from(first.slice(64)));assert.ok(before.outputPreview.every(n=>n===0));
  const saved=Array.from(before.inputPreview);
  w.render();const p=w.packets[0],wet=Float32Array.from({length:256},(_,i)=>i/512);
  w.instance.receive({...p,dry:new Float32Array(256),wet:[wet,new Float32Array(wet)]});w.render();
  w.instance.nextReport=0;w.render([new Float32Array(128).fill(-.5)]);
  const after=w.control.filter(p=>p.type==='stats').at(-1);
  assert.equal(after.inputPreview.length,64);assert.ok(after.inputPreview.every(n=>n===-.5));assert.deepEqual(Array.from(after.outputPreview),Array.from(wet.slice(64,128)));assert.deepEqual(Array.from(before.inputPreview),saved);
  assert.equal(w.instance.previewInput.length,64);assert.equal(w.instance.previewOutput.length,64);
});
test('invalid or expired packets cannot render; output queue is bounded and stop clears it',()=>{
  const w=worklet();w.render();w.render();const p=w.packets[0];
  w.instance.receive({...p,dry:new Float32Array(256),wet:[new Float32Array(256).fill(NaN),new Float32Array(256)]});assert.equal(w.instance.queue.length,0);
  for(let sequence=10;sequence<14;sequence++){w.instance.pending.add(sequence);w.instance.receive({type:'audio',sequence,frameStart:w.sandbox.currentFrame,dry:new Float32Array(256),wet:[new Float32Array(256),new Float32Array(256)]});}
  assert.equal(w.instance.queue.length,2);assert.equal(w.instance.queue[0].sequence,12);
  w.instance.port.onmessage({data:{type:'stop'}});assert.equal(w.instance.pending.size,0);assert.equal(w.instance.queue.length,0);assert.equal(w.render().active,false);
});
test('recording is bounded by requested sample count and includes actual rendered silence',()=>{
  const w=worklet();w.instance.port.onmessage({data:{type:'recordStart',maxSamples:250}});w.render();w.render();w.render();
  const records=w.control.filter(p=>p.type==='record');assert.equal(records.reduce((n,p)=>n+p.wet.length,0),250);assert.ok(records.every(p=>p.wet.every(n=>n===0)));assert.equal(w.instance.recording,null);
});
test('stop releases microphone tracks, source, audio nodes and context before disposing worker',async()=>{
  const engine=new RaveStreamEngine();const events=[];engine.state='running';engine.contract=contract;
  engine.source={stop:()=>events.push('sourceStop'),disconnect:()=>events.push('sourceDisconnect')};engine.mediaStream={getTracks:()=>[{stop:()=>events.push('trackStop')}]};
  engine.node={port:{postMessage:p=>events.push(p.type),close:()=>{}},disconnect:()=>events.push('nodeDisconnect')};engine.context={state:'suspended',close:async()=>events.push('contextClose')};engine.safetyCeiling={disconnect:()=>events.push('ceilingDisconnect')};
  await engine.stop();assert.ok(events.includes('trackStop'));assert.ok(events.includes('ceilingDisconnect'));assert.ok(events.includes('contextClose'));assert.equal(engine.safetyCeiling,null);assert.equal(engine.context,null);assert.equal(engine.state,'ready');
});
test('playback ceiling is the final destination node with bounded endpoints and no oversampling',()=>{
  const connections=[],destination={},shaper={connect:n=>connections.push(['ceiling',n])};
  const source={connect:n=>connections.push(['source',n])};const context={destination,createWaveShaper:()=>shaper};
  assert.equal(connectPlaybackCeiling(context,source),shaper);assert.deepEqual(connections,[['source',shaper],['ceiling',destination]]);
  assert.equal(shaper.oversample,'none');assert.equal(shaper.curve.length,2);assert.ok(Math.abs(shaper.curve[0]+.98)<1e-6);assert.ok(Math.abs(shaper.curve[1]-.98)<1e-6);
});
test('rapid knob input coalesces into one pending parameter update without an unbounded RPC queue',async()=>{
  const engine=new RaveStreamEngine();engine.worker={};const calls=[],resolvers=[];
  engine._rpc=(command,args)=>{calls.push({command,args});return new Promise(resolve=>resolvers.push(resolve));};
  for(let i=0;i<100;i++)engine.setParameters({latentBias:[i/100]});
  assert.equal(calls.length,1);assert.equal(engine.requests.size,0);
  resolvers.shift()();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls.length,2);assert.equal(calls[1].args.parameters.latentBias[0],.99);
  resolvers.shift()();await new Promise(resolve=>setImmediate(resolve));assert.equal(engine.pendingParameters,null);assert.equal(engine.parameterSending,false);
});
test('fatal worker error invalidates the model and closes the active audio context',async()=>{
  const previous=globalThis.Worker;const events=[];globalThis.Worker=class{terminate(){events.push('workerTerminate');}};
  try {
    const engine=new RaveStreamEngine({onError:()=>{}});engine.contract=contract;engine.state='running';engine.worker=engine._makeWorker();
    engine.context={state:'suspended',close:async()=>events.push('contextClose')};engine.mediaStream={getTracks:()=>[{stop:()=>events.push('trackStop')}]};
    engine.worker.onerror({message:'fatal test'});await engine.chain;
    assert.equal(engine.worker,null);assert.equal(engine.contract,null);assert.equal(engine.context,null);assert.equal(engine.state,'empty');assert.deepEqual(events,['workerTerminate','trackStop','contextClose']);
  } finally {globalThis.Worker=previous;}
});
