import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {Worker as NodeWorker} from 'node:worker_threads';
import * as Runtime from '../web/stream-engine.mjs';
import {normalizeContract,normalizeParameters,validateMetadata,chooseInferenceThreads,connectPlaybackCeiling,requestMicrophone,configureOutputDevice,RaveStreamEngine,createSharedTransport,sharedViews,SHARED_LAYOUT,writeFrame,readFrame,cancelSharedTransport} from '../web/stream-engine.mjs';

// Pin the 16-dimensional Water contract regardless of which UI model is default.
const contract=JSON.parse(fs.readFileSync(new URL('../../rave-morph/web/models/catalog.json',import.meta.url),'utf8')).models.find(c=>c.id==='water-latent-2048');
assert.ok(contract,'The Water 2048 contract must remain available for the 16/4 switch test');
test('thread selection admits one/two/four and requires isolation for multiple WASM threads',()=>{
  assert.equal(chooseInferenceThreads(),1);assert.equal(chooseInferenceThreads(2,'wasm',true),2);
  assert.equal(chooseInferenceThreads(2,'wasm',false),1);assert.equal(chooseInferenceThreads(2,'webgpu',true),1);
  assert.equal(chooseInferenceThreads(4,'wasm',true),4);assert.equal(chooseInferenceThreads(4,'wasm',false),1);assert.equal(chooseInferenceThreads(4,'webgpu',true),1);
  for(const n of [0,3,8,1.5,'2',NaN])assert.throws(()=>chooseInferenceThreads(n,'wasm',true),/threads/);
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
function worklet({size=256,delay=0,shared=null,guard=1,fixedDeadline=true,targetDelayMs,clockNow,clockKind='performance'}={}) {
  let Processor;const control=[],packets=[];
  const sandbox={Float32Array,Int32Array,Atomics,Set,Math,Number,Array,currentFrame:0,sampleRate:48000,registerProcessor:(name,value)=>{assert.equal(name,'rave-stream-transport');Processor=value;},AudioWorkletProcessor:class{constructor(){this.port={postMessage:p=>control.push(p)};}}};
  if(clockNow){sandbox.Date=class{static now(){return clockNow();}};sandbox.performance=clockKind==='performance'?{now:clockNow}:undefined;}
  if(clockKind==='unavailable'){sandbox.Date=undefined;sandbox.performance=undefined;}
  vm.runInNewContext(fs.readFileSync(new URL('../web/transport-worklet.js',import.meta.url),'utf8'),sandbox);
  const instance=new Processor({processorOptions:{frameSize:size,dryOffsetSamples:delay,shared,startupGuardQuanta:guard,fixedDeadline,targetDelayMs}});
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
  w.render();assert.equal(w.instance.stats.configuredGuardSamples,128);assert.equal(w.instance.stats.targetDelayMs,384/48000*1000);assert.equal(w.instance.stats.startupBufferSamples,0);
  w.render();w.render();
  // Sequence 1 arrives one quantum after its earliest possible playback time.
  w.instance.receive(packet(w.packets[1]));const next=w.render().out;
  assert.ok(next[0][0].every(n=>n===.25));assert.equal(w.instance.stats.underrunSamples,0);assert.equal(w.instance.stats.transportMs,384/48000*1000);
});
test('fixed deadlines skip the actual late samples then restore the original timeline and dry offset',()=>{
  const w=worklet({delay:3});w.render();w.render();
  const packet=p=>({...p,dry:Float32Array.from({length:256},(_,i)=>i/256),wet:[Float32Array.from({length:256},(_,i)=>i/512),new Float32Array(256)]});
  w.instance.receive(packet(w.packets[0]));w.render();w.render();w.render();
  const gap=w.render().out;assert.ok(gap[0][0].every(n=>n===0));assert.equal(w.instance.stats.underrunSamples,128);
  w.instance.receive(packet(w.packets[1]));const resumed=w.render().out;
  assert.equal(resumed[0][0][0],128/512/32);assert.equal(resumed[1][0][0],125/256/32);assert.equal(w.instance.stats.skippedOutputSamples,128);
  assert.equal(w.instance.stats.rawPacketAgeMs,512/48000*1000);assert.equal(w.instance.stats.transportMs,384/48000*1000);
  // No permanent extra latency: the following on-time block starts at index 0.
  w.instance.pending.add(2);w.instance.receive(packet({type:'audio',sequence:2,frameStart:512}));const next=w.render().out;
  assert.equal(next[0][0][0],0);assert.equal(w.instance.stats.transportMs,384/48000*1000);
});
test('actual underrun recovery fades only 32 playback samples in place while aligned raw recording and loss counters stay unchanged',()=>{
  const w=worklet({delay:3});w.render();w.render();
  const packet=p=>({...p,dry:new Float32Array(256).fill(.5),wet:[new Float32Array(256).fill(.25),new Float32Array(256).fill(-.75)]});
  w.instance.receive(packet(w.packets[0]));w.render();
  const first=w.render().out;assert.equal(first[0][0][0],.25);assert.equal(w.instance.stats.recoveryFadeEvents,0,'startup silence must not arm recovery');
  w.render();const gap=w.render().out;assert.ok(gap.flat().every(c=>c.every(n=>n===0)));assert.equal(w.instance.stats.underrunSamples,128);
  w.instance.port.onmessage({data:{type:'recordStart',maxSamples:128}});
  w.instance.receive(packet(w.packets[1]));const resumed=w.render().out;
  for(let i=0;i<128;i++){
    const gain=i<32?(i+1)/32:1;
    assert.equal(resumed[0][0][i],.25*gain);assert.equal(resumed[0][1][i],-.75*gain);
    assert.equal(resumed[1][0][i],.5*gain);assert.equal(resumed[1][1][i],.5*gain);
    assert.ok(Math.abs(resumed[0][0][i]*.4+resumed[1][0][i]*.6-(.25*.4+.5*.6)*gain)<1e-12,'Dry/Wet shares one recovery envelope');
  }
  const records=w.control.filter(p=>p.type==='record');assert.equal(records.length,1);assert.equal(records[0].dry.length,128);
  assert.ok(records[0].dry.every(n=>n===.5));assert.ok(records[0].wet.every(n=>n===-.25),'raw stereo-downmixed model samples remain unfaded');
  assert.equal(w.instance.stats.recoveryFadeEvents,1);assert.equal(w.instance.stats.underrunSamples,128);assert.equal(w.instance.stats.skippedOutputSamples,128);assert.equal(w.instance.stats.renderedBlocks,2);
  assert.equal(w.instance.stats.targetDelayMs,384/48000*1000);assert.equal(w.instance.stats.transportMs,384/48000*1000);
  w.instance.pending.add(2);w.instance.receive(packet({type:'audio',sequence:2,frameStart:512}));const next=w.render().out;
  assert.equal(next[0][0][0],.25);assert.equal(next[1][0][0],.5);assert.equal(w.instance.stats.recoveryFadeEvents,1,'continuous packets are not repeatedly faded');
});
test('a wholly late packet preserves known dry history and drains a newer ready packet in the same sample',()=>{
  const w=worklet({size:512,delay:600});w.instance.targetDelaySamples=640;w.instance.stats.targetDelayMs=640/48000*1000;w.instance.started=true;w.instance.startedAtFrame=640;w.instance.startupHoldRemaining=0;w.instance.lastSequence=0;
  w.sandbox.currentFrame=1664; // seq1's 1152 deadline is exactly one full block late.
  const first={sequence:1,frameStart:512,dry:new Float32Array(512).fill(.25),wet:[new Float32Array(512),new Float32Array(512)]};
  const next={sequence:2,frameStart:1024,dry:new Float32Array(512).fill(.5),wet:[new Float32Array(512).fill(.75),new Float32Array(512).fill(.75)]};
  w.instance.queue.push(first,next);const out=w.render().out;
  assert.ok(out[0][0].every(n=>n===.75));assert.equal(out[1][0][88],.25);assert.equal(w.instance.stats.deadlineSkippedOutputSamples,512);assert.equal(w.instance.stats.skippedOutputSamples,512);assert.equal(w.instance.stats.droppedOutputBlocks,1);assert.equal(w.instance.stats.underrunSamples,0);
});
test('shared ring ignores unpublished slots, preserves ownership and emits paired audio without packet messages',()=>{
  const bundle=createSharedTransport(512),v=sharedViews(bundle),l=SHARED_LAYOUT,w=worklet({size:512,shared:bundle});
  for(let i=0;i<4;i++)w.render([new Float32Array(128).fill(i+.25)]);
  assert.equal(w.packets.length,0);assert.equal(Atomics.load(v.inputMeta,l.status),l.ready);assert.equal(readFrame(v.inputMeta,l.startLo,l.startHi),0);assert.equal(readFrame(v.inputMeta,l.endLo,l.endHi),512);
  const dry=new Float32Array(v.inputAudio.subarray(0,512));assert.equal(dry[511],3.25);Atomics.store(v.inputMeta,l.status,l.empty);
  Atomics.store(v.outputMeta,l.status,l.writing);v.outputAudio.set(dry,0);v.outputAudio.fill(.25,512,1024);v.outputAudio.fill(-.5,1024,1536);v.outputMeta[l.sequence]=0;
  writeFrame(v.outputMeta,l.startLo,l.startHi,0);writeFrame(v.outputMeta,l.endLo,l.endHi,512);writeFrame(v.outputMeta,l.runLo,l.runHi,512);writeFrame(v.outputMeta,l.publishLo,l.publishHi,512);
  assert.ok(w.render().out[0][0].every(n=>n===0));assert.equal(Atomics.load(v.outputMeta,l.status),l.writing);
  Atomics.store(v.outputMeta,l.status,l.ready);const rendered=w.render().out;
  assert.ok(rendered[0][0].every(n=>n===.25));assert.ok(rendered[0][1].every(n=>n===-.5));assert.equal(rendered[1][0][0],.25);assert.equal(Atomics.load(v.outputMeta,l.status),l.reading);
  w.render();w.render();w.render();assert.equal(Atomics.load(v.outputMeta,l.status),l.empty);
  w.instance.port.onmessage({data:{type:'stop'}});assert.equal(Atomics.load(v.control,l.active),0);
});
test('shared input ring remains bounded when no worker consumes, including render clock above 32 bits',()=>{
  const bundle=createSharedTransport(256),v=sharedViews(bundle),l=SHARED_LAYOUT,w=worklet({size:256,shared:bundle});
  w.sandbox.currentFrame=4294967296+256;for(let i=0;i<(l.slots+2)*2;i++)w.render();
  assert.equal(w.packets.length,0);assert.equal(w.instance.stats.droppedInputBlocks,2);assert.equal(Atomics.load(v.control,l.inputDrops),2);assert.equal(v.inputAudio.length,256*l.slots);assert.equal(v.outputAudio.length,256*l.slots*3);
  assert.equal(readFrame(v.inputMeta,l.startLo,l.startHi),4294967296+256);
});
for (const target of [24,48]) test(`fast output producer plus ${target}ms deadline retains future blocks without capacity loss`,()=>{
  const bundle=createSharedTransport(512),v=sharedViews(bundle),l=SHARED_LAYOUT,w=worklet({size:512,shared:bundle,targetDelayMs:target});
  let produced=0;
  for(let callback=0;callback<120;callback++){
    w.render();
    for(let input=0;input<l.slots;input++){
      const from=input*l.stride;if(Atomics.load(v.inputMeta,from+l.status)!==l.ready)continue;
      const slot=Array.from({length:l.slots},(_,i)=>i).find(i=>Atomics.load(v.outputMeta,i*l.stride+l.status)===l.empty);
      assert.notEqual(slot,undefined,'A fast producer must not overwrite a future-deadline packet');
      const at=slot*l.stride,offset=slot*512*3;Atomics.store(v.outputMeta,at+l.status,l.writing);v.outputMeta[at+l.sequence]=v.inputMeta[from+l.sequence];
      writeFrame(v.outputMeta,at+l.startLo,at+l.startHi,readFrame(v.inputMeta,from+l.startLo,from+l.startHi));writeFrame(v.outputMeta,at+l.endLo,at+l.endHi,readFrame(v.inputMeta,from+l.endLo,from+l.endHi));writeFrame(v.outputMeta,at+l.runLo,at+l.runHi,w.sandbox.currentFrame-128);writeFrame(v.outputMeta,at+l.publishLo,at+l.publishHi,w.sandbox.currentFrame-128);
      v.outputAudio.fill(.25,offset,offset+512*3);Atomics.store(v.outputMeta,at+l.status,l.ready);Atomics.store(v.inputMeta,from+l.status,l.empty);produced++;
    }
  }
  assert.ok(produced>=29);assert.equal(w.instance.stats.underrunSamples,0);assert.equal(w.instance.stats.skippedOutputSamples,0);assert.equal(w.instance.stats.targetDelayMs,target);assert.equal(w.instance.stats.droppedOutputBlocks,0);
});
test('shared worker output losses are included in totals without double counting worklet losses or overriding real inference counts',()=>{
  const bundle=createSharedTransport(512),v=sharedViews(bundle),l=SHARED_LAYOUT,w=worklet({size:512,shared:bundle});
  Atomics.store(v.control,l.outputDrops,3);w.instance.discard(null);w.render();
  const stats=w.control.find(p=>p.type==='stats');assert.equal(stats.workerDroppedOutputBlocks,3);assert.equal(stats.workletDroppedOutputBlocks,1);assert.equal(stats.droppedOutputBlocks,4);assert.equal(stats.skippedOutputSamples,2048);assert.equal(Object.hasOwn(stats,'inferredBlocks'),false);assert.equal(stats.receivedOutputBlocks,0);
});
test('cancellation wakes a pending Atomics.waitAsync instead of leaving worker stop waiting',async()=>{
  if(typeof Atomics.waitAsync!=='function')return;
  const bundle=createSharedTransport(128),v=sharedViews(bundle),l=SHARED_LAYOUT,observed=Atomics.load(v.control,l.wake);
  const waiter=Atomics.waitAsync(v.control,l.wake,observed,1000);assert.equal(waiter.async,true);
  let timeout;try{cancelSharedTransport(bundle);const value=await Promise.race([waiter.value,new Promise((_,reject)=>{timeout=setTimeout(()=>reject(Error('wait was not cancelled')),100);})]);assert.equal(value,'ok');assert.equal(Atomics.load(v.control,l.active),0);}finally{clearTimeout(timeout);}
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
test('simultaneous automatic/manual recording stops share one completion promise',async()=>{
  const engine=new RaveStreamEngine();engine.contract=contract;engine.recording={dry:[new Float32Array([.2,.3])],wet:[new Float32Array([.4,.5])]};let stopMessages=0;
  engine.node={port:{postMessage:p=>{assert.equal(p.type,'recordStop');stopMessages++;}}};
  const first=engine.stopRecording(),second=engine.stopRecording();assert.equal(first,second);assert.equal(stopMessages,1);engine.recordStopResolve();
  const [a,b]=await Promise.all([first,second]);assert.equal(a,b);assert.equal(a.wet.length,2);assert.equal(engine.recording,null);assert.equal(engine.recordStopPromise,null);
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
test('microphone permission times out, then stops every track from a late grant',async()=>{
  let grant;const stopped=[];
  const permission=requestMicrophone({timeoutMs:5,deviceId:'test-input',getUserMedia:constraints=>{assert.equal(constraints.audio.deviceId.exact,'test-input');return new Promise(resolve=>{grant=resolve;});}});
  await assert.rejects(permission,error=>error.name==='TimeoutError');
  grant({getTracks:()=>[{stop:()=>stopped.push(1)},{stop:()=>stopped.push(2)}]});await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(stopped,[1,2]);
});
test('pending microphone permission creates no audio graph and performs no warmup inference; timeout leaves no active resources',async()=>{
  const engine=new RaveStreamEngine(),commands=[];engine.contract=contract;engine.worker={};engine.state='ready';
  engine._rpc=async command=>{commands.push(command);return {};};
  engine._acquireMicrophone=()=>requestMicrophone({timeoutMs:5,getUserMedia:()=>new Promise(()=>{})});
  await assert.rejects(engine.start({source:'microphone'}),error=>error.name==='TimeoutError');
  assert.equal(commands.includes('warmup'),false);assert.equal(commands.includes('reset'),false);assert.equal(commands.includes('connectShared'),false);assert.equal(commands.includes('connect'),false);
  assert.equal(engine.context,undefined);assert.equal(engine.node,undefined);assert.equal(engine.mediaStream,null);assert.equal(engine.state,'ready');
});
test('output sink selection awaits the specified context routing and reports booleans without identifiers',async()=>{
  let apply,selected,completed=false;const context={setSinkId:id=>{selected=id;return new Promise(resolve=>{apply=resolve;});}};
  const pending=configureOutputDevice(context,'private-output-id').then(value=>{completed=true;return value;});
  assert.equal(selected,'private-output-id');assert.equal(completed,false);apply();const result=await pending;
  assert.deepEqual(result,{outputDeviceSelected:true,sinkIdConfigured:true,outputSinkSupported:true});assert.equal(JSON.stringify(result).includes('private-output-id'),false);
});
test('default output needs no setSinkId support, while an unsupported explicit output rejects',async()=>{
  assert.deepEqual(await configureOutputDevice({},''),{outputDeviceSelected:false,sinkIdConfigured:false,outputSinkSupported:false});
  assert.equal((await configureOutputDevice({},'default')).sinkIdConfigured,false);
  await assert.rejects(configureOutputDevice({},'chosen-output'),/setSinkId/);
});
test('wall-clock callback metrics distinguish render bursts, long pauses and audio-frame jumps',()=>{
  let now=1000;const w=worklet({clockNow:()=>now});w.render();w.render();w.render();now=1020;w.render();
  const s=w.instance.wallStats;assert.equal(s.wallClockSource,'performance.now');assert.equal(s.workletHasDateNow,true);assert.equal(s.callbackGapMaxMs,20);assert.ok(Math.abs(s.callbackGapMeanMs-20/3)<1e-9);
  assert.equal(s.sameTimestampBurstMaxCallbacks,3);assert.equal(s.sameTimestampBurstMaxSamples,384);assert.equal(s.estimatedBatchMaxSamples,384);assert.equal(s.audioFrameGapMaxSamples,128);assert.equal(s.audioVsWallDriftMs,-12);
  w.sandbox.currentFrame+=512;now+=2;w.render();assert.equal(s.audioFrameGapMaxSamples,640);assert.equal(s.nonContiguousAudioFrameCallbacks,1);
});
test('wall-clock diagnostics fall back to Date.now and remain safe when no clock is available',()=>{
  let now=500;const dated=worklet({clockNow:()=>now,clockKind:'date'});dated.render();now+=3;dated.render();assert.equal(dated.instance.wallStats.wallClockSource,'Date.now');assert.equal(dated.instance.wallStats.workletHasPerformanceNow,false);assert.equal(dated.instance.wallStats.callbackGapMeanMs,3);
  const unavailable=worklet({clockKind:'unavailable'});unavailable.render();assert.equal(unavailable.instance.wallStats.wallClockSource,'unavailable');assert.equal(unavailable.instance.wallStats.workletHasDateNow,false);assert.equal(unavailable.instance.wallStats.callbackGapMeanMs,null);
});
test('unsupported explicit output fails before graph creation and closes the context and microphone',async()=>{
  const original=globalThis.AudioContext,events=[];
  globalThis.AudioContext=class{constructor(){this.sampleRate=48000;this.state='suspended';}async close(){events.push('contextClose');}};
  try {
    const engine=new RaveStreamEngine();engine.contract=contract;engine.worker={};engine.state='ready';engine._rpc=async()=>({});
    engine._acquireMicrophone=async()=>({getTracks:()=>[{stop:()=>events.push('trackStop')}]});
    await assert.rejects(engine.start({source:'microphone',outputDeviceId:'chosen-output'}),/setSinkId/);
    assert.deepEqual(events,['trackStop','contextClose']);assert.equal(engine.context,null);assert.equal(engine.mediaStream,null);assert.equal(engine.node,undefined);assert.equal(engine.state,'ready');
  } finally {globalThis.AudioContext=original;}
});
test('shared parameter seqlock rejects partial/stale commits and validates model dimensions',()=>{
  const b=createSharedTransport(512,2),h=new Int32Array(b.parameterControl),data=new Float32Array(b.parameterData);
  const initial=Runtime.readSharedParameters(b,2);assert.deepEqual(initial.parameters.latentGain,[1,1]);assert.equal(Runtime.readSharedParameters(b,2,initial.version),null);
  Atomics.add(h,0,1);data[0]=2;assert.equal(Runtime.readSharedParameters(b,2,initial.version),null);
  data[1]=.5;data[2]=-.25;data[3]=.75;Atomics.add(h,0,1);
  const updated=Runtime.readSharedParameters(b,2,initial.version);assert.deepEqual(updated.parameters.latentGain,[2,.5]);assert.deepEqual(updated.parameters.latentBias,[-.25,.75]);assert.equal(updated.version,initial.version+2);
  assert.throws(()=>Runtime.readSharedParameters(b,4),/dimensions/);
});
test('live shared parameter updates bypass RPC while preserving latest float32 control commits',()=>{
  const e=new RaveStreamEngine();e.contract={...contract,latentShape:[1,2,1]};e.parameters=normalizeParameters({},2);e.sharedTransport=createSharedTransport(512,2);e.worker={};let rpcs=0;e._rpc=()=>{rpcs++;return Promise.resolve({});};
  for(let i=0;i<20;i++)e.setParameters({latentGain:[1+i/20,1],latentBias:[i/20,0]});
  const applied=Runtime.readSharedParameters(e.sharedTransport,2);assert.equal(rpcs,0);assert.ok(Math.abs(applied.parameters.latentGain[0]-1.95)<1e-6);assert.ok(Math.abs(applied.parameters.latentBias[0]-.95)<1e-6);assert.equal(applied.version,42);
});
test('production blocking connect posts both RPC replies before arming wait and cancels a stale start timer',async()=>{
  const posted=[],timers=[],bundle=createSharedTransport(512,2);let closed=0;
  const self={crossOriginIsolated:true,postMessage:m=>posted.push(m)};
  const sandbox={...Runtime,self,Float32Array,Int32Array,SharedArrayBuffer,Atomics,performance,queueMicrotask,setTimeout:cb=>timers.push(cb)};
  const context=vm.createContext(sandbox),source=fs.readFileSync(new URL('../web/inference-worker.js',import.meta.url),'utf8').replace(/^import[^\n]+\n/,'');
  vm.runInContext(source,context);vm.runInContext('session={}; contract={frameSize:512,latentShape:[1,2,1]};',context);
  self.onmessage({data:{id:1,command:'connectShared',args:{shared:bundle,wakeMode:'blocking',port:{start(){},close(){closed++;}}}}});await new Promise(resolve=>setImmediate(resolve));
  assert.equal(posted[0].id,1);assert.equal(posted[0].ok,true);assert.equal(posted[0].result.wakeMode,'atomics-wait');assert.equal(timers.length,0);
  self.onmessage({data:{id:2,command:'beginShared'}});await new Promise(resolve=>setImmediate(resolve));assert.equal(posted[1].id,2);assert.equal(timers.length,1);
  self.onmessage({data:{id:3,command:'disconnect'}});await new Promise(resolve=>setImmediate(resolve));assert.equal(posted[2].id,3);assert.equal(closed,1);assert.equal(Atomics.load(new Int32Array(bundle.control),SHARED_LAYOUT.active),0);
  timers.shift()();assert.equal(timers.length,0);
});
test('a real blocking worker receives seqlock knob updates and resumes queued commands after shared stop',async()=>{
  const bundle=createSharedTransport(512,2),moduleUrl=new URL('../web/stream-engine.mjs',import.meta.url).href,messages=[],waiters=[];
  const worker=new NodeWorker(`const {parentPort,workerData}=require('node:worker_threads');
    (async()=>{const r=await import(workerData.moduleUrl),b=workerData.bundle,c=new Int32Array(b.control);let revision=-1;
      parentPort.on('message',()=>parentPort.postMessage({type:'command-resumed'}));parentPort.postMessage({type:'ready'});
      while(Atomics.load(c,r.SHARED_LAYOUT.active)){const observed=Atomics.load(c,r.SHARED_LAYOUT.wake),u=r.readSharedParameters(b,2,revision);if(u){revision=u.version;parentPort.postMessage({type:'applied',revision,gain:u.parameters.latentGain[0]});}if(Atomics.load(c,r.SHARED_LAYOUT.active))await r.waitSharedSignal(c,observed,'atomics-wait');}
      parentPort.postMessage({type:'stopped'});
    })().catch(e=>parentPort.postMessage({type:'error',message:e.message}));`,{eval:true,workerData:{bundle,moduleUrl}});
  worker.on('message',m=>{messages.push(m);for(const wake of waiters.splice(0))wake();});
  const next=async predicate=>{const until=Date.now()+1500;while(!messages.some(predicate)){if(Date.now()>until)throw Error('blocking worker timeout: '+JSON.stringify(messages));await new Promise(resolve=>{const timer=setTimeout(resolve,20);waiters.push(()=>{clearTimeout(timer);resolve();});});}return messages.find(predicate);};
  try {
    await next(m=>m.type==='ready');await next(m=>m.type==='applied'&&m.revision===2);worker.postMessage({type:'queued-control'});
    Runtime.writeSharedParameters(bundle,normalizeParameters({latentGain:[2,1],latentBias:[.5,0]},2));await next(m=>m.type==='applied'&&m.revision===4&&m.gain===2);
    assert.equal(messages.some(m=>m.type==='command-resumed'),false);cancelSharedTransport(bundle);
    await next(m=>m.type==='stopped');await next(m=>m.type==='command-resumed');assert.equal(messages.some(m=>m.type==='error'),false);
  } finally {cancelSharedTransport(bundle);await worker.terminate();}
});
