// Audio packet transport only; the RAVE graph runs in inference-worker.js.
class RaveTransport extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const p=options.processorOptions || {};
    this.size=p.frameSize; this.maxPending=2; this.maxQueue=2;
    if(!Number.isInteger(this.size)||this.size<128)throw Error('Invalid frameSize');
    this.delay=Number.isInteger(p.dryOffsetSamples)?p.dryOffsetSamples:0;
    this.dryRing=new Float32Array(Math.max(1,this.delay)); this.dryAt=0;
    this.input=new Float32Array(this.size);this.inputAt=0;this.sequence=0;
    this.pending=new Set();this.queue=[];this.playing=null;this.outputAt=0;
    this.active=true;this.started=false;this.startupHoldRemaining=null;this.nextReport=0;this.lastSequence=null;
    this.recoveryFadeLength=32;this.recoveryFadeRemaining=0;this.recoveryFadeArmed=false;
    this.guardQuanta=p.startupGuardQuanta??1;if(!Number.isInteger(this.guardQuanta)||this.guardQuanta<0||this.guardQuanta>2)throw Error('startupGuardQuanta must be 0–2');
    this.fixedDeadline=p.fixedDeadline!==false;this.warmupP95Ms=p.warmupP95Ms??0;this.requestedTargetMs=p.targetDelayMs;this.targetDelaySamples=null;
    this.shared=p.shared?{...p.shared,control:new Int32Array(p.shared.control),inputMeta:new Int32Array(p.shared.inputMeta),outputMeta:new Int32Array(p.shared.outputMeta),inputAudio:new Float32Array(p.shared.inputAudio),outputAudio:new Float32Array(p.shared.outputAudio)}:null;
    this.previewInput=new Float32Array(64);this.previewOutput=new Float32Array(64);this.previewAt=0;
    this.stats={renderCallbacks:0,capturedBlocks:0,receivedOutputBlocks:0,renderedBlocks:0,droppedInputBlocks:0,workletDroppedOutputBlocks:0,droppedOutputBlocks:0,skippedOutputSamples:0,deadlineSkippedOutputSamples:0,underrunSamples:0,startupSilentSamples:0,transportMs:null,transportMeanMs:null,wetPeak:0,wetRms:0,recoveryFadeSamples:this.recoveryFadeLength,recoveryFadeEvents:0,recoveryFadeScope:'32-sample playback-only fade-in after actual underrun; no added buffering. Raw dry/wet recording remains unfaded.'};this.localSkippedOutputSamples=0;
    this.delayTotal=0;this.delayCount=0;this.delayWindow=[];this.recording=null;this.inUnderrun=false;
    const hasPerformance=typeof performance!=='undefined'&&typeof performance.now==='function',hasDate=typeof Date!=='undefined'&&typeof Date.now==='function';
    this.wallNow=hasPerformance?()=>performance.now():hasDate?()=>Date.now():null;
    this.wallStats={workletHasPerformanceNow:hasPerformance,workletHasDateNow:hasDate,wallClockSource:hasPerformance?'performance.now':hasDate?'Date.now':'unavailable',wallClockRegressions:0,callbackGapLastMs:null,callbackGapMeanMs:null,callbackGapMaxMs:0,audioFrameGapMaxSamples:0,nonContiguousAudioFrameCallbacks:0,audioVsWallDriftMs:0,audioVsWallDriftMinMs:0,audioVsWallDriftMaxMs:0,sameTimestampBurstMaxCallbacks:0,sameTimestampBurstMaxSamples:0,sameMillisecondBurstMaxCallbacks:0,sameMillisecondBurstMaxSamples:0,estimatedBatchMaxCallbacks:0,estimatedBatchMaxSamples:0,estimatedBatchMaxMs:0,wallTimingScope:'One clock read per render callback; Date.now is millisecond resolution and may jump. Batch is an observed short-gap audio-frame span, not a measured hardware I/O buffer.'};
    this.wallGapCount=0;this.wallGapTotal=0;
    this.port.onmessage=({data:p})=>{
      if(p.type==='connect'){this.sharedWake=p.sharedWake;this.audioPort=p.port;this.audioPort.onmessage=({data})=>this.receive(data);this.audioPort.start();}
      if(p.type==='stop'){this.flushRecord();this.active=false;for(const p of this.queue)this.release(p);this.release(this.playing);this.queue=[];this.pending.clear();this.playing=null;if(this.shared){const s=this.shared,l=s.layout;Atomics.store(s.control,l.active,0);Atomics.add(s.control,l.wake,1);Atomics.notify(s.control,l.wake);}this.audioPort?.close();}
      if(p.type==='recordStart')this.recording={left:p.maxSamples,dry:new Float32Array(this.size),wet:new Float32Array(this.size),at:0};
      if(p.type==='recordStop'){this.flushRecord();this.port.postMessage({type:'recordStopped',id:p.id});}
    };
  }
  observeWallClock(n){
    if(!this.wallNow)return;let now;
    try{now=this.wallNow();}catch{this.wallNow=null;this.wallStats.wallClockSource='unavailable';return;}
    if(!Number.isFinite(now))return;
    const s=this.wallStats,threshold=Math.min(.5,n/sampleRate*1000/4);
    s.batchGapThresholdMs=threshold;
    if(this.firstWall===undefined){this.firstWall=now;this.firstWallFrame=currentFrame;this.sameWallStart=currentFrame;this.sameMsStart=currentFrame;this.batchStart=currentFrame;this.sameWallCount=this.sameMsCount=this.batchCount=1;}
    else {
      const gap=now-this.previousWall,frames=currentFrame-this.previousWallFrame;
      s.callbackGapLastMs=gap;s.audioFrameGapLastSamples=frames;s.audioFrameGapMaxSamples=Math.max(s.audioFrameGapMaxSamples,frames);
      if(frames!==this.previousQuantum)s.nonContiguousAudioFrameCallbacks++;
      if(gap<0)s.wallClockRegressions++;
      else {this.wallGapTotal+=gap;s.callbackGapMeanMs=this.wallGapTotal/++this.wallGapCount;s.callbackGapMaxMs=Math.max(s.callbackGapMaxMs,gap);s.callbackGapMinMs=Math.min(s.callbackGapMinMs??Infinity,gap);s.wallGapOverAudioMaxMs=Math.max(s.wallGapOverAudioMaxMs??0,gap-frames/sampleRate*1000);}
      if(now===this.previousWall)this.sameWallCount++;else {this.sameWallStart=currentFrame;this.sameWallCount=1;}
      if(Math.floor(now)===Math.floor(this.previousWall))this.sameMsCount++;else {this.sameMsStart=currentFrame;this.sameMsCount=1;}
      if(gap>=0&&gap<=threshold)this.batchCount++;else {this.batchStart=currentFrame;this.batchCount=1;}
    }
    s.wallElapsedMs=now-this.firstWall;s.audioFrameElapsedMs=(currentFrame-this.firstWallFrame)/sampleRate*1000;
    s.audioVsWallDriftMs=s.audioFrameElapsedMs-s.wallElapsedMs;s.audioVsWallDriftMinMs=Math.min(s.audioVsWallDriftMinMs,s.audioVsWallDriftMs);s.audioVsWallDriftMaxMs=Math.max(s.audioVsWallDriftMaxMs,s.audioVsWallDriftMs);
    s.sameTimestampBurstMaxCallbacks=Math.max(s.sameTimestampBurstMaxCallbacks,this.sameWallCount);s.sameTimestampBurstMaxSamples=Math.max(s.sameTimestampBurstMaxSamples,currentFrame-this.sameWallStart+n);
    s.sameMillisecondBurstMaxCallbacks=Math.max(s.sameMillisecondBurstMaxCallbacks,this.sameMsCount);s.sameMillisecondBurstMaxSamples=Math.max(s.sameMillisecondBurstMaxSamples,currentFrame-this.sameMsStart+n);
    s.estimatedBatchMaxCallbacks=Math.max(s.estimatedBatchMaxCallbacks,this.batchCount);s.estimatedBatchMaxSamples=Math.max(s.estimatedBatchMaxSamples,currentFrame-this.batchStart+n);s.estimatedBatchMaxMs=s.estimatedBatchMaxSamples/sampleRate*1000;
    this.previousWall=now;this.previousWallFrame=currentFrame;this.previousQuantum=n;
  }
  release(p){if(p?.sharedSlot!==undefined)Atomics.store(this.shared.outputMeta,p.sharedSlot*this.shared.layout.stride+this.shared.layout.status,this.shared.layout.empty);}
  maxPacketAge(){return Math.max(this.size*4,(this.targetDelaySamples??0)+this.size*2);}
  updateDropTotals(){
    const workerDrops=this.shared?Atomics.load(this.shared.control,this.shared.layout.outputDrops):0;
    this.stats.workerDroppedOutputBlocks=workerDrops;this.stats.droppedOutputBlocks=this.stats.workletDroppedOutputBlocks+workerDrops;
    this.stats.workerSkippedOutputSamples=workerDrops*this.size;this.stats.skippedOutputSamples=this.localSkippedOutputSamples+workerDrops*this.size;
  }
  discard(packet,valid=true){this.release(packet);this.stats.workletDroppedOutputBlocks++;if(valid)this.localSkippedOutputSamples+=this.size;this.updateDropTotals();}
  publishInput(sequence,frameStart) {
    const s=this.shared,l=s.layout;
    for(let slot=0;slot<l.slots;slot++){
      const at=slot*l.stride;if(Atomics.compareExchange(s.inputMeta,at+l.status,l.empty,l.writing)!==l.empty)continue;
      s.inputAudio.set(this.input,slot*this.size);s.inputMeta[at+l.sequence]=sequence;
      s.inputMeta[at+l.startLo]=frameStart>>>0;s.inputMeta[at+l.startHi]=Math.floor(frameStart/4294967296);
      const end=frameStart+this.size;s.inputMeta[at+l.endLo]=end>>>0;s.inputMeta[at+l.endHi]=Math.floor(end/4294967296);
      Atomics.store(s.inputMeta,at+l.status,l.ready);Atomics.add(s.control,l.wake,1);Atomics.notify(s.control,l.wake,1);
      if(this.sharedWake)this.audioPort?.postMessage({type:'wake'});return true;
    }
    Atomics.add(s.control,l.inputDrops,1);return false;
  }
  pollShared() {
    const s=this.shared,l=s.layout;let chosen=-1,sequence=Infinity;
    for(let slot=0;slot<l.slots;slot++){const at=slot*l.stride;if(Atomics.load(s.outputMeta,at+l.status)===l.ready&&s.outputMeta[at+l.sequence]<sequence){chosen=slot;sequence=s.outputMeta[at+l.sequence];}}
    if(chosen<0)return;const at=chosen*l.stride;
    if(Atomics.compareExchange(s.outputMeta,at+l.status,l.ready,l.reading)!==l.ready)return;
    const frame=(lo,hi)=>(s.outputMeta[at+hi]>>>0)*4294967296+(s.outputMeta[at+lo]>>>0),start=chosen*this.size*3;
    const packet={type:'audio',sequence:s.outputMeta[at+l.sequence],frameStart:frame(l.startLo,l.startHi),captureEnd:frame(l.endLo,l.endHi),publishedFrame:frame(l.publishLo,l.publishHi),processingStartFrame:frame(l.runLo,l.runHi),sharedSlot:chosen,dry:s.outputAudio.subarray(start,start+this.size),wet:[s.outputAudio.subarray(start+this.size,start+this.size*2),s.outputAudio.subarray(start+this.size*2,start+this.size*3)]};
    if(packet.wet.some(c=>!c.every(Number.isFinite))){this.discard(packet,false);return false;}
    this.stats.receivedOutputBlocks++;this.queue.push(packet);return true;
  }
  selectPacket(now){
    // Bounded drain: a fully late slot must not delay an already-ready newer
    // slot by another render quantum. Never loop on concurrently-produced data.
    for(let attempt=0;attempt<this.maxQueue+(this.shared?this.shared.layout.slots:0);attempt++){
      if(!this.queue.length){if(!this.shared)return;this.pollShared();if(!this.queue.length)continue;}
      const packet=this.queue[0];if(this.fixedDeadline&&now<packet.frameStart+this.targetDelaySamples)return;
      this.queue.shift();
      const expired=now-packet.frameStart>this.maxPacketAge();
      const skipped=this.fixedDeadline?Math.min(this.size,Math.max(0,now-packet.frameStart-this.targetDelaySamples)):0;
      if(this.lastSequence!==null&&packet.sequence!==this.lastSequence+1){this.dryRing.fill(0);this.dryAt=0;}
      if(this.delay)for(let j=0;j<skipped;j++){this.dryRing[this.dryAt]=packet.dry[j];this.dryAt=(this.dryAt+1)%this.delay;}
      this.lastSequence=packet.sequence; // Includes wholly skipped, known dry history.
      this.localSkippedOutputSamples+=skipped;this.stats.deadlineSkippedOutputSamples+=skipped;
      if(expired||skipped===this.size){this.release(packet);this.stats.workletDroppedOutputBlocks++;if(skipped<this.size)this.localSkippedOutputSamples+=this.size-skipped;this.updateDropTotals();continue;}
      this.updateDropTotals();this.playing=packet;this.outputAt=skipped;
      if(!this.started)this.startedAtFrame=now;this.started=true;
      this.stats.rawPacketAgeMs=(now-packet.frameStart)/sampleRate*1000;
      this.stats.transportMs=(now-packet.frameStart-skipped)/sampleRate*1000;
      this.delayTotal+=this.stats.transportMs;this.stats.transportMeanMs=this.delayTotal/++this.delayCount;
      this.stats.transportMinMs=Math.min(this.stats.transportMinMs??Infinity,this.stats.transportMs);this.stats.transportMaxMs=Math.max(this.stats.transportMaxMs??0,this.stats.transportMs);this.delayWindow.push(this.stats.transportMs);if(this.delayWindow.length>128)this.delayWindow.shift();
      if(packet.publishedFrame!==undefined){this.stats.workerToRenderMs=(now-packet.publishedFrame)/sampleRate*1000;this.stats.workerWakeWaitMs=Math.max(0,(packet.processingStartFrame-packet.captureEnd)/sampleRate*1000);}
      return;
    }
  }
  flushRecord() {
    const r=this.recording;
    if(r?.at){const dry=r.dry.slice(0,r.at),wet=r.wet.slice(0,r.at);this.port.postMessage({type:'record',dry,wet},[dry.buffer,wet.buffer]);r.at=0;}
    this.recording=null;
  }
  receive(p) {
    if(!this.active||!this.pending.delete(p.sequence))return;
    if(p.type==='drop'){this.discard(null);return;}
    if(p.type!=='audio'||!(p.dry instanceof Float32Array)||p.dry.length!==this.size||!p.dry.every(Number.isFinite)||!Array.isArray(p.wet)||p.wet.length!==2||p.wet.some(c=>!(c instanceof Float32Array)||c.length!==this.size||!c.every(Number.isFinite))) {this.discard(null,false);return;}
    this.stats.receivedOutputBlocks++;
    if(this.queue.length>=this.maxQueue)this.discard(this.queue.shift());
    this.queue.push(p);
  }
  process(inputs,outputs) {
    for(const output of outputs)for(const channel of output)channel.fill(0);
    if(!this.active)return false;
    const n=outputs[0]?.[0]?.length || 128,channels=inputs[0] || [];
    this.observeWallClock(n);
    this.stats.renderCallbacks++;
    if(this.fixedDeadline&&this.targetDelaySamples===null){
      this.targetDelaySamples=this.requestedTargetMs!==undefined?Math.ceil(this.requestedTargetMs*sampleRate/1000/n)*n:this.size+Math.ceil(this.warmupP95Ms*sampleRate/1000/n)*n+this.guardQuanta*n;
      this.stats.targetDelayMs=this.targetDelaySamples/sampleRate*1000;this.stats.configuredGuardSamples=n*this.guardQuanta;this.stats.deadlinePolicy='fixed render timeline; late output samples are skipped, never hidden';
    }
    if(this.shared){const s=this.shared,l=s.layout;Atomics.add(s.control,l.clockVersion,1);Atomics.store(s.control,l.clockLo,currentFrame>>>0);Atomics.store(s.control,l.clockHi,Math.floor(currentFrame/4294967296));Atomics.add(s.control,l.clockVersion,1);if(!this.playing&&!this.queue.length)for(let attempt=0;attempt<l.slots&&!this.queue.length;attempt++)this.pollShared();}
    // One render quantum of headroom only at first playback. Later packets
    // continue sample-exactly; underruns never add an entire model block.
    if(!this.started&&this.queue.length&&this.startupHoldRemaining===null){this.startupHoldRemaining=this.fixedDeadline?0:n*this.guardQuanta;this.stats.startupBufferSamples=this.startupHoldRemaining;this.stats.startupBufferMs=this.startupHoldRemaining/sampleRate*1000;}
    let sum=0,peak=0,inputPeak=0;
    for(let i=0;i<n;i++) {
      let x=0;for(const c of channels)x+=Number.isFinite(c[i])?c[i]:0;
      if(channels.length)x/=channels.length;
      inputPeak=Math.max(inputPeak,Math.abs(x));
      if(!this.inputAt)this.frameStart=currentFrame+i;
      this.input[this.inputAt++]=x;
      if(this.inputAt===this.size) {
        const sequence=this.sequence++;this.stats.capturedBlocks++;
        if(this.shared){if(!this.publishInput(sequence,this.frameStart))this.stats.droppedInputBlocks++;}
        else if(this.audioPort&&this.pending.size<this.maxPending){this.pending.add(sequence);this.audioPort.postMessage({type:'audio',sequence,frameStart:this.frameStart,audio:this.input},[this.input.buffer]);}
        else this.stats.droppedInputBlocks++;
        if(!this.shared)this.input=new Float32Array(this.size);this.inputAt=0;
      }
      if(!this.playing&&this.queue.length&&(this.started||this.startupHoldRemaining===0))this.selectPacket(currentFrame+i);
      let dry=0,left=0,right=0,playbackGain=1;
      if(this.playing) {
        const p=this.playing,j=this.outputAt++;
        dry=p.dry[j];left=p.wet[0][j];right=p.wet[1][j];
        if(this.delay){const previous=this.dryRing[this.dryAt];this.dryRing[this.dryAt]=dry;this.dryAt=(this.dryAt+1)%this.delay;dry=previous;}
        // Only an actually rendered gap arms recovery. Fade real resumed
        // samples in place, without holding them or changing the timeline.
        if(this.recoveryFadeArmed){this.recoveryFadeRemaining=this.recoveryFadeLength;this.recoveryFadeArmed=false;this.stats.recoveryFadeEvents++;}
        if(this.recoveryFadeRemaining>0){playbackGain=(this.recoveryFadeLength-this.recoveryFadeRemaining+1)/this.recoveryFadeLength;this.recoveryFadeRemaining--;}
        this.inUnderrun=false;if(this.outputAt===this.size){this.release(this.playing);this.playing=null;this.stats.renderedBlocks++;}
      } else if(this.started){this.recoveryFadeArmed=true;this.recoveryFadeRemaining=0;this.stats.underrunSamples++;const phase=currentFrame+i-this.startedAtFrame<sampleRate*2?'startup':'steady';this.stats[phase+'UnderrunSamples']=(this.stats[phase+'UnderrunSamples']??0)+1;if(!this.inUnderrun){this.stats.underrunEvents=(this.stats.underrunEvents??0)+1;this.stats[phase+'UnderrunEvents']=(this.stats[phase+'UnderrunEvents']??0)+1;}this.inUnderrun=true;}
      else this.stats.startupSilentSamples++;
      const playbackLeft=left*playbackGain,playbackRight=right*playbackGain,playbackDry=dry*playbackGain;
      outputs[0][0][i]=playbackLeft;outputs[0][1][i]=playbackRight;
      outputs[1][0][i]=playbackDry;outputs[1][1][i]=playbackDry;
      this.previewInput[this.previewAt]=x;this.previewOutput[this.previewAt]=(playbackLeft+playbackRight)/2;this.previewAt=(this.previewAt+1)%64;
      if(this.startupHoldRemaining>0)this.startupHoldRemaining--;
      sum+=(playbackLeft*playbackLeft+playbackRight*playbackRight)/2;peak=Math.max(peak,Math.abs(playbackLeft),Math.abs(playbackRight));
      const r=this.recording;
      if(r){r.dry[r.at]=dry;r.wet[r.at]=(left+right)/2;r.at++;r.left--;
        if(r.at===this.size){this.port.postMessage({type:'record',dry:r.dry,wet:r.wet},[r.dry.buffer,r.wet.buffer]);r.dry=new Float32Array(this.size);r.wet=new Float32Array(this.size);r.at=0;}
        if(r.left<=0){this.flushRecord();this.port.postMessage({type:'recordLimit'});}
      }
    }
    this.stats.wetPeak=peak;this.stats.outputPeak=peak;this.stats.inputPeak=inputPeak;this.stats.wetRms=Math.sqrt(sum/n);this.stats.transportDelayMs=this.stats.transportMs;
    if(currentFrame>=this.nextReport){
      this.nextReport=currentFrame+sampleRate/4;
      const inputPreview=new Float32Array(64),outputPreview=new Float32Array(64);
      for(let i=0;i<64;i++){inputPreview[i]=this.previewInput[(this.previewAt+i)%64];outputPreview[i]=this.previewOutput[(this.previewAt+i)%64];}
      const sorted=[...this.delayWindow].sort((a,b)=>a-b);let pendingBlocks=this.pending.size,queuedBlocks=this.queue.length;
      if(this.shared){const s=this.shared,l=s.layout;pendingBlocks=0;for(let slot=0;slot<l.slots;slot++){if(Atomics.load(s.inputMeta,slot*l.stride+l.status)!==l.empty)pendingBlocks++;if(Atomics.load(s.outputMeta,slot*l.stride+l.status)===l.ready)queuedBlocks++;}this.stats.droppedInputBlocks=Atomics.load(s.control,l.inputDrops);}
      this.updateDropTotals();
      this.port.postMessage({type:'stats',...this.stats,...this.wallStats,inputPreview,outputPreview,pendingBlocks,queuedBlocks,queueSamples:queuedBlocks*this.size+(this.playing?this.size-this.outputAt:0),contextFrame:currentFrame,renderTimeSeconds:currentFrame/sampleRate,transportP95Ms:sorted.length?sorted[Math.ceil(sorted.length*.95)-1]:null,transportP95WindowBlocks:sorted.length,steadyStateAfterSeconds:2,lossAccounting:'underrunSamples and skippedOutputSamples can describe the same missed timeline interval; do not add them as unique loss',transportTimingScope:this.shared?'Effective rendered sample age; rawPacketAgeMs also reported. Worker wake/publication timing quantized to one render quantum':'Effective rendered sample age; rawPacketAgeMs also reported. MessageChannel transport'},[inputPreview.buffer,outputPreview.buffer]);
    }
    return true;
  }
}
registerProcessor('rave-stream-transport',RaveTransport);
