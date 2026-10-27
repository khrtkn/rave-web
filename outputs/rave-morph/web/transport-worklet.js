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
    this.previewInput=new Float32Array(64);this.previewOutput=new Float32Array(64);this.previewAt=0;
    this.stats={renderCallbacks:0,capturedBlocks:0,inferredBlocks:0,renderedBlocks:0,droppedInputBlocks:0,droppedOutputBlocks:0,underrunSamples:0,startupSilentSamples:0,transportMs:null,transportMeanMs:null,wetPeak:0,wetRms:0};
    this.delayTotal=0;this.delayCount=0;this.recording=null;
    this.port.onmessage=({data:p})=>{
      if(p.type==='connect'){this.audioPort=p.port;this.audioPort.onmessage=({data})=>this.receive(data);this.audioPort.start();}
      if(p.type==='stop'){this.flushRecord();this.active=false;this.queue=[];this.pending.clear();this.playing=null;this.audioPort?.close();}
      if(p.type==='recordStart')this.recording={left:p.maxSamples,dry:new Float32Array(this.size),wet:new Float32Array(this.size),at:0};
      if(p.type==='recordStop'){this.flushRecord();this.port.postMessage({type:'recordStopped',id:p.id});}
    };
  }
  flushRecord() {
    const r=this.recording;
    if(r?.at){const dry=r.dry.slice(0,r.at),wet=r.wet.slice(0,r.at);this.port.postMessage({type:'record',dry,wet},[dry.buffer,wet.buffer]);r.at=0;}
    this.recording=null;
  }
  receive(p) {
    if(!this.active||!this.pending.delete(p.sequence))return;
    if(p.type==='drop'){this.stats.droppedOutputBlocks++;return;}
    if(p.type!=='audio'||!(p.dry instanceof Float32Array)||p.dry.length!==this.size||!p.dry.every(Number.isFinite)||!Array.isArray(p.wet)||p.wet.length!==2||p.wet.some(c=>!(c instanceof Float32Array)||c.length!==this.size||!c.every(Number.isFinite))) {this.stats.droppedOutputBlocks++;return;}
    this.stats.inferredBlocks++;
    // Expire stale work rather than accumulating seconds of delayed playback.
    if(currentFrame-p.frameStart>this.size*4){this.stats.droppedOutputBlocks++;return;}
    if(this.queue.length>=this.maxQueue){this.queue.shift();this.stats.droppedOutputBlocks++;}
    this.queue.push(p);
  }
  process(inputs,outputs) {
    for(const output of outputs)for(const channel of output)channel.fill(0);
    if(!this.active)return false;
    const n=outputs[0]?.[0]?.length || 128,channels=inputs[0] || [];
    this.stats.renderCallbacks++;
    // One render quantum of headroom only at first playback. Later packets
    // continue sample-exactly; underruns never add an entire model block.
    if(!this.started&&this.queue.length&&this.startupHoldRemaining===null){this.startupHoldRemaining=n;this.stats.startupBufferSamples=n;this.stats.startupBufferMs=n/sampleRate*1000;}
    let sum=0,peak=0,inputPeak=0;
    for(let i=0;i<n;i++) {
      let x=0;for(const c of channels)x+=Number.isFinite(c[i])?c[i]:0;
      if(channels.length)x/=channels.length;
      inputPeak=Math.max(inputPeak,Math.abs(x));
      if(!this.inputAt)this.frameStart=currentFrame+i;
      this.input[this.inputAt++]=x;
      if(this.inputAt===this.size) {
        const sequence=this.sequence++;this.stats.capturedBlocks++;
        if(this.audioPort&&this.pending.size<this.maxPending){this.pending.add(sequence);this.audioPort.postMessage({type:'audio',sequence,frameStart:this.frameStart,audio:this.input},[this.input.buffer]);}
        else this.stats.droppedInputBlocks++;
        this.input=new Float32Array(this.size);this.inputAt=0;
      }
      if(!this.playing&&this.queue.length&&(this.started||this.startupHoldRemaining===0)) {
        while(this.queue.length&&currentFrame+i-this.queue[0].frameStart>this.size*4){this.queue.shift();this.stats.droppedOutputBlocks++;}
        this.playing=this.queue.shift() || null;this.outputAt=0;
        if(this.playing){
          if(this.lastSequence!==null&&this.playing.sequence!==this.lastSequence+1){this.dryRing.fill(0);this.dryAt=0;}
          this.lastSequence=this.playing.sequence;this.started=true;
          this.stats.transportMs=(currentFrame+i-this.playing.frameStart)/sampleRate*1000;
          this.delayTotal+=this.stats.transportMs;this.stats.transportMeanMs=this.delayTotal/++this.delayCount;
        }
      }
      let dry=0,left=0,right=0;
      if(this.playing) {
        const p=this.playing,j=this.outputAt++;
        dry=p.dry[j];left=p.wet[0][j];right=p.wet[1][j];
        if(this.delay){const previous=this.dryRing[this.dryAt];this.dryRing[this.dryAt]=dry;this.dryAt=(this.dryAt+1)%this.delay;dry=previous;}
        if(this.outputAt===this.size){this.playing=null;this.stats.renderedBlocks++;}
      } else if(this.started)this.stats.underrunSamples++;
      else this.stats.startupSilentSamples++;
      outputs[0][0][i]=left;outputs[0][1][i]=right;
      outputs[1][0][i]=dry;outputs[1][1][i]=dry;
      this.previewInput[this.previewAt]=x;this.previewOutput[this.previewAt]=(left+right)/2;this.previewAt=(this.previewAt+1)%64;
      if(this.startupHoldRemaining>0)this.startupHoldRemaining--;
      sum+=(left*left+right*right)/2;peak=Math.max(peak,Math.abs(left),Math.abs(right));
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
      this.port.postMessage({type:'stats',...this.stats,inputPreview,outputPreview,pendingBlocks:this.pending.size,queuedBlocks:this.queue.length,queueSamples:this.queue.length*this.size+(this.playing?this.size-this.outputAt:0)},[inputPreview.buffer,outputPreview.buffer]);
    }
    return true;
  }
}
registerProcessor('rave-stream-transport',RaveTransport);
