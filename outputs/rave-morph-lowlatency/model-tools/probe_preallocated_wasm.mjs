// Test the CPU preallocated-output API before attempting state-buffer reuse.
// Uses original ORT Web 1.30.0 bundle/WASM bytes, never onnxruntime-node.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../..');
const vendor=path.join(root,'outputs/rave-morph-lowlatency/web/vendor/ort');
const args=process.argv.slice(2),threads=Number(args[args.indexOf('--threads')+1]||4);
const out=path.join(root,'outputs/rave-morph-lowlatency/research/preallocated-output-wasm.json');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const wasm=fs.readFileSync(path.join(vendor,'ort-wasm-simd-threaded.wasm'));
globalThis.location={origin:'null'};
const ort=await import('data:text/javascript;base64,'+fs.readFileSync(path.join(vendor,'ort.wasm.bundle.min.mjs')).toString('base64'));
ort.env.wasm.numThreads=threads;ort.env.wasm.proxy=false;
ort.env.wasm.wasmBinary=new Uint8Array(wasm);
ort.env.wasm.wasmPaths={mjs:pathToFileURL(path.join(vendor,'ort-wasm-simd-threaded.mjs')).href};
// Standard FLOAT32 [4] -> Mul(x,2) -> Add(...,1) -> [4], opset17/IR8.
const toy=Buffer.from('CAg6kwEKFQoBeAoDdHdvEgZkb3VibGUiA011bAoVCgZkb3VibGUKA29uZRIBeSIDQWRkEh9wcmVhbGxvY2F0ZS1vdXRwdXQtdXBkYXRlLXByb2JlKg8IARABIgQAAABAQgN0d28qDwgBEAEiBAAAgD9CA29uZVoPCgF4EgoKCAgBEgQKAggEYg8KAXkSCgoICAESBAoCCARCBAoAEBE=','base64');
const report={createdAt:new Date().toISOString(),node:process.version,platform:process.platform,architecture:process.arch,runtime:'onnxruntime-web 1.30.0',executionProvider:'wasm',requestedThreads:threads,wasmSha256:hash(wasm),toySha256:hash(toy),toyGraph:'float32[4] y=x*2+1, opset17, IR8',nodeAdapter:'Unmodified bundle bytes via data URL import; original pthread mjs file URL; exact app wasmBinary bytes',runs:[]};
const session=await ort.InferenceSession.create(toy,{executionProviders:['wasm'],executionMode:'sequential',graphOptimizationLevel:'all'});
report.configuredThreads=ort.env.wasm.numThreads;
let dynamicPassed=true,preallocatedPassed=true;
for(const values of [[1,2,-3,.25],[-4,0,10,.5]]){
  const input=new ort.Tensor('float32',Float32Array.from(values),[4]);
  const expected=values.map(x=>x*2+1),dynamic=await session.run({x:input});
  const providedArray=new Float32Array(4).fill(12345),provided=new ort.Tensor('float32',providedArray,[4]);
  const result=await session.run({x:input},{y:provided});
  const equal=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
  const row={input:values,expected,dynamic:Array.from(dynamic.y.data),providedArrayAfter:Array.from(providedArray),providedTensorAfter:Array.from(provided.data),returnedTensorAfter:Array.from(result.y.data),returnedSameTensor:result.y===provided,returnedSameData:result.y.data===providedArray};
  row.dynamicPassed=equal(row.dynamic,expected);
  row.preallocatedPassed=equal(row.providedArrayAfter,expected)&&equal(row.returnedTensorAfter,expected);
  dynamicPassed&&=row.dynamicPassed;preallocatedPassed&&=row.preallocatedPassed;
  report.runs.push(row);input.dispose();dynamic.y.dispose();provided.dispose();
}
report.dynamicPassed=dynamicPassed;report.preallocatedPassed=preallocatedPassed;
report.status=dynamicPassed&&!preallocatedPassed?'blocked_cpu_preallocated_output_not_copied_back':preallocatedPassed?'toy_pass_requires_real_model_parity_and_benchmark':'dynamic_toy_failed';
report.realModelTestPerformed=false;
report.performanceBenchmarkPerformed=false;
report.reason=preallocatedPassed?'Do not integrate until real model output and all states match, then benchmark.':'Ordinary CPU Float32Array remains unchanged and session.run returns that stale Tensor. Stop before testing or integrating ping-pong state buffers.';
await session.release();fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
