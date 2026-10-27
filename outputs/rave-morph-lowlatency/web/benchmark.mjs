import { RaveStreamEngine as CurrentEngine } from './stream-engine.mjs';
import { RaveStreamEngine as PreviousEngine } from './legacy/stream-engine.mjs';
import { demoSamples, saveBlob } from './audio-utils.mjs';
const $ = id => document.getElementById(id);
let models = [], active = null, cancelled = false, report = null;
function clean(m) { const { inputPreview, outputPreview, inputDeviceSettings, ...rest } = m; return rest; }
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function update() { $('report').textContent = JSON.stringify({ ...report, runs: report.runs.map(({timeline, ...run}) => ({ ...run, timelineSnapshots: timeline.length })) }, null, 2); $('full-report').textContent = JSON.stringify(report, null, 2); }
async function measure(contract, kind, seconds, options) {
  let latest = {}, afterStartup = null, failure = null;
  const timeline = [];
  const Engine = kind === 'old' ? PreviousEngine : CurrentEngine;
  const engine = new Engine({ onMetrics: m => { latest = clean(m); $('current-metrics').textContent = JSON.stringify(latest, null, 2); }, onError: e => { failure = e; } });
  active = engine;
  try {
    $('status').textContent = `${kind === 'old' ? '既存' : '新'}経路 / モデルロード中`;
    await engine.loadModel({ url: new URL(`./models/${contract.url}`, location.href).href, contract, ...options });
    engine.setParameters({ latentGain: 1, latentBias: 0, wet: 1, outputGain: 0 });
    const samples = demoSamples(contract.sampleRate, 12);
    const buffer = new AudioBuffer({ length: samples.length, numberOfChannels: 1, sampleRate: contract.sampleRate }); buffer.copyToChannel(samples, 0);
    if (cancelled) return null;
    await engine.start({ source: 'buffer', buffer, loop: true, seed: 20261026, ...(kind === 'new' ? { wakeMode: options.wakeMode, ...(options.targetDelayMs !== undefined ? { targetDelayMs: options.targetDelayMs } : {}) } : {}) });
    const startWall = performance.now();
    while (!cancelled && performance.now() - startWall < seconds * 1000) {
      if (failure) throw failure;
      await pause(250);
      const elapsed = (performance.now() - startWall) / 1000;
      if (!afterStartup && elapsed >= 2) afterStartup = { ...latest, elapsedSeconds: elapsed };
      timeline.push({ elapsedSeconds: elapsed, ...latest });
      $('status').textContent = `${kind === 'old' ? '既存' : '新'}経路 / ${elapsed.toFixed(1)} / ${seconds} 秒`;
    }
    const last = clean(engine.getMetrics()), duration = (performance.now() - startWall) / 1000;
    await engine.stop();
    const audioDelivered = (last.renderedBlocks || 0) > 0;
    return { path: kind, elapsedSeconds: duration, cancelled, audioDelivered, realtimeAssessment: audioDelivered ? 'Audio delivery started. Consult underrun, skipped samples and block loss for continuity; completion alone is not a gapless pass.' : 'FAILED: no processed audio was delivered before its deadline.', final: last, afterStartup, steadyState: afterStartup ? { observationSeconds: duration - afterStartup.elapsedSeconds, droppedInputBlocks: (last.droppedInputBlocks || 0) - (afterStartup.droppedInputBlocks || 0), droppedOutputBlocks: (last.droppedOutputBlocks || 0) - (afterStartup.droppedOutputBlocks || 0), underrunSamples: (last.underrunSamples || 0) - (afterStartup.underrunSamples || 0), underrunMs: ((last.underrunSamples || 0) - (afterStartup.underrunSamples || 0)) / contract.sampleRate * 1000, skippedOutputSamples: (last.skippedOutputSamples || 0) - (afterStartup.skippedOutputSamples || 0), skippedOutputMs: ((last.skippedOutputSamples || 0) - (afterStartup.skippedOutputSamples || 0)) / contract.sampleRate * 1000, stateResets: (last.stateResets || 0) - (afterStartup.stateResets || 0), lossAccounting: last.lossAccounting || 'old path: no deadline skipping' } : null, timeline };
  } finally { await engine.dispose(); if (active === engine) active = null; }
}
$('run').addEventListener('click', async () => {
  const contract = models.find(m => m.id === $('model').value), seconds = +$('duration').value;
  const value = $('backend').value;
  const options = { wakeMode: $('wake').value, ...( $('target').value !== 'auto' ? { targetDelayMs: +$('target').value } : {} ), backend: ['wasm-2', 'wasm-4'].includes(value) ? 'wasm' : value === 'webgpu-retain' ? 'webgpu' : value, ...(value === 'webgpu-retain' ? { stateMode: 'gpu-retain' } : {}), threads: value === 'wasm-4' ? 4 : value === 'wasm-2' ? 2 : 1 };
  cancelled = false; $('run').disabled = true; $('cancel').disabled = false; $('save').disabled = true; $('message').hidden = true;
  report = { test: 'rave_live_sequential_ab', timestamp: new Date().toISOString(), userAgent: navigator.userAgent, crossOriginIsolated, model: contract.id, modelSHA256: contract.sha256, sampleRate: contract.sampleRate, frameSize: contract.frameSize, blockDurationMs: contract.blockDurationMs, addedModelDelayMs: Number.isFinite(contract.additionalDelaySamples ?? contract.modelDelaySamples) ? (contract.additionalDelaySamples ?? contract.modelDelaySamples) / contract.sampleRate * 1000 : null, declaredDryAlignmentSamples: contract.modelDelaySamples, sourceModelDelaySamples: contract.sourceModelDelaySamples ?? null, physicalRoundtripLatencyMs: null, source: 'same local deterministic 12-second demo loop; output gain 0', ...options, requestedDurationSeconds: seconds, runs: [] }; update();
  try {
    for (const path of $('mode').value === 'both' ? ['old', 'new'] : [$('mode').value]) {
      if (cancelled) break;
      const run = await measure(contract, path, seconds, options); if (run) report.runs.push(run); update();
      await pause(500);
    }
    report.complete = !cancelled; report.audioDelivered = report.runs.length > 0 && report.runs.every(run => run.audioDelivered); report.finished = new Date().toISOString(); update();
    $('status').textContent = cancelled ? '中止 / 取得済みデータを表示' : report.audioDelivered ? '比較完了 / 音声出力あり' : '測定完了 / 処理音声の出力なし';
  } catch (e) { report.error = e.message; update(); $('message').hidden = false; $('message').textContent = e.message; $('status').textContent = '測定失敗'; }
  finally { $('run').disabled = false; $('cancel').disabled = true; $('save').disabled = false; }
});
$('cancel').addEventListener('click', () => { cancelled = true; });
$('backend').addEventListener('change', () => {
  const newOnly = ['wasm-4', 'webgpu-retain'].includes($('backend').value);
  for (const option of $('mode').options) if (option.value !== 'new') option.disabled = newOnly;
  if (newOnly) $('mode').value = 'new'; // The preserved previous engine supports neither 4 threads nor GPU retention.
});
$('save').addEventListener('click', () => saveBlob(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }), 'rave-live-ab.json'));
window.addEventListener('pagehide', () => { cancelled = true; void active?.dispose(); });
try { const r = await fetch('./models/catalog.json'); if (!r.ok) throw Error(`Catalog ${r.status}`); models = (await r.json()).models; for (const model of models) $('model').add(new Option(`${model.name} / ${model.frameSize}`, model.id)); $('status').textContent = '開始できます'; } catch (e) { $('status').textContent = e.message; $('run').disabled = true; }
