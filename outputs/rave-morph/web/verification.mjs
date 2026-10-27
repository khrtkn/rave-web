import { demoSamples, floatStats, encodeWav } from './audio-utils.mjs';

function compare(a, b) {
  if (a.length !== b.length) return { maxAbsError: Infinity, rmsError: Infinity, comparedSamples: 0 };
  let max = 0, sum = 0;
  for (let i = 0; i < a.length; i++) { const e = Math.abs(a[i] - b[i]); max = Math.max(max, e); sum += e * e; }
  return { maxAbsError: max, rmsError: Math.sqrt(sum / a.length), comparedSamples: a.length };
}

export async function verifyRave({ contract, url, backend = 'wasm', threads = 1, onProgress = () => {} }) {
  const worker = new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' });
  let id = 0;
  const pending = new Map();
  worker.onmessage = ({ data }) => { const p = pending.get(data.id); if (!p) return; pending.delete(data.id); clearTimeout(p.timer); data.ok ? p.resolve(data.result) : p.reject(new Error(data.error)); };
  worker.onerror = e => { for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new Error(e.message || '検証Workerが停止しました')); } pending.clear(); };
  const rpc = (command, args = {}, transfer = []) => new Promise((resolve, reject) => { const key = ++id, timer = setTimeout(() => { pending.delete(key); reject(new Error(`検証 ${command} がタイムアウトしました`)); }, 120000); pending.set(key, { resolve, reject, timer }); worker.postMessage({ id: key, command, args }, transfer); });
  const fetchBytes = async target => { const r = await fetch(target); if (!r.ok) throw new Error(`検証データ取得失敗 ${r.status}: ${target}`); return r.arrayBuffer(); };
  try {
    onProgress('実モデルのSHA-256と入出力契約を検証しています…');
    const bytes = await fetchBytes(url);
    const runtime = await rpc('loadModel', { bytes, contract, backend, threads }, [bytes]);
    const neutral = { latentGain: Array(contract.latentDimensions).fill(1), latentBias: Array(contract.latentDimensions).fill(0) };
    const result = { test: 'real_rave_latent_browser', timestamp: new Date().toISOString(), model: contract.id, source: contract.sourceURL, modelSHA256: contract.sha256, runtime, userAgent: navigator.userAgent, sampleRate: contract.sampleRate, frameSize: contract.frameSize, blockDurationMs: contract.blockDurationMs, latentCoordinates: contract.latentCoordinateSystem, morphoEngineUsed: false, cpuPercent: null, gpuPercent: null, physicalRoundtripLatencyMs: null, modelDelaySamples: contract.modelDelaySamples };
    let input;
    const fixturePath = contract.parityFixture || (contract.id === 'water-latent-2048' ? 'parity-fixture.json' : null);
    if (fixturePath) {
      onProgress('ネイティブ推論と、全参照サンプルを照合しています…');
      const base = new URL('./models/', location.href);
      const fixtureURL = new URL(fixturePath, base);
      const fixtureResponse = await fetch(fixtureURL);
      if (!fixtureResponse.ok) throw new Error('ネイティブ比較データが見つかりません');
      const fixture = await fixtureResponse.json();
      if (fixture.modelSha256 !== contract.sha256) throw new Error('ネイティブ比較用モデルのSHA-256が一致しません');
      const inputBytes = await fetchBytes(new URL(fixture.input.file, fixtureURL)), expectedBytes = await fetchBytes(new URL(fixture.expectedOutput.file, fixtureURL));
      const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), n => n.toString(16).padStart(2, '0')).join('');
      if (await hash(inputBytes) !== fixture.input.sha256 || await hash(expectedBytes) !== fixture.expectedOutput.sha256) throw new Error('ネイティブ比較データのSHA-256が一致しません');
      input = new Float32Array(inputBytes);
      const expected = new Float32Array(expectedBytes);
      const nativeRun = await rpc('render', { audio: input, noiseMode: 'zeros', parameters: neutral, seed: 1 });
      const difference = compare(nativeRun.audio, expected);
      const tolerance = { maxAbsError: fixture.tolerance?.maxAbs ?? 0.0002, rmsError: fixture.tolerance?.rms ?? 0.00002 };
      result.nativeParity = { ...difference, tolerance, passed: difference.maxAbsError <= tolerance.maxAbsError && difference.rmsError <= tolerance.rmsError, fixture, outputStats: floatStats(nativeRun.audio) };
    } else {
      input = demoSamples(contract.sampleRate, contract.frameSize * 8 / contract.sampleRate);
      result.nativeParity = { passed: null, note: 'Native reference fixture is supplied for the 2048 variant. Select 2048 for full parity validation.' };
    }
    const exposed = contract.exposedDimensions || Math.min(6, contract.latentDimensions);
    onProgress(`状態のリセットと、${exposed}軸それぞれのgain / biasを照合しています…`);
    const first = await rpc('render', { audio: input, noiseMode: 'zeros', parameters: neutral, seed: 1 });
    const again = await rpc('render', { audio: input, noiseMode: 'zeros', parameters: neutral, seed: 1 });
    result.reset = { ...compare(first.audio, again.audio) };
    result.reset.passed = result.reset.maxAbsError <= 0.000001;
    result.axes = [];
    for (let axis = 0; axis < exposed; axis++) {
      for (const kind of ['latentBias', 'latentGain']) {
        const parameters = structuredClone(neutral); parameters[kind][axis] = kind === 'latentBias' ? .5 : .5;
        const changed = await rpc('render', { audio: input, noiseMode: 'zeros', parameters, seed: 1 });
        const difference = compare(first.audio, changed.audio), stats = floatStats(changed.audio);
        result.axes.push({ axis: axis + 1, control: kind, value: .5, rmsDifference: difference.rmsError, peakDifference: difference.maxAbsError, nonFinite: stats.nonFinite, passed: difference.rmsError > 0.000001 && stats.nonFinite === 0 });
      }
      onProgress(`潜在 ${axis + 1} / ${exposed}: gainとbiasが実際の出力を変更。次の軸を検証中…`);
    }
    onProgress('本物のRAVEで音声を連続変換し、ノブ変更前後の音を生成しています…');
    const demo = demoSamples(contract.sampleRate, 6), preroll = (contract.warmupBlocks || 32) * contract.frameSize;
    const audio = new Float32Array(preroll + demo.length); audio.set(demo, preroll);
    const dryRun = await rpc('render', { audio, parameters: neutral, seed: 20261026 });
    const moved = structuredClone(neutral); moved.latentBias[0] = 1; moved.latentBias[1] = -.75; moved.latentGain[2] = .5;
    const changedRun = await rpc('render', { audio, parameters: moved, seed: 20261026 });
    const neutralAudio = dryRun.audio.slice(preroll), changedAudio = changedRun.audio.slice(preroll);
    result.render = { input: floatStats(demo), neutral: floatStats(neutralAudio), changed: floatStats(changedAudio), difference: compare(neutralAudio, changedAudio), parameters: moved, warmupSamplesTrimmed: preroll, audioSamples: demo.length, seed: 20261026, timingScope: 'Offline render, session.run + all state/audio CPU readback; includes preroll in timing. No Worklet/device latency included.' };
    result.performance = dryRun.metrics;
    result.performance.p95WithinBlockBudget = dryRun.metrics.p95InferenceMs < contract.blockDurationMs;
    onProgress('32-bit float WAVをブラウザで再デコードし、保存する振幅と長さを照合しています…');
    const decoder = new OfflineAudioContext(1, 1, contract.sampleRate);
    const decoded = await decoder.decodeAudioData(await encodeWav(neutralAudio, contract.sampleRate).arrayBuffer());
    result.wavRoundtrip = { ...compare(neutralAudio, decoded.getChannelData(0)), sampleRate: decoded.sampleRate, channels: decoded.numberOfChannels, format: 'IEEE float32 mono WAV with fact chunk; no normalization or clipping' };
    result.wavRoundtrip.passed = result.wavRoundtrip.maxAbsError < 0.000001 && decoded.sampleRate === contract.sampleRate;
    result.passed = result.nativeParity.passed !== false && result.reset.passed && result.axes.every(x => x.passed) && result.render.neutral.nonFinite === 0 && result.render.changed.nonFinite === 0 && result.performance.p95WithinBlockBudget && result.wavRoundtrip.passed;
    return { result, audio: { dry: demo, neutral: neutralAudio, changed: changedAudio, sampleRate: contract.sampleRate } };
  } finally {
    worker.terminate();
    for (const p of pending.values()) clearTimeout(p.timer);
  }
}
