import { RaveStreamEngine } from './stream-engine.mjs';
import { demoSamples, encodeWav, saveBlob } from './audio-utils.mjs';
import { verifyRave } from './verification.mjs';

const $ = id => document.getElementById(id);
let catalog, contract, engine, loaded = false, running = false, busy = false, source = 'demo', selectedBuffer = null, preview = null, recordData = null, recording = false, verification = null;
let lastMetrics = {}, verificationAudio = null, lastMetricPaint = -Infinity, lastMetricState = null;
let latentGain = Array(16).fill(1), latentBias = Array(16).fill(0);

function status(text, state = '') { $('status').textContent = text; $('status').dataset.state = state; }
function error(err) { const text = err?.message || String(err); $('message').textContent = text; $('message').hidden = false; status('処理を停止しました', 'error'); console.error(text); }
function clearError() { $('message').hidden = true; }
function clearVerification() { verification = null; verificationAudio = null; $('verification-output').textContent = '未実行'; for (const id of ['download-results', 'download-neutral', 'download-changed']) $(id).disabled = true; }
function controls() {
  $('load-model').disabled = busy || recording || !catalog?.models?.length;
  for (const id of ['model-select', 'variant-select', 'backend']) $(id).disabled = busy || running;
  $('start').disabled = busy || running || !loaded;
  $('stop').disabled = !running; for (const id of ['input-device', 'output-device', 'refresh-devices']) $(id).disabled = busy || running; $('latency-profile').disabled = busy || running || contract?.family !== 'organ';
  $('verify').disabled = busy || running || !loaded;
  $('record').disabled = !running;
  $('save-wet').disabled = !recordData; $('save-dry').disabled = !recordData;
}
function params() { return { latentGain, latentBias, wet: +$('wet').value, outputGain: +$('volume').value }; }
function applyParams() { engine?.setParameters(params()); }
function updateKnob(i) {
  const bias = latentBias[i], gain = latentGain[i];
  $(`bias-value-${i}`).textContent = `${bias >= 0 ? '+' : ''}${bias.toFixed(2)}`;
  $(`gain-value-${i}`).textContent = `${gain.toFixed(2)}×`;
  $(`dial-${i}`).style.setProperty('--angle', `${bias / 3 * 135}deg`);
}
function makeKnobs(count = 6) {
  $('knobs').replaceChildren(); $('knobs').dataset.dimensions = String(count);
  for (let i = 0; i < count; i++) {
    const card = document.createElement('div'); card.className = 'knob-card';
    card.innerHTML = `<label for="bias-${i}">LATENT ${String(i + 1).padStart(2, '0')}</label><div class="knob-dial" aria-hidden="true"><div class="knob-face" id="dial-${i}"></div></div><output class="knob-value" id="bias-value-${i}">+0.00</output><input id="bias-${i}" aria-label="潜在 ${i + 1} オフセット" type="range" min="-3" max="3" step="0.01" value="0"><div class="gain-row"><label for="gain-${i}">GAIN</label><input id="gain-${i}" aria-label="潜在 ${i + 1} ゲイン" type="range" min="0" max="2" step="0.01" value="1"><output class="gain-value" id="gain-value-${i}">1.00×</output></div>`;
    $('knobs').append(card);
    $(`bias-${i}`).addEventListener('input', () => { latentBias[i] = +$(`bias-${i}`).value; updateKnob(i); applyParams(); });
    $(`gain-${i}`).addEventListener('input', () => { latentGain[i] = +$(`gain-${i}`).value; updateKnob(i); applyParams(); });
    const dial = card.querySelector('.knob-dial'); let gesture = null;
    dial.addEventListener('pointerdown', event => { if (event.button !== 0) return; gesture = { y: event.clientY, value: latentBias[i] }; dial.setPointerCapture(event.pointerId); $(`bias-${i}`).focus(); event.preventDefault(); });
    dial.addEventListener('pointermove', event => { if (!gesture) return; latentBias[i] = Math.round(Math.max(-3, Math.min(3, gesture.value + (gesture.y - event.clientY) / 40)) * 100) / 100; $(`bias-${i}`).value = latentBias[i]; updateKnob(i); applyParams(); });
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) dial.addEventListener(type, () => { gesture = null; });
  }
}
function resetKnobs() { latentGain.fill(1); latentBias.fill(0); for (let i = 0; i < $('knobs').children.length; i++) { $(`bias-${i}`).value = 0; $(`gain-${i}`).value = 1; updateKnob(i); } applyParams(); }
function setSource(next) {
  source = next;
  for (const button of document.querySelectorAll('[data-source]')) { const active = button.dataset.source === source; button.classList.toggle('selected', active); button.setAttribute('aria-pressed', String(active)); }
  for (const name of ['demo', 'file', 'microphone']) $(`${name}-options`).hidden = name !== source;
}
async function guarded(action) { if (busy) return; busy = true; controls(); clearError(); try { await action(); } catch (e) { error(e); } finally { busy = false; controls(); } }
function selectedModel() { return catalog.models.find(m => m.id === $('variant-select').value); }
function modelFamily(model) { return model.family || (model.id.startsWith('water') ? 'water' : model.id.split('-')[0]); }
function populateVariants() {
  const family = $('model-select').value;
  $('variant-select').replaceChildren();
  const variants = catalog.models.filter(m => modelFamily(m) === family);
  const shortest = Math.min(...variants.map(m => m.frameSize));
  for (const model of variants) $('variant-select').add(new Option(`${model.frameSize} samples / ${model.blockDurationMs.toFixed(2)} ms${variants.length > 1 ? family === 'organ' ? model.frameSize === shortest ? '（応答優先）' : '（負荷を抑える）' : model.frameSize === shortest ? '（短いブロック）' : '（比較用）' : ''}`, model.id));
}
function backendOptions() { const value = $('backend').value; return { backend: ['wasm-2', 'wasm-4'].includes(value) ? 'wasm' : value === 'webgpu-retain' ? 'webgpu' : value, ...(value === 'webgpu-retain' ? { stateMode: 'gpu-retain' } : {}), threads: value === 'wasm-4' ? 4 : value === 'wasm-2' ? 2 : 1 }; }
function responseProfiles(model) {
  const selected = $('latency-profile').value;
  const profiles = model?.frameSize === 1024 ? [['余裕あり / 40 ms', '40'], ['標準 / 32 ms', '32'], ['最小待機 / 自動', 'auto']] : [['マイク用の余裕 / 48 ms', '48'], ['短い応答 / 29.3 ms', '29.333333333333332'], ['最短固定 / 24 ms', '24'], ['最小待機 / 自動', 'auto']];
  $('latency-profile').replaceChildren(...profiles.map(([name, value]) => new Option(name, value)));
  $('latency-profile').value = profiles.some(([, value]) => value === selected) ? selected : profiles[0][1];
}
async function loadModel() {
  const previousFamily = contract ? modelFamily(contract) : null;
  clearVerification();
  await stop(); loaded = false; contract = selectedModel(); controls(); status('モデルをロード中…');
  if (!contract) throw new Error('モデルが選択されていません。');
  const url = new URL(`./models/${contract.url}`, location.href).href;
  await engine.loadModel({ url, contract, ...backendOptions() });
  loaded = true;
  responseProfiles(contract);
  latentGain = Array(contract.latentDimensions).fill(1); latentBias = Array(contract.latentDimensions).fill(0); makeKnobs(contract.exposedDimensions || Math.min(6, contract.latentDimensions));
  if (previousFamily !== modelFamily(contract)) { $('volume').value = contract.defaultOutputGain ?? .35; $('volume-value').textContent = `${Math.round(+$('volume').value * 100)}%`; }
  $('model-title').textContent = contract.familyName || contract.name;
  $('latent-description').textContent = contract.controlDescription || (contract.encodedLatentDimensions ? `元の${contract.encodedLatentDimensions}次元を保持し、学習済みPCAの先頭${contract.exposedDimensions}方向へ差分を加算` : `学習済み${contract.latentDimensions}座標のうち${contract.exposedDimensions}座標を操作（PCAではありません）`);
  $('model-info').textContent = `${contract.name} / ${contract.author} / ${contract.license} / ${(contract.bytes / 1048576).toFixed(1)} MiB / ${contract.sampleRate} Hz・${contract.frameSize} samples・${contract.states.length}個の状態。学習済み重みを保持し、潜在gain/bias入力を追加。SHA-256: ${contract.sha256}。${contract.id.startsWith('djembe') ? ' 実験モデル：元の出力がフルスケールを超えるため、初期再生音量を4%に設定。録音WAVは音量調整前の32-bit floatです。' : contract.experimental ? ' ブラウザ向け実験モデル。録音WAVは音量調整前の32-bit floatです。' : ''}`;
  $('metric-note').textContent = `処理単位 ${(contract.frameSize / contract.sampleRate * 1000).toFixed(2)} ms。経路遅延はWorkletの入力取得→出力開始。${Number.isFinite(contract.modelDelaySamples) ? `書き出しによる追加ずれは${(contract.modelDelaySamples / contract.sampleRate * 1000).toFixed(2)} ms、Dryに同量を補償。元モデル内部の時間応答は別です。` : Number.isFinite(contract.additionalDelaySamples) ? `書き出しによる追加ずれは${(contract.additionalDelaySamples / contract.sampleRate * 1000).toFixed(2)} ms。元モデルとDryの絶対時間合わせは未測定。` : 'モデル固有の遅延は未測定。'}物理入出力の往復遅延は未測定。処理時間比はCPU / GPU使用率ではありません。`;
  applyParams(); status('ロード完了 / 開始できます');
}
async function decodeFile(file) {
  const context = new AudioContext({ sampleRate: 48000 });
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 48000), 48000);
    const node = offline.createBufferSource(); node.buffer = decoded; node.connect(offline.destination); node.start();
    return await offline.startRendering();
  } finally { await context.close(); }
}
function demoBuffer() { const data = demoSamples(contract?.sampleRate || 48000); const buffer = new AudioBuffer({ numberOfChannels: 1, length: data.length, sampleRate: contract?.sampleRate || 48000 }); buffer.copyToChannel(data, 0); return buffer; }
async function start() {
  if (!loaded) return;
  preview?.close(); preview = null;
  $('preview-dry').textContent = '原音を聴く';
  if (source === 'file' && !selectedBuffer) throw new Error('入力する音声ファイルを選択してください。');
  status('音声を準備中…');
  const responseOptions = { outputDeviceId: $('output-device').value || undefined, ...(contract.family === 'organ' && $('latency-profile').value !== 'auto' ? { targetDelayMs: +$('latency-profile').value } : {}) };
  await engine.start(source === 'microphone' ? { source: 'microphone', deviceId: $('input-device').value || undefined, ...responseOptions } : { source: 'buffer', buffer: source === 'demo' ? demoBuffer() : selectedBuffer, loop: true, ...responseOptions });
  applyParams(); running = true; controls(); status('変換中 / ブラウザ内で実行', 'running');
  if ($('performance-mode').checked) $('runtime-note').textContent = '演奏優先で表示更新を停止しています。計測は継続し、停止後に結果を表示します。';
  if (source === 'microphone') await refreshDevices();
}
async function stop() {
  if (recording) await finishRecording();
  await engine?.stop(); running = false; controls();
  if (loaded) status('停止 / モデル保持');
}
async function refreshDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const selected = $('input-device').value, selectedOutput = $('output-device').value;
  const all = await navigator.mediaDevices.enumerateDevices();
  const devices = all.filter(d => d.kind === 'audioinput');
  $('input-device').replaceChildren(new Option('システムのデフォルト', ''));
  for (const [i, d] of devices.entries()) $('input-device').add(new Option(d.label || `入力 ${i + 1}`, d.deviceId));
  $('input-device').value = selected;
  $('output-device').replaceChildren(new Option('システムのデフォルト', ''));
  for (const [i, d] of all.filter(d => d.kind === 'audiooutput').entries()) $('output-device').add(new Option(d.label || `出力 ${i + 1}`, d.deviceId));
  $('output-device').value = selectedOutput;
}
function metricNumber(value, places = 1) { return Number.isFinite(value) ? value.toFixed(places) : '—'; }
function metric(id, value, unit = '') { $(id).replaceChildren(document.createTextNode(value)); if (unit) { const small = document.createElement('small'); small.textContent = unit; $(id).append(small); } }
function publicMetrics(m) {
  const { inputPreview, outputPreview, inputDeviceSettings, ...view } = m;
  if (inputPreview) view.inputPreviewSamples = inputPreview.length;
  if (outputPreview) view.outputPreviewSamples = outputPreview.length;
  if (inputDeviceSettings) { const { sampleRate, channelCount, latency, echoCancellation, noiseSuppression, autoGainControl } = inputDeviceSettings; view.inputDeviceSettings = { sampleRate, channelCount, latency, echoCancellation, noiseSuppression, autoGainControl }; }
  return view;
}
function metrics(m) {
  lastMetrics = m;
  if (m.recordingLimitReached && recording) { recording = false; void engine.stopRecording().then(data => { recordData = data; $('record').textContent = '● 変換音を録音（最大30秒）'; $('record-state').textContent = '30秒の録音を保存可能'; controls(); }).catch(error); }
  if (m.state === 'running' && $('performance-mode').checked) return;
  // Keep audio/control events independent of layout, canvas and accessibility updates.
  const now = performance.now();
  const shouldPaint = m.state !== lastMetricState || now - lastMetricPaint >= 1000;
  lastMetricState = m.state;
  if (!shouldPaint) return;
  lastMetricPaint = now;
  $('live-metrics').textContent = JSON.stringify(publicMetrics(m), null, 2);
  metric('metric-inference', metricNumber(m.inferenceMeanMs ?? m.workerMeanMs ?? m.meanInferenceMs), 'ms');
  metric('metric-tail', `${metricNumber(m.inferenceP95Ms ?? m.p95InferenceMs)} / ${metricNumber(m.inferenceMaxMs ?? m.workerMaxMs ?? m.maxInferenceMs)}`, 'ms');
  metric('metric-latency', `${metricNumber(m.transportMeanMs ?? m.transportDelayMs ?? m.pathDelayMs)} / ${metricNumber(m.transportP95Ms)}`, 'ms');
  $('runtime-note').textContent = `${m.transportMode === 'shared-ring' ? '共有メモリ / ' + (m.wakeMode === 'atomics-wait' ? 'Atomics直接起床' : m.wakeMode === 'atomics-wait-async' ? 'Atomics非同期起床' : 'MessagePort起床') : m.transportMode || '経路準備中'} · ${m.stateMode === 'gpu-retain' ? 'GPU上に状態を保持' : m.stateMode === 'retain' ? '状態テンソル保持' : '状態テンソルコピー'}${Number.isFinite(m.targetDelayMs) ? ` · 出力期限 ${m.targetDelayMs.toFixed(2)} ms` : ''}${Number.isFinite(m.skippedOutputSamples) ? ` · 遅れた出力のskip ${(m.skippedOutputSamples / (contract?.sampleRate || 48000) * 1000).toFixed(1)} ms` : ''}。p95は直近128ブロック。`;
  const mean = m.inferenceMeanMs ?? m.workerMeanMs ?? m.meanInferenceMs;
  metric('metric-load', metricNumber(mean / ((contract?.frameSize || 2048) / (contract?.sampleRate || 48000) * 1000) * 100), '%');
  metric('metric-drops', `${m.droppedInputBlocks ?? m.inputDrops ?? 0}+${m.droppedOutputBlocks ?? 0} / ${((m.underrunSamples ?? 0) / (contract?.sampleRate || 48000) * 1000).toFixed(1)}`, 'ms');
  drawScope('input-scope', m.inputWaveform ?? m.inputPreview, m.inputPeak ?? m.peak ?? 0, '#aeb7ca');
  drawScope('output-scope', m.outputWaveform ?? m.outputPreview, m.outputPeak ?? m.wetPeak ?? 0, '#86a6ff');
}
function drawScope(id, data, peak, color) {
  const canvas = $(id), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h); ctx.strokeStyle = '#414956'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = 1.3; ctx.beginPath();
  if (data?.length) { for (let i = 0; i < data.length; i++) { const x = i / (data.length - 1) * w, y = h / 2 - Math.max(-1, Math.min(1, data[i])) * h * .42; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); } }
  else { ctx.moveTo(w * .5, h * .5 - peak * h * .4); ctx.lineTo(w * .5, h * .5 + peak * h * .4); }
  ctx.stroke();
}
async function finishRecording() {
  if (!recording) return;
  recordData = await engine.stopRecording(); recording = false; $('record').textContent = '● 変換音を録音（最大30秒）';
  $('record-state').textContent = `${((recordData?.wet?.length || 0) / (recordData?.sampleRate || 48000)).toFixed(1)}秒を保存可能`;
  controls();
}
async function verifyModel() {
  status('実モデルを検証中…'); $('verification-output').textContent = 'モデルを読み込み、同じ入力とノイズで中立・ノブ変更・リセットを照合しています…';
  const checked = await verifyRave({ contract, url: new URL(`./models/${contract.url}`, location.href).href, ...backendOptions(), onProgress: text => { $('verification-output').textContent = text; } });
  verification = checked.result; verificationAudio = checked.audio;
  $('verification-output').textContent = JSON.stringify(verification, null, 2); $('download-results').disabled = false;
  $('download-neutral').disabled = false; $('download-changed').disabled = false;
  status(verification.passed ? '実モデル検証 / 成功' : '実モデル検証 / 要確認', verification.passed ? '' : 'error');
}

makeKnobs();
drawScope('input-scope', null, 0, '#aeb7ca'); drawScope('output-scope', null, 0, '#86a6ff');
engine = new RaveStreamEngine({ onMetrics: metrics, onError: e => { running = false; controls(); error(e); } });
$('performance-mode').addEventListener('change', () => { lastMetricPaint = -Infinity; metrics(lastMetrics); });
$('refresh-devices').addEventListener('click', () => guarded(refreshDevices));
$('load-model').addEventListener('click', () => guarded(loadModel));
for (const id of ['variant-select', 'backend']) $(id).addEventListener('change', () => { loaded = false; clearVerification(); status('設定変更 / モデルをロードしてください'); controls(); });
$('model-select').addEventListener('change', () => { populateVariants(); loaded = false; clearVerification(); status('音色変更 / モデルをロードしてください'); controls(); });
$('start').addEventListener('click', () => guarded(start));
$('stop').addEventListener('click', () => guarded(stop));
$('verify').addEventListener('click', () => guarded(verifyModel));
$('reset-knobs').addEventListener('click', resetKnobs);
for (const key of ['wet', 'volume']) $(key).addEventListener('input', () => { $(`${key}-value`).textContent = `${Math.round(+$(key).value * 100)}%`; applyParams(); });
for (const b of document.querySelectorAll('[data-source]')) b.addEventListener('click', () => guarded(async () => { if (running) await stop(); setSource(b.dataset.source); }));
$('audio-file').addEventListener('change', () => guarded(async () => { const file = $('audio-file').files[0]; if (!file) return; if (file.size > 128 * 1048576) throw new Error('128 MiB以下のファイルを選択してください。'); if (running) await stop(); status('音声をデコード中…'); selectedBuffer = await decodeFile(file); $('file-name').textContent = `${file.name} / ${selectedBuffer.duration.toFixed(1)}秒`; status(loaded ? '開始できます' : 'モデルをロードしてください'); }));
$('preview-dry').addEventListener('click', () => guarded(async () => { if (running) await stop(); if (preview) { await preview.close(); preview = null; $('preview-dry').textContent = '原音を聴く'; return; } preview = new AudioContext({ sampleRate: 48000 }); await preview.resume(); const node = preview.createBufferSource(); node.buffer = demoBuffer(); const gain = preview.createGain(); gain.gain.value = .35; node.connect(gain).connect(preview.destination); node.start(); $('preview-dry').textContent = '原音を停止'; node.onended = () => { preview?.close(); preview = null; $('preview-dry').textContent = '原音を聴く'; }; }));
$('record').addEventListener('click', () => guarded(async () => { if (recording) await finishRecording(); else { await engine.startRecording({ maxSeconds: 30 }); recording = true; recordData = null; $('record').textContent = '■ 録音を終了'; $('record-state').textContent = '録音中…'; controls(); } }));
$('save-wet').addEventListener('click', () => { if (recordData) saveBlob(encodeWav(recordData.wet, recordData.sampleRate), 'rave-morph-wet-raw32f.wav'); });
$('save-dry').addEventListener('click', () => { if (recordData) saveBlob(encodeWav(recordData.dry, recordData.sampleRate), 'rave-morph-dry-raw32f.wav'); });
$('download-results').addEventListener('click', () => saveBlob(new Blob([JSON.stringify({ ...verification, liveMetrics: publicMetrics(lastMetrics) }, null, 2)], { type: 'application/json' }), 'rave-morph-verification.json'));
$('download-neutral').addEventListener('click', () => { if (verificationAudio) saveBlob(encodeWav(verificationAudio.neutral, verificationAudio.sampleRate), 'rave-morph-neutral.wav'); });
$('download-changed').addEventListener('click', () => { if (verificationAudio) saveBlob(encodeWav(verificationAudio.changed, verificationAudio.sampleRate), 'rave-morph-latent-changed.wav'); });
window.addEventListener('pagehide', () => { engine?.dispose(); preview?.close(); });
try {
  const response = await fetch('./models/catalog.json'); if (!response.ok) throw new Error('モデルの準備が必要です。リポジトリーの MODEL_SOURCES.md を参照してください。'); catalog = await response.json();
  if (!catalog.models?.length) throw new Error('モデルが未登録です。MODEL_SOURCES.md の手順で取得・変換してください。');
  const families = new Map(); for (const model of catalog.models) if (!families.has(modelFamily(model))) families.set(modelFamily(model), model.familyName || (model.id.startsWith('water') ? 'Water / IIL · Pond Brain' : model.name));
  for (const [id, name] of families) $('model-select').add(new Option(name, id));
  $('model-select').value = modelFamily(catalog.models.find(m => m.id === catalog.defaultModel) || catalog.models[0]); populateVariants();
  $('variant-select').value = catalog.defaultModel;
  const initialModel = selectedModel(); if (initialModel.family === 'organ') $('backend').value = navigator.gpu ? 'webgpu-retain' : globalThis.crossOriginIsolated ? navigator.hardwareConcurrency >= 4 ? 'wasm-4' : 'wasm-2' : 'wasm'; responseProfiles(initialModel); latentGain = Array(initialModel.latentDimensions).fill(1); latentBias = Array(initialModel.latentDimensions).fill(0); makeKnobs(initialModel.exposedDimensions);
  status('モデルをロードしてください'); controls(); await refreshDevices();
} catch (e) { error(e); controls(); }
