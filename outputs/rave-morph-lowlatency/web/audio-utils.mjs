export function demoSamples(sampleRate = 48000, seconds = 8) {
  const data = new Float32Array(Math.round(sampleRate * seconds));
  let seed = 20261026;
  const random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 4294967296 * 2 - 1; };
  const notes = [130.8128, 164.8138, 195.9977, 146.8324, 174.6141, 220, 164.8138, 195.9977];
  for (let i = 0; i < data.length; i++) {
    const t = i / sampleRate;
    const beat = t % .5;
    const freq = notes[Math.floor(t) % notes.length];
    const env = Math.min(1, beat / .008) * Math.exp(-beat * 8);
    const harmonic = Math.sin(2 * Math.PI * freq * t) + .28 * Math.sin(2 * Math.PI * freq * 2 * t) + .14 * Math.sin(2 * Math.PI * freq * 3 * t);
    const noise = random() * Math.exp(-beat * 65);
    const tailFade = Math.min(1, t / .02, (seconds - t) / .03);
    data[i] = (.21 * harmonic * env + .13 * noise) * tailFade;
  }
  return data;
}

export function encodeWav(samples, sampleRate = 48000) {
  if (!Number.isSafeInteger(sampleRate) || sampleRate < 8000 || sampleRate > 192000 || !samples.every(Number.isFinite)) throw new Error('WAV requires finite audio and a valid sample rate');
  // IEEE float32 WAV retains real model amplitudes above 1 without PCM clipping.
  const buffer = new ArrayBuffer(56 + samples.length * 4);
  const view = new DataView(buffer);
  const str = (at, value) => { for (let i = 0; i < value.length; i++) view.setUint8(at + i, value.charCodeAt(i)); };
  str(0, 'RIFF'); view.setUint32(4, 48 + samples.length * 4, true); str(8, 'WAVE'); str(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 3, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 4, true); view.setUint16(32, 4, true); view.setUint16(34, 32, true);
  str(36, 'fact'); view.setUint32(40, 4, true); view.setUint32(44, samples.length, true);
  str(48, 'data'); view.setUint32(52, samples.length * 4, true);
  for (let i = 0; i < samples.length; i++) view.setFloat32(56 + 4 * i, samples[i], true);
  return new Blob([buffer], { type: 'audio/wav' });
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function floatStats(samples) {
  let square = 0, peak = 0, nonFinite = 0;
  for (const x of samples) { if (!Number.isFinite(x)) { nonFinite++; continue; } square += x * x; peak = Math.max(peak, Math.abs(x)); }
  return { samples: samples.length, rms: Math.sqrt(square / (samples.length || 1)), peak, nonFinite };
}
