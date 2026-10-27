import test from 'node:test';
import assert from 'node:assert/strict';
import { encodeWav } from '../web/audio-utils.mjs';

test('float WAV preserves the neural output above full scale and declares interoperable mono format/fact/data chunks', async () => {
  const input = new Float32Array([-17.125, -1.25, 0, .125, 2.375, 15.5]);
  const bytes = await encodeWav(input, 48000).arrayBuffer(), v = new DataView(bytes);
  const name = at => new TextDecoder().decode(new Uint8Array(bytes, at, 4));
  assert.equal(name(0), 'RIFF'); assert.equal(v.getUint32(4, true), bytes.byteLength - 8); assert.equal(name(8), 'WAVE');
  assert.equal(name(12), 'fmt '); assert.equal(v.getUint16(20, true), 3); assert.equal(v.getUint16(22, true), 1); assert.equal(v.getUint32(24, true), 48000); assert.equal(v.getUint16(34, true), 32);
  assert.equal(name(36), 'fact'); assert.equal(v.getUint32(44, true), input.length); assert.equal(name(48), 'data'); assert.equal(v.getUint32(52, true), input.length * 4);
  assert.deepEqual(input, new Float32Array(bytes.slice(56)));
  assert.throws(() => encodeWav(new Float32Array([NaN])), /finite/);
});

