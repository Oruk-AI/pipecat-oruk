// Explicit opt-in fixture: real public Hume SDK + native Node WebSocket.
// No listener, provider, credential lookup, dependency installation or TLS claim.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import dns from 'node:dns';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import tls from 'node:tls';
import { createRequire } from 'node:module';

const KEY = 'local-synthetic-gateway-key-not-a-provider-key';
const [sdkRoot, portText, mode] = process.argv.slice(2);
const port = Number(portText);
const violations = [];
const sockets = new Set();
const webSockets = new Set();
const queue = [];
const outcomes = [];
let phase = 'arguments';
let socket;
let notify;
let closing = false;
let connections = 0;
let constructors = 0;
let messages = 0;
let bytes = 0;
let failure;
let rejectFatal;
const fatal = new Promise((_, reject) => { rejectFatal = reject; });
void fatal.catch(() => {});
function fail(code) {
  process.exitCode = 1;
  if (!failure) { failure = code; rejectFatal(new Error(code)); }
}
function refuse(code) { violations.push(code); fail(code); throw new Error(code); }
function checkDestination(host, selectedPort) {
  if (closing || host !== '127.0.0.1' || String(selectedPort) !== portText) refuse('unowned_destination');
}
function deadline(promise, milliseconds, code, observeFatal = true) {
  let timer;
  const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(code)), milliseconds); });
  return Promise.race(observeFatal ? [promise, timeout, fatal] : [promise, timeout]).finally(() => clearTimeout(timer));
}
function digest(file) { return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'); }

// Install guards before loading the SDK. They delegate actual permitted sockets;
// they do not synthesize SDK events, frames, HTTP responses or transport results.
const originalConnect = net.Socket.prototype.connect;
net.Socket.prototype.connect = function (...args) {
  const normalized = Array.isArray(args[0]) ? args[0] : args;
  const first = normalized[0];
  const options = first !== null && typeof first === 'object' ? first : {
    port: first, host: typeof normalized[1] === 'string' ? normalized[1] : undefined,
  };
  if (options.path !== undefined) refuse('socket_path_refused');
  checkDestination(options.host ?? options.hostname, options.port);
  if (++connections > 1) refuse('reconnect_refused');
  sockets.add(this);
  this.once('close', () => sockets.delete(this));
  return originalConnect.apply(this, args);
};
tls.connect = () => refuse('tls_refused');
globalThis.fetch = async () => refuse('fetch_refused');
const originalLookup = dns.lookup;
dns.lookup = function (host, ...args) {
  if (host !== '127.0.0.1') refuse('dns_refused');
  return originalLookup.call(this, host, ...args);
};
const originalPromiseLookup = dns.promises.lookup;
dns.promises.lookup = async function (host, ...args) {
  if (host !== '127.0.0.1') refuse('dns_refused');
  return originalPromiseLookup.call(this, host, ...args);
};
for (const name of ['resolve', 'resolve4', 'resolve6', 'resolveAny', 'reverse']) {
  dns[name] = () => refuse('dns_refused');
  dns.promises[name] = async () => refuse('dns_refused');
}
const NativeWebSocket = globalThis.WebSocket;
if (typeof NativeWebSocket === 'function') {
  globalThis.WebSocket = class GuardedNativeWebSocket extends NativeWebSocket {
    constructor(url, protocols) {
      const parsed = new URL(url);
      checkDestination(parsed.hostname, parsed.port);
      if (parsed.protocol !== 'ws:' || parsed.pathname !== '/v0/evi/chat' || parsed.username || parsed.password || parsed.hash) refuse('websocket_route_refused');
      if (++constructors > 1) refuse('reconnect_refused');
      assert.deepEqual([...parsed.searchParams.keys()].sort(), ['api_key', 'fernSdkLanguage', 'fernSdkVersion']);
      assert.equal(parsed.searchParams.get('api_key'), KEY);
      assert.equal(parsed.searchParams.get('fernSdkLanguage'), 'JavaScript');
      assert.equal(parsed.searchParams.get('fernSdkVersion'), '0.15.17');
      super(url, protocols);
      webSockets.add(this);
      this.addEventListener('close', () => webSockets.delete(this), { once: true });
      this.addEventListener('error', () => fail('native_websocket_error'));
    }
  };
}
process.on('unhandledRejection', () => { violations.push('unhandled_rejection'); fail('unhandled_rejection'); });
process.on('SIGTERM', () => fail('terminated'));
process.on('SIGINT', () => fail('interrupted'));

function receive(message) {
  try {
    const encoded = JSON.stringify(message);
    bytes += Buffer.byteLength(encoded);
    assert.ok(++messages <= 64 && bytes <= 256 * 1024 && queue.length < 32, 'message_budget');
    assert.ok(!encoded.includes(KEY), 'credential_echo');
    assert.ok(message && typeof message.type === 'string', 'sdk_message_shape');
    assert.notEqual(message.type, 'error', 'gateway_error');
    queue.push(message);
    const next = notify; notify = undefined; next?.();
  } catch { fail('received_message_invalid'); }
}
async function until(type) {
  const found = [];
  for (;;) {
    if (!queue.length) await deadline(new Promise(resolve => { notify = resolve; }), 4000, 'message_timeout');
    const item = queue.shift();
    if (!item) continue;
    found.push(item);
    if (item.type === type) return found;
  }
}
function only(records, type) {
  const selected = records.filter(item => item.type === type);
  assert.equal(selected.length, 1, 'one_message_required');
  return selected[0];
}
function wav(data) {
  assert.equal(typeof data, 'string');
  assert.match(data, /^[A-Za-z0-9+/]+={0,2}$/);
  const value = Buffer.from(data, 'base64');
  assert.equal(value.toString('base64'), data);
  assert.equal(value.length, 1964);
  assert.equal(value.toString('ascii', 0, 4), 'RIFF');
  assert.equal(value.readUInt32LE(4), value.length - 8);
  assert.equal(value.toString('ascii', 8, 16), 'WAVEfmt ');
  assert.equal(value.readUInt32LE(16), 16);
  assert.equal(value.readUInt16LE(20), 1);
  assert.equal(value.readUInt16LE(22), 1);
  assert.equal(value.readUInt32LE(24), 48000);
  assert.equal(value.readUInt32LE(28), 96000);
  assert.equal(value.readUInt16LE(32), 2);
  assert.equal(value.readUInt16LE(34), 16);
  assert.equal(value.toString('ascii', 36, 40), 'data');
  assert.equal(value.readUInt32LE(40), 1920);
  for (let offset = 44; offset < value.length; offset += 2) assert.equal(value.readInt16LE(offset), 1);
  return { frames: 960, sample_rate: 48000, channels: 1, sample_width: 2 };
}
function answer(records, replyNumber) {
  const text = only(records, 'assistant_message');
  assert.deepEqual(text.models, {});
  assert.deepEqual(text.message, { role: 'assistant', content: `Synthetic reply ${replyNumber}.` });
  const audio = only(records, 'audio_output');
  assert.equal(audio.id, text.id);
  assert.equal(audio.index, 0);
  assert.match(audio.id, /^[0-9a-f]{32}\.[0-9a-f]{32}$/);
  return { id: audio.id, ...wav(audio.data) };
}

const wall = setTimeout(() => fail('fixture_deadline'), 20000);
try {
  assert.equal(process.versions.node.split('.')[0], '22');
  assert.equal(typeof NativeWebSocket, 'function');
  assert.ok(path.isAbsolute(sdkRoot) && fs.realpathSync(sdkRoot) === sdkRoot);
  assert.ok(Number.isInteger(port) && port > 0 && port <= 65535 && String(port) === portText);
  assert.ok(['normal', 'tool'].includes(mode));
  phase = 'source_pin';
  const pins = {
    'package.json': '3ff2e2e54f88a21ef1605a31964dfe787056c17eac889900d0ae9ef79b3af3df',
    'dist/cjs/api/resources/empathicVoice/resources/chat/client/Client.js': '6175d41a38eb5871db7f7c0321f4240a0085fa2fb54433bfc24c932f94ff0bed',
    'dist/cjs/api/resources/empathicVoice/resources/chat/client/Socket.js': '7021438057dad9518f1288bcd2c7f2eab3c39b7d83dc1f9e86eb79adb7bcf418',
    'dist/cjs/core/websocket/ws.js': '87284eb0ac5f522d46f27896220a7c20b859236305125959d4fd5a782228fd9a',
  };
  for (const [relative, expected] of Object.entries(pins)) assert.equal(digest(path.join(sdkRoot, relative)), expected, 'sdk_source_changed');
  assert.equal(digest(path.resolve(sdkRoot, '../../package-lock.json')), 'cfe911bddabdd14e086f750f0e4bbfbabfec65c153775079ae6d97f536651b6e', 'sdk_lock_changed');
  const require = createRequire(path.join(sdkRoot, 'package.json'));
  assert.equal(require.resolve('hume'), path.join(sdkRoot, 'dist/cjs/index.js'));
  const { HumeClient } = require('hume');
  phase = 'connect';
  const origin = `http://127.0.0.1:${port}`;
  const client = new HumeClient({ apiKey: KEY, environment: {
    base: origin, evi: `ws://127.0.0.1:${port}/v0/evi`,
    tts: `ws://127.0.0.1:${port}/unsupported`, stream: `ws://127.0.0.1:${port}/unsupported`,
  } });
  socket = client.empathicVoice.chat.connect({ reconnectAttempts: 0, debug: false });
  socket.on('message', receive);
  socket.on('error', () => fail('sdk_socket_error'));
  await deadline(socket.waitForOpen(), 4000, 'open_timeout');
  socket.sendSessionSettings({ audio: { encoding: 'linear16', sampleRate: 16000, channels: 1 } });
  const metadata = only(await until('chat_metadata'), 'chat_metadata');
  assert.equal(typeof metadata.chatId, 'string');
  assert.equal(typeof metadata.chatGroupId, 'string');
  assert.notEqual(metadata.chatId, metadata.chatGroupId);
  if (mode === 'normal') {
    for (let n = 1; n <= 2; n++) {
      phase = `typed_${n}`;
      const input = `SDK typed ${n}.`;
      socket.sendUserInput(input);
      const records = await until('assistant_end');
      const primary = only(records, 'user_message');
      assert.equal(primary.fromText, true);
      assert.equal(primary.interim, false);
      assert.deepEqual(primary.message, { role: 'user', content: input });
      assert.deepEqual(primary.models, {});
      assert.deepEqual(primary.time, { begin: 0, end: 0 });
      outcomes.push({ kind: phase, ...answer(records, n) });
    }
    assert.notEqual(outcomes[0].id, outcomes[1].id);
    phase = 'pcm_input';
    const pcm = Buffer.alloc(640);
    for (let offset = 0; offset < pcm.length; offset += 2) pcm.writeInt16LE(2, offset);
    socket.sendAudioInput({ data: pcm.toString('base64') });
    const records = await until('assistant_end');
    const primary = only(records, 'user_message');
    assert.equal(primary.fromText, false);
    assert.deepEqual(primary.message, { role: 'user', content: 'Primary 1.' });
    assert.deepEqual(primary.models, {});
    assert.deepEqual(primary.time, { begin: 0, end: 20 });
    outcomes.push({ kind: phase, ...answer(records, 3) });
    assert.equal(new Set(outcomes.map(row => row.id)).size, 3);
    assert.equal(new Set(outcomes.map(row => row.id.split('.')[0])).size, 1);
  } else {
    phase = 'tool';
    socket.sendUserInput('SDK order status.');
    const records = await until('tool_call');
    assert.equal(records.some(item => item.type === 'assistant_end'), false);
    const call = only(records, 'tool_call');
    assert.equal(call.name, 'lookup_demo_order');
    assert.equal(call.toolType, 'function');
    assert.equal(call.responseRequired, true);
    assert.equal(typeof call.toolCallId, 'string');
    assert.deepEqual(JSON.parse(call.parameters), { order_id: 'DEMO-100' });
    socket.sendToolResponseMessage({ toolCallId: call.toolCallId, content: 'SDK synthetic packed.' });
    outcomes.push({ kind: 'tool_continuation', ...answer(await until('assistant_end'), 2) });
  }
  assert.equal(queue.length, 0, 'unconsumed_messages');
  assert.equal(constructors, 1);
  assert.equal(connections, 1);
} catch { if (!failure) failure = `failed_${phase}`; }
finally {
  phase = failure ? phase : 'close';
  closing = true;
  try {
    socket?.close();
    for (const ws of webSockets) ws.close();
    await deadline(new Promise(resolve => {
      function check() {
        if (!webSockets.size && !sockets.size) resolve();
        else setTimeout(check, 10).unref();
      }
      check();
    }), 2500, 'transport_close_timeout', false);
  } catch { if (!failure) failure = 'transport_close_timeout'; }
  finally {
    // Forced cleanup is failure, never evidence of a graceful SDK close.
    if (sockets.size || webSockets.size) {
      if (!failure) failure = 'transport_not_closed';
      for (const owned of sockets) owned.destroy();
    }
    clearTimeout(wall);
  }
}
const report = {
  version: 1, ok: !failure && !violations.length, mode, sdk: 'hume@0.15.17', node: process.versions.node,
  phase, error_code: failure ?? null, outcomes, messages, response_bytes: bytes,
  websocket_constructors: constructors, socket_connections: connections,
  open_websockets: webSockets.size, open_sockets: sockets.size, violations,
  transport: 'native Node WebSocket over owned plaintext IPv4 loopback',
};
process.stdout.write(JSON.stringify(report) + '\n');
if (!report.ok) process.exitCode = 1;
