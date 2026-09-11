const el = id => document.getElementById(id);
let pc, channel, context, output, source, sessionId, silence, recordBus, recorder;
let playing = false, finished = false, generation = 0, recording = false;
let interimCount = 0;
let stage = 'Ready to connect', recordedAt = 0;
const receipts = [];
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function log(message) {
  const li = document.createElement('li');
  li.textContent = new Date().toLocaleTimeString() + ' · ' + message;
  el('events').append(li);
}
function status(text) { el('status').textContent = text; log(text); }
function fail(err) { status('Error: ' + err.message); }
function result(row) {
  if (row.type === 'interim') interimCount++;
  if (row.type === 'connected') { status('Connected. Ready to stream.'); el('play').disabled = false; }
  if (row.type === 'interim' || row.type === 'final') {
    el('transcript').textContent = row.text;
    el('transcript').className = '';
  }
  if (row.type === 'final') log('Final transcript received');
  if (row.type === 'phrase' && row.event.type.endsWith('.completed')) {
    const scores = [...row.event.emotions].sort((a, b) => b.score - a.score);
    el('estimate').className = '';
    el('estimate').textContent = scores[0]?.label || 'No estimate';
    el('scores').textContent = scores.map(s => s.label + ' ' + s.score.toFixed(3)).join(' · ');
    log('Phrase estimates received');
  }
  if (row.type === 'complete') {
    finished = true;
    receipts.push(row.result);
    const r = row.result;
    el('receipt').textContent = 'Completed turn ' + receipts.length + ' · ' + r.request_id + ' · ' + r.usage.billable_seconds + ' billed seconds';
    status('Turn complete. Stream another sample or disconnect.');
    log(JSON.stringify(row));
  }
  if (row.type === 'error') status('Service error: ' + row.error);
}
async function until(test, timeout = 20000) {
  const start = performance.now();
  while (!test()) {
    if (performance.now() - start > timeout) throw new Error('The expected service event did not arrive');
    await pause(50);
  }
}
async function connect() {
  el('connect').disabled = true;
  const mine = ++generation;
  status('Connecting WebRTC…');
  context ??= new AudioContext({sampleRate: 48000});
  await context.resume();
  output = context.createMediaStreamDestination();
  output.channelCount = 1;
  // Keep a continuous silence track before and between samples, just as an
  // open microphone does. This allows WebRTC to start before speech arrives.
  silence = context.createConstantSource();
  silence.offset.value = 0;
  silence.connect(output);
  silence.start();
  pc = new RTCPeerConnection({iceServers: []});
  pc.addTrack(output.stream.getAudioTracks()[0], output.stream);
  channel = pc.createDataChannel('pipecat');
  channel.onmessage = event => { if (mine === generation) result(JSON.parse(event.data)); };
  pc.onconnectionstatechange = () => { if (mine === generation) log('WebRTC: ' + pc.connectionState); };
  await pc.setLocalDescription(await pc.createOffer());
  await until(() => pc.iceGatheringState === 'complete', 10000);
  const response = await fetch('/offer', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(pc.localDescription)});
  if (!response.ok) throw new Error(await response.text());
  const answer = await response.json();
  sessionId = answer.session_id;
  await pc.setRemoteDescription({type:answer.type, sdp:answer.sdp});
  el('disconnect').disabled = false;
  await until(() => !el('play').disabled);
}
async function play() {
  if (playing) return;
  playing = true; finished = false;
  el('receipt').textContent = '';
  el('play').disabled = true;
  el('transcript').textContent = 'Listening…';
  el('estimate').textContent = 'Processing this phrase…';
  el('scores').textContent = '';
  status('Streaming the public sample at playback speed…');
  const wav = await fetch('/sample.wav');
  if (!wav.ok) throw new Error('Sample unavailable');
  const buffer = await context.decodeAudioData(await wav.arrayBuffer());
  source = context.createBufferSource(); source.buffer = buffer;
  source.connect(output); source.connect(context.destination);
  if (recordBus) source.connect(recordBus);
  const mine = generation;
  source.onended = () => {
    playing = false;
    if (mine === generation && pc?.connectionState === 'connected') {
      if (!finished) status('Audio sent. Waiting for final results…');
      el('play').disabled = false;
    }
  };
  source.start();
}
async function disconnect() {
  const old = sessionId;
  ++generation;
  try { source?.stop(); } catch {}
  silence?.stop();
  playing = false;
  pc?.close();
  if (!recording && context) { await context.close(); context = undefined; }
  el('play').disabled = true; el('disconnect').disabled = true; el('connect').disabled = false;
  const response = await fetch('/close', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({session_id:old})});
  if (!response.ok) throw new Error('Server did not acknowledge disconnect');
  status('Disconnected. Reconnect to start a fresh session.');
}
el('connect').onclick = () => connect().catch(fail);
el('play').onclick = () => play().catch(fail);
el('disconnect').onclick = () => disconnect().catch(fail);

const canvas = el('recording-canvas'), draw = canvas.getContext('2d');
const logo = new Image(); logo.src = '/logo.png';
function text(value, x, y, size=24, color='#192322', weight=400) {
  draw.fillStyle = color; draw.font = `${weight} ${size}px system-ui, sans-serif`; draw.fillText(value, x, y);
}
function wrap(value, x, y, width, size, lineHeight) {
  draw.font = `400 ${size}px system-ui, sans-serif`;
  let line = '';
  for (const word of value.split(' ')) {
    if (draw.measureText(line + word).width > width && line) { text(line, x, y, size); y += lineHeight; line = ''; }
    line += word + ' ';
  }
  text(line, x, y, size);
}
function render() {
  draw.fillStyle = '#f7f5ef'; draw.fillRect(0, 0, 1440, 900);
  if (logo.complete && logo.naturalWidth) draw.drawImage(logo, 64, 45, 160, 160 * logo.naturalHeight / logo.naturalWidth);
  text('PIPECAT 1.8.1  /  REAL WEBRTC + ORUK API', 685, 82, 20, '#53665b', 500);
  text('Words, with vocal context.', 64, 200, 56, '#192322', 550);
  text(stage, 66, 252, 24, '#235e4c', 500);
  draw.fillStyle='#fffdf8'; draw.fillRect(64,300,774,308); draw.fillRect(858,300,518,308);
  text('LIVE TRANSCRIPT', 90, 340, 17, '#65756b', 600);
  wrap(el('transcript').textContent, 90, 403, 712, 37, 54);
  text('PHRASE ESTIMATE', 885, 340, 17, '#65756b', 600);
  wrap(el('estimate').textContent, 885, 407, 450, 36, 50);
  wrap(el('scores').textContent, 885, 494, 450, 20, 32);
  wrap(el('status').textContent, 64, 660, 1270, 24, 35);
  wrap(el('receipt').textContent, 64, 726, 1290, 18, 28);
  text('Public sample recording • Live inference • No microphone, LLM or TTS', 64, 815, 19, '#65756b');
  text('Expression estimates are not verified feelings.  /  oruk.ai/docs', 64, 850, 19, '#65756b');
  if (recording) text(((performance.now()-recordedAt)/1000).toFixed(1)+'s', 1280, 850, 20, '#65756b');
  requestAnimationFrame(render);
}
render();
async function recordDemo() {
  if (recording || pc?.connectionState==='connected') throw new Error('Disconnect before recording a fresh demonstration');
  receipts.length = 0;
  recording = true; recordedAt = performance.now();
  el('record').disabled = true; canvas.hidden = false;
  context = new AudioContext({sampleRate:48000}); await context.resume();
  recordBus = context.createMediaStreamDestination();
  const recordSilence = context.createConstantSource(); recordSilence.offset.value=0; recordSilence.connect(recordBus); recordSilence.start();
  const stream = canvas.captureStream(24);
  for (const track of recordBus.stream.getAudioTracks()) stream.addTrack(track);
  const chunks = [];
  recorder = new MediaRecorder(stream, {mimeType:'video/webm;codecs=vp9,opus', videoBitsPerSecond:4000000});
  recorder.ondataavailable = event => { if(event.data.size) chunks.push(event.data); };
  const saved = new Promise((resolve,reject) => {
    recorder.onstop = async () => {
      try {
        const response = await fetch('/recording', {method:'POST', headers:{'Content-Type':'video/webm'}, body:new Blob(chunks,{type:'video/webm'})});
        if (!response.ok) throw new Error('Recording could not be saved');
        const data = await response.json(); status('Recording saved: '+data.file); resolve(data.file);
      } catch (err) { reject(err); }
    };
  });
  recorder.start(1000);
  try {
    stage = '1 / Connect the browser'; await connect(); await pause(2200);
    stage = '2 / First utterance · interim text, final text, phrase estimate';
    await play(); await until(()=>finished); await pause(3000);
    stage = '3 / A second utterance on the same connection';
    await play(); await until(()=>finished); await pause(3000);
    stage = '4 / Disconnect during speech · cancel the unfinished turn';
    const before = receipts.length, beforeInterim = interimCount; await play();
    await until(()=>interimCount > beforeInterim,10000); await disconnect();
    el('estimate').textContent='Cancelled before a final estimate';
    if (receipts.length !== before) throw new Error('Cancellation occurred after a final receipt; the intended test did not run');
    el('receipt').textContent = 'Cancelled during playback. No completed-turn receipt was claimed.';
    await pause(3000);
    stage = '5 / Reconnect · a new session, with a new API request';
    await connect(); await pause(800); await play(); await until(()=>finished);
    stage = 'Three completed turns. A cancelled turn. Successful reconnection.';
    await pause(Math.max(2500, 45000 - (performance.now()-recordedAt)));
    await disconnect();
  } catch(err) { stage='Demonstration stopped: '+err.message; fail(err); await pause(2000); }
  finally {
    recorder.stop(); recordSilence.stop(); recording=false; recordBus=undefined;
    for (const track of stream.getTracks()) track.stop();
    await saved; if(context){await context.close();context=undefined;}
    el('record').disabled=false;
  }
}
el('record').onclick=()=>recordDemo().catch(fail);
