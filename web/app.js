import { extractFeatures, GazePrediction, trainModels } from "./gaze-core.mjs";
import { getPrebuiltProfile, PREBUILT_PROFILES } from "./prebuilt-profiles.mjs";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const MEDIAPIPE_VERSION = "0.10.35";
const MEDIAPIPE_MODULE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
const MEDIAPIPE_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const PROFILE_KEY = "visioals.web.profile.v2";
const INTERACTIONS_KEY = "visioals.web.interactions.v2";
const MODE_KEY = "visioals.web.tracking-mode";
const SETUP_KEY = "visioals.web.first-run-complete";
const DWELL_SECONDS = 1.2;
const CALIBRATION_SECONDS = 14;
const CALIBRATION_PATH = [
  [0.50, 0.50, 0.00], [0.08, 0.08, 0.08], [0.50, 0.08, 0.15],
  [0.92, 0.08, 0.23], [0.92, 0.50, 0.30], [0.92, 0.92, 0.38],
  [0.50, 0.92, 0.45], [0.08, 0.92, 0.53], [0.08, 0.50, 0.60],
  [0.08, 0.08, 0.68], [0.92, 0.92, 0.78], [0.92, 0.08, 0.86],
  [0.08, 0.92, 0.94], [0.50, 0.50, 1.00],
];

const el = {
  app: $("#app"), main: $("#main-screen"), waiting: $("#waiting-screen"), calibration: $("#calibration-screen"),
  tracking: $("#tracking-screen"), closed: $("#closed-screen"), modeLabel: $("#mode-label"), calibrationHint: $("#calibration-hint"),
  calibrationDot: $("#calibration-dot"), calibrationProgress: $("#calibration-progress"), question: $("#question-text"), status: $("#status-text"),
  recording: $("#recording-indicator"), responses: $$(".response"), none: $("#none-response"), modelPoints: $$(".point"),
  camera: $("#camera-video"), setup: $("#setup-overlay"), setupWelcome: $("#setup-page-welcome"), setupCamera: $("#setup-page-camera"),
  setupNext: $("#setup-next"), setupBack: $("#setup-back"), setupQuit: $("#setup-quit"), setupVideo: $("#setup-camera-video"),
  setupFrame: $("#setup-camera-frame"), setupStatus: $("#setup-camera-status"), loading: $("#loading-overlay"), modelProgress: $("#model-progress"),
  modelStatus: $("#model-status"), studio: $("#studio-overlay"), studioName: $("#studio-name"), studioPicker: $("#studio-picker"),
  studioFiles: $("#studio-files"), studioRecord: $("#studio-record"), studioSamples: $("#studio-samples"),
  studioSourceStatus: $("#studio-source-status"), studioSegments: $("#studio-segments"), studioProgress: $("#studio-progress-value"),
  studioStatus: $("#studio-status"), studioSummary: $("#studio-summary"), studioGenerate: $("#studio-generate"), studioUse: $("#studio-use"),
  studioCancel: $("#studio-cancel"), preferences: $("#preferences-overlay"), preferencesPatient: $("#preferences-patient"),
  preferencesMeta: $("#preferences-meta"), preferencesRules: $("#preferences-rules"), clearPreferences: $("#clear-preferences"),
  closePreferences: $("#close-preferences"), message: $("#message-overlay"), messageText: $("#message-text"), messageOk: $("#message-ok"),
};

const state = {
  phase: "waiting", mode: localStorage.getItem(MODE_KEY) || "eye", cameraStream: null, landmarker: null, cameraReady: false,
  lastVideoTime: -1, frameRequest: 0, features: null, calibrationStarted: 0, calibrationSamples: [], models: null,
  modelPositions: { lr: null, poly: null, svr: null, knn: null }, ensemble: null, prediction: new GazePrediction(), lastFrameAt: performance.now(), faceLostAt: 0,
  responses: [], context: null, history: [], rejected: [], rejectionRound: 0, dwellTarget: null, dwellSeconds: 0,
  selectionLocked: false, recording: false, recordingStarting: false, questionRecorder: null, questionChunks: [], questionMicStream: null, speechUtterance: null,
  profile: loadJSON(PROFILE_KEY, null), interactions: loadJSON(INTERACTIONS_KEY, []), importedSamples: [], importedMedia: [],
  studioRecording: false, mediaRecorder: null, mediaChunks: [], micStream: null, audioPlayer: null,
  closed: false, sessionStarted: performance.now(),
};

function loadJSON(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }
function saveJSON(key, value) { localStorage.setItem(key, JSON.stringify(value)); }
function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y) || 1e-7; }

async function api(path, payload, timeoutMs = 45000) {
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`/api/${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The server request failed.");
    return data;
  } catch (error) { if (error.name === "AbortError") throw new Error("The server took too long to respond."); throw error; }
  finally { clearTimeout(timeout); }
}

function showMessage(text) { el.messageText.textContent = text; el.message.hidden = false; el.messageOk.focus(); }
function closeMessage() { el.message.hidden = true; el.main.focus(); }
function recordingMimeType(){
  const candidates=["audio/webm;codecs=opus","audio/mp4","audio/ogg;codecs=opus"];
  return candidates.find(type=>MediaRecorder.isTypeSupported?.(type))||"";
}
function createAudioRecorder(stream){const mimeType=recordingMimeType();return mimeType?new MediaRecorder(stream,{mimeType}):new MediaRecorder(stream);}
function recordingExtension(mimeType){const type=String(mimeType||"").split(";",1)[0].toLowerCase();return {"audio/webm":"webm","audio/mp4":"m4a","audio/ogg":"ogg","audio/opus":"opus","audio/wav":"wav"}[type]||"webm";}
function recordedAudioFile(chunks,recorder,prefix){const type=recorder?.mimeType||"audio/webm";const blob=new Blob(chunks,{type});return new File([blob],`${prefix}_${Date.now()}.${recordingExtension(type)}`,{type});}

function setPhase(phase) {
  state.phase = phase;
  el.waiting.hidden = phase !== "waiting"; el.calibration.hidden = phase !== "calibration"; el.tracking.hidden = phase !== "tracking";
  const showOptions = phase === "tracking";
  [...el.responses, el.none].forEach((item) => { item.hidden = !showOptions; });
  if (!showOptions) $("#model-points").classList.remove("visible");
}

function setStatus(text) { el.status.textContent = text || ""; }
function updateModeLabel() {
  const label = state.mode === "eye" ? "Eye Tracking" : "Head Tracking";
  el.modeLabel.innerHTML = `Mode: ${label}&nbsp;&nbsp;•&nbsp;&nbsp;Press M to switch`;
  el.calibrationHint.textContent = state.mode === "eye" ? "Follow the dot with your eyes" : "Follow the dot with your head";
}

function resetDwell() {
  if (state.dwellTarget) { state.dwellTarget.classList.remove("dwelling"); state.dwellTarget.style.setProperty("--dwell", "0"); }
  state.dwellTarget = null; state.dwellSeconds = 0;
}

function resetCalibration(status = "") {
  state.models = null; state.calibrationSamples = []; state.modelPositions = { lr:null, poly:null, svr:null, knn:null }; state.ensemble = null; state.prediction.reset();
  state.faceLostAt = 0; resetDwell(); setPhase("waiting"); setStatus(status); $("#model-points").classList.remove("visible");
}

function beginCalibration() {
  if (!state.cameraReady || !state.landmarker) { showMessage("Cannot open webcam."); return; }
  state.models = null; state.calibrationSamples = []; state.prediction.reset(); state.calibrationStarted = performance.now(); state.faceLostAt = 0;
  el.calibrationProgress.style.width = "0"; setPhase("calibration");
}

function calibrationPosition(fraction) {
  for (let i=0; i<CALIBRATION_PATH.length-1; i+=1) {
    const [x0,y0,t0] = CALIBRATION_PATH[i], [x1,y1,t1] = CALIBRATION_PATH[i+1];
    if (fraction <= t1) { const f=(fraction-t0)/(t1-t0); const ease=.5*(1-Math.cos(Math.PI*f)); return [x0+(x1-x0)*ease, y0+(y1-y0)*ease]; }
  }
  return [0.5,0.5];
}

function updateCalibration(now) {
  const fraction = clamp((now-state.calibrationStarted)/(CALIBRATION_SECONDS*1000),0,1); const [x,y]=calibrationPosition(fraction);
  el.calibrationDot.style.left=`${x*100}%`; el.calibrationDot.style.top=`${y*100}%`; el.calibrationProgress.style.width=`${fraction*100}%`;
  if (state.features) state.calibrationSamples.push({ feature:[...state.features], target:[x*innerWidth,y*innerHeight] });
  if (fraction < 1) return;
  if (state.calibrationSamples.length < 40) { resetCalibration(); showMessage("Calibration failed because a face could not be tracked. Press SPACE to try again."); return; }
  state.models = trainModels(state.calibrationSamples); setPhase("tracking"); setStatus("Calibration complete. Press SPACE to ask a question.");
}

function updatePrediction(feature, dt) {
  const prediction=state.prediction.update(feature,state.models,dt);state.modelPositions=prediction.positions;state.ensemble=prediction.ensemble;
  const colors=["lr","poly","svr","knn"]; colors.forEach((name,i)=>{el.modelPoints[i].style.left=`${state.modelPositions[name][0]}px`;el.modelPoints[i].style.top=`${state.modelPositions[name][1]}px`;});
  $("#model-points").classList.add("visible");
}

function trackingTick(dt) {
  if (!state.features) { if(!state.faceLostAt)state.faceLostAt=performance.now(); if(performance.now()-state.faceLostAt>3000){resetCalibration();setStatus("Face lost — press SPACE to recalibrate.");} return; }
  state.faceLostAt=0; updatePrediction(state.features,dt);
  if (!state.responses.length || state.selectionLocked) { resetDwell(); return; }
  const [x,y]=state.ensemble; let target=null;
  for(const card of [...el.responses,el.none]){const r=card.getBoundingClientRect();if(x>=r.left&&x<=r.right&&y>=r.top&&y<=r.bottom){target=card;break;}}
  if(target!==state.dwellTarget){resetDwell();state.dwellTarget=target;if(target)target.classList.add("dwelling");}
  if(!target)return; state.dwellSeconds+=dt; target.style.setProperty("--dwell",String(clamp(state.dwellSeconds/DWELL_SECONDS,0,1)));
  if(state.dwellSeconds>=DWELL_SECONDS){const selected=target;resetDwell();if(selected===el.none)rejectResponses();else selectResponse(Number(selected.dataset.index));}
}

function processFrame(now) {
  state.frameRequest=requestAnimationFrame(processFrame); if(!state.landmarker||el.camera.readyState<2||el.camera.currentTime===state.lastVideoTime)return;
  const dt=Math.min(.1,(now-state.lastFrameAt)/1000);state.lastFrameAt=now;state.lastVideoTime=el.camera.currentTime;
  try { const result=state.landmarker.detectForVideo(el.camera,now); state.features=result.faceLandmarks?.[0]?extractFeatures(result.faceLandmarks[0],state.mode):null; } catch { state.features=null; }
  if(state.phase==="calibration")updateCalibration(now); else if(state.phase==="tracking")trackingTick(dt);
}

async function startCameraAndModel() {
  state.cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}},audio:false});
  el.camera.srcObject=state.cameraStream;el.setupVideo.srcObject=state.cameraStream;await el.camera.play();await el.setupVideo.play();state.cameraReady=true;
  el.setupFrame.classList.add("live");el.setupStatus.className="camera-status ok";el.setupStatus.textContent="Camera detected — select Get started below.";el.setupNext.disabled=false;el.setupNext.focus();
}
async function loadTrackingModel() {
  el.loading.hidden=false;el.modelProgress.style.width="30%";el.modelStatus.textContent="Loading face tracking model...";
  const {FaceLandmarker,FilesetResolver}=await import(MEDIAPIPE_MODULE);const vision=await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);el.modelProgress.style.width="70%";
  const opts={baseOptions:{modelAssetPath:FACE_MODEL,delegate:"GPU"},runningMode:"VIDEO",numFaces:1,outputFaceBlendshapes:false,outputFacialTransformationMatrixes:false};
  try{state.landmarker=await FaceLandmarker.createFromOptions(vision,opts);}catch{opts.baseOptions.delegate="CPU";state.landmarker=await FaceLandmarker.createFromOptions(vision,opts);}
  el.modelProgress.style.width="100%";el.modelStatus.textContent="Ready.";await new Promise(r=>setTimeout(r,250));el.loading.hidden=true;
  if(!state.frameRequest)state.frameRequest=requestAnimationFrame(processFrame);
}

function identityPayload(question,includePrefs=true){if(!state.profile)return{};const p={};if(state.profile.summary)p.linguistic_profile_summary=state.profile.summary;if(state.profile.samples?.length)p.exemplars=retrieveExemplars(question,state.profile.samples);if(includePrefs&&state.profile.preferenceRules?.length)p.preference_rules=state.profile.preferenceRules;return p;}
function retrieveExemplars(question,samples){const terms=new Set(String(question).toLowerCase().match(/[a-z0-9']{3,}/g)||[]);return samples.map(sample=>({sample:String(sample).slice(0,400),score:[...terms].filter(t=>String(sample).toLowerCase().includes(t)).length})).sort((a,b)=>b.score-a.score).slice(0,5).map(x=>x.sample);}
function showResponses(options){state.responses=options.slice(0,4).map(String);el.responses.forEach((card,i)=>{card.querySelector("span").textContent=state.responses[i];});}
function clearResponses(){state.responses=[];el.responses.forEach(card=>card.querySelector("span").textContent="");}

async function questionReady(question) {
  state.context=question;state.rejected=[];state.rejectionRound=0;clearResponses();el.question.hidden=false;el.question.textContent=`Q: ${question}`;setStatus("Getting responses...");
  try{const data=await api("generate-options",{question,history:state.history.slice(-5),...identityPayload(question,true)});showResponses(data.options||[]);setStatus("");}
  catch(error){showResponses(["(error)","(error)","(error)","(error)"]);setStatus(error.message);}
}

function logInteraction(entry){state.interactions.push({...entry,timestamp:Date.now()});state.interactions=state.interactions.slice(-100);saveJSON(INTERACTIONS_KEY,state.interactions);}
async function selectResponse(index) {
  if(state.selectionLocked||!state.responses[index])return;state.selectionLocked=true;const selected=state.responses[index], options=[...state.responses], history=[...state.history];setStatus(`Selected: ${selected}`);
  logInteraction({question:state.context,options_presented:options,selected,rejected:[...state.rejected],rejection_round:state.rejectionRound});state.history.push({question:state.context,answer:selected});
  try{const data=await api("expand-response",{question:state.context,response:selected,history:history.slice(-5),...identityPayload(state.context,false)});speak(data.expanded||selected);}catch{speak(selected);}
  clearResponses();state.context=null;state.rejected=[];state.rejectionRound=0;el.question.hidden=true;state.selectionLocked=false;setStatus("Press SPACE to ask a new question.");maybeUpdatePreferences();
}
async function rejectResponses() {
  if(state.selectionLocked||!state.context)return;state.selectionLocked=true;const rejectedRound=[...state.responses];logInteraction({question:state.context,options_presented:rejectedRound,selected:null,rejected:rejectedRound,rejection_round:state.rejectionRound});state.rejected.push(...rejectedRound);state.rejectionRound+=1;clearResponses();setStatus("Getting new options...");
  try{const data=await api("generate-options",{question:state.context,history:state.history.slice(-5),rejected:state.rejected,...identityPayload(state.context,true)});showResponses(data.options||[]);setStatus("");}catch(error){setStatus(error.message);}state.selectionLocked=false;maybeUpdatePreferences();
}
async function maybeUpdatePreferences(){if(!state.profile||state.interactions.length<20||state.interactions.length%20!==0)return;try{const data=await api("analyze-preferences",{interactions:state.interactions});state.profile.preferenceRules=data.rules||[];state.profile.preferenceUpdatedAt=Date.now();saveJSON(PROFILE_KEY,state.profile);}catch{}}
async function speak(text){
  if(state.profile?.voiceId){
    try{const response=await fetch("/api/tts",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({voice_id:state.profile.voiceId,text:String(text)})});if(!response.ok)throw new Error();const blob=await response.blob();const url=URL.createObjectURL(blob);state.audioPlayer?.pause();state.audioPlayer=new Audio(url);state.audioPlayer.addEventListener("ended",()=>URL.revokeObjectURL(url),{once:true});await state.audioPlayer.play();return;}catch{}
  }
  if(!window.speechSynthesis)return;window.speechSynthesis.cancel();state.speechUtterance=new SpeechSynthesisUtterance(String(text));window.speechSynthesis.speak(state.speechUtterance);
}

async function startRecording() {
  if(state.recording||state.recordingStarting)return;
  if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder){setStatus("Microphone recording is unavailable in this browser.");return;}
  state.recordingStarting=true;
  try{
    state.questionMicStream=await navigator.mediaDevices.getUserMedia({audio:true});state.questionChunks=[];state.questionRecorder=createAudioRecorder(state.questionMicStream);
    state.questionRecorder.ondataavailable=event=>{if(event.data.size)state.questionChunks.push(event.data);};state.questionRecorder.start();state.recording=true;el.recording.hidden=false;setStatus("Recording... press SPACE to stop.");
  }catch(error){setStatus(`Microphone error: ${error.message}`);}finally{state.recordingStarting=false;}
}
async function stopRecording(transcribe=true) {
  if(!state.recording)return;state.recording=false;el.recording.hidden=true;const recorder=state.questionRecorder;
  if(recorder?.state!=="inactive")await new Promise(resolve=>{recorder.addEventListener("stop",resolve,{once:true});recorder.stop();});
  state.questionMicStream?.getTracks().forEach(track=>track.stop());state.questionMicStream=null;
  if(!transcribe)return;
  setStatus("Transcribing...");const file=recordedAudioFile(state.questionChunks,recorder,"question");
  if(!file.size){setStatus("No audio recorded. Press SPACE to try again.");return;}
  try{const text=await transcribeMedia(file);if(text)await questionReady(text);else setStatus("Could not transcribe. Try again.");}
  catch{setStatus("Could not transcribe. Try again.");}
}

function splitSamples(text){return text.split(/\n\s*\n+/).map(s=>s.trim()).filter(Boolean).slice(0,50);}
function styleSummary(style){return [style.tone_description&&`Tone: ${style.tone_description}`,style.humor_style&&`Humor: ${style.humor_style}`,style.personality_notes,style.language_variety&&style.language_variety!=="unknown"&&`Language: ${style.language_variety}`,style.slang_and_regionalisms?.length&&`Characteristic wording: ${style.slang_and_regionalisms.join(", ")}`].filter(Boolean).join(". ");}
function buildSegments(){el.studioSegments.innerHTML="";for(let i=0;i<72;i+=1)el.studioSegments.append(document.createElement("i"));}
function setStudioProgress(value,label=`${value}%`){el.studioProgress.textContent=label;[...el.studioSegments.children].forEach((node,i)=>node.classList.toggle("on",i<Math.round(value*.72)));}
function validateStudio(){const ready=Boolean(el.studioName.value.trim()&&(el.studioSamples.value.trim()||state.importedSamples.length||state.importedMedia.length));el.studioGenerate.disabled=!ready;el.studioStatus.textContent=ready?"Ready to analyze the patient's communication style.":"Name the patient and add at least one source to begin.";}
function populateProfilePicker(){el.studioPicker.innerHTML='<option value="">Create a new profile…</option>';PREBUILT_PROFILES.forEach(profile=>{const option=document.createElement("option");option.value=`prebuilt:${profile.id}`;option.textContent=profile.name;el.studioPicker.append(option);});if(state.profile&&!state.profile.builtInId){const option=document.createElement("option");option.value="saved";option.textContent=state.profile.name;el.studioPicker.append(option);}}
function resetStudioResult(){state.studioSelectedProfile=null;setStudioProgress(0,"Ready");el.studioSummary.hidden=true;el.studioSummary.value="";el.studioGenerate.hidden=false;el.studioUse.hidden=true;el.studioCancel.hidden=false;el.studioCancel.disabled=false;}
function showPrebuiltProfile(id){const profile=getPrebuiltProfile(id);if(!profile)return;state.studioSelectedProfile=profile;el.studioName.value=profile.name;el.studioSamples.value=profile.samples.join("\n\n");setStudioProgress(100,"Pre-built");el.studioStatus.textContent="This pre-built profile is trained and ready to use.";el.studioSummary.value=profile.summary;el.studioSummary.hidden=false;el.studioGenerate.hidden=true;el.studioUse.hidden=false;el.studioUse.textContent=`Use ${profile.name}  →`;el.studioCancel.hidden=false;el.studioUse.focus();}
function openStudio(){if(state.recording){setStatus("Stop the current recording before editing the patient profile.");return;}populateProfilePicker();state.importedSamples=[];state.importedMedia=[];resetStudioResult();if(state.profile?.builtInId){el.studioPicker.value=`prebuilt:${state.profile.builtInId}`;showPrebuiltProfile(state.profile.builtInId);}else if(state.profile){el.studioName.value=state.profile.name||"";el.studioSamples.value=(state.profile.samples||[]).join("\n\n");el.studioPicker.value="saved";validateStudio();}else{el.studioName.value="";el.studioSamples.value="";el.studioPicker.value="";validateStudio();}el.studio.hidden=false;if(!state.profile?.builtInId)el.studioName.focus();}
function closeStudio(){if(state.studioRecording)toggleStudioRecording();el.studio.hidden=true;el.main.focus();}
async function loadStudioFiles(files){state.importedSamples=[];state.importedMedia=[];for(const file of files){if(file.size>25*1024*1024)continue;if(file.type.startsWith("audio/")||file.type.startsWith("video/")){state.importedMedia.push(file);continue;}let text=await file.text();if(file.name.endsWith(".json")){try{const data=JSON.parse(text);text=Array.isArray(data)?data.map(String).join("\n\n"):Object.values(data).map(String).join("\n\n");}catch{}}state.importedSamples.push(...splitSamples(text));}el.studioSourceStatus.hidden=false;el.studioSourceStatus.textContent=`Added ${state.importedSamples.length} text snippet(s) and ${state.importedMedia.length} media file(s). Media will be transcribed when the profile is generated.`;validateStudio();}
async function toggleStudioRecording(){
  if(!state.studioRecording){
    if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder){showMessage("Microphone recording is unavailable in this browser.");return;}
    try{state.micStream=await navigator.mediaDevices.getUserMedia({audio:true});state.mediaChunks=[];state.mediaRecorder=createAudioRecorder(state.micStream);state.mediaRecorder.ondataavailable=e=>{if(e.data.size)state.mediaChunks.push(e.data);};state.mediaRecorder.start();state.studioRecording=true;el.studioRecord.textContent="■  Stop & save";el.studioSourceStatus.hidden=false;el.studioSourceStatus.textContent="Recording… speak naturally, then press Stop.";}catch(error){showMessage(`Microphone error: ${error.message}`);}return;
  }
  state.studioRecording=false;const recorder=state.mediaRecorder;await new Promise(resolve=>{recorder.addEventListener("stop",resolve,{once:true});recorder.stop();});state.micStream?.getTracks().forEach(track=>track.stop());const file=recordedAudioFile(state.mediaChunks,recorder,"recording");if(file.size){state.importedMedia.push(file);el.studioSourceStatus.textContent=`Saved ${file.name}. It will be converted to WAV, transcribed, and cloned.`;}else el.studioSourceStatus.textContent="No audio was captured. Try again.";el.studioRecord.textContent="●  Record another";validateStudio();
}
async function transcribeMedia(file){const form=new FormData();form.append("files",file,file.name);const response=await fetch("/api/transcribe-media",{method:"POST",body:form});const data=await response.json();if(!response.ok)throw new Error(data.error||"Media transcription failed.");return String(data.text||"").trim();}
async function cloneVoice(name,files){const form=new FormData();form.append("patient_name",name);files.forEach(file=>form.append("files",file,file.name));const response=await fetch("/api/clone-voice",{method:"POST",body:form});const data=await response.json();if(!response.ok)throw new Error(data.error||"Voice cloning failed.");return data;}
async function analyzeProfile(){
  const name=el.studioName.value.trim();let samples=[...splitSamples(el.studioSamples.value),...state.importedSamples];if(!name||(!samples.length&&!state.importedMedia.length))return;
  el.studioGenerate.disabled=true;el.studioCancel.disabled=true;setStudioProgress(5);el.studioStatus.textContent="Preparing sources and building the patient profile…";
  try{
    for(let i=0;i<state.importedMedia.length;i+=1){el.studioStatus.textContent=`Transcribing media ${i+1} of ${state.importedMedia.length}…`;const transcript=await transcribeMedia(state.importedMedia[i]);if(transcript)samples.push(transcript);setStudioProgress(5+Math.round(((i+1)/state.importedMedia.length)*25));}
    let voiceProfile=null;if(state.importedMedia.length){el.studioStatus.textContent="Creating the ElevenLabs voice clone…";voiceProfile=await cloneVoice(name,state.importedMedia);}setStudioProgress(42);
    const style=await api("analyze-style",{sample_texts:samples.slice(0,50)},60000);setStudioProgress(82);const summary=styleSummary(style);
    state.profile={name,samples:samples.slice(0,50),style,summary,voiceId:voiceProfile?.voice_id||state.profile?.voiceId||null,requiresVerification:Boolean(voiceProfile?.requires_verification),preferenceRules:state.profile?.preferenceRules||[],updatedAt:Date.now()};saveJSON(PROFILE_KEY,state.profile);setStudioProgress(100,"Complete");
    el.studioStatus.textContent=state.profile.requiresVerification?"Profile generated. ElevenLabs requires verification before the cloned voice can be used.":state.profile.voiceId?"Profile generated and ElevenLabs cloned voice activated.":"Profile generated. Add a recording later to enable cloned speech.";
    el.studioSummary.hidden=false;el.studioSummary.value=summary;el.studioGenerate.hidden=true;el.studioUse.hidden=false;el.studioUse.textContent=`Use ${name}  →`;el.studioCancel.hidden=true;
  }catch(error){setStudioProgress(0,"Needs attention");el.studioStatus.textContent=`Failed — ${error.message}`;el.studioGenerate.disabled=false;el.studioCancel.disabled=false;}
}
function useProfile(){if(state.studioSelectedProfile){state.profile={...state.studioSelectedProfile,updatedAt:Date.now()};saveJSON(PROFILE_KEY,state.profile);}closeStudio();setStatus(`Patient profile loaded: ${state.profile.name}`);}
function openPreferences(){if(!state.profile)return;const rules=state.profile.preferenceRules||[];el.preferencesPatient.textContent=`Patient: ${state.profile.name}`;el.preferencesMeta.textContent=state.profile.preferenceUpdatedAt?`Last updated: ${new Date(state.profile.preferenceUpdatedAt).toLocaleString()}  ·  Based on ${state.interactions.length} interactions`:"No preferences learned yet.";el.preferencesRules.innerHTML="";if(rules.length)rules.forEach(rule=>{const p=document.createElement("p");p.textContent=`•  ${rule}`;el.preferencesRules.append(p);});else el.preferencesRules.textContent="No preference rules yet. They will be generated after 20+ interactions.";el.clearPreferences.hidden=!rules.length;el.preferences.hidden=false;el.closePreferences.focus();}

function closeApp(){state.closed=true;if(state.recording)stopRecording(false);window.speechSynthesis?.cancel();state.audioPlayer?.pause();state.cameraStream?.getTracks().forEach(track=>track.stop());state.micStream?.getTracks().forEach(track=>track.stop());state.questionMicStream?.getTracks().forEach(track=>track.stop());cancelAnimationFrame(state.frameRequest);el.setup.hidden=true;el.loading.hidden=true;el.studio.hidden=true;el.preferences.hidden=true;el.message.hidden=true;el.main.hidden=true;el.closed.hidden=false;}
async function handleKey(event) {
  if(state.closed)return;
  if(!el.message.hidden){if(event.key==="Enter"||event.key==="Escape"){event.preventDefault();closeMessage();}return;}
  if(!el.studio.hidden){if(event.key==="Escape"||event.key.toLowerCase()==="p"){event.preventDefault();closeStudio();}return;}
  if(!el.preferences.hidden){if(event.key==="Escape"){event.preventDefault();el.preferences.hidden=true;el.main.focus();}return;}
  if(event.code==="Space"){event.preventDefault();if(state.phase==="waiting")beginCalibration();else if(state.phase==="tracking"){if(state.recording)stopRecording();else startRecording();}return;}
  const key=event.key.toLowerCase();
  if(key==="m"&&(state.phase==="waiting"||state.phase==="tracking")){event.preventDefault();state.mode=state.mode==="eye"?"head":"eye";localStorage.setItem(MODE_KEY,state.mode);updateModeLabel();resetCalibration(`Switched to ${state.mode==="eye"?"Eye":"Head"} Tracking. Press SPACE to calibrate.`);}
  else if(key==="r"){event.preventDefault();resetCalibration();}
  else if(key==="c"){event.preventDefault();if(state.recording)stopRecording(false);beginCalibration();}
  else if(key==="f11"){event.preventDefault();if(document.fullscreenElement)await document.exitFullscreen();else await el.app.requestFullscreen();}
  else if(key==="p"&&(state.phase==="waiting"||state.phase==="tracking")){event.preventDefault();openStudio();}
  else if(key==="v"&&(state.phase==="waiting"||state.phase==="tracking")&&state.profile){event.preventDefault();openPreferences();}
  else if(key==="q"||event.key==="Escape"){event.preventDefault();closeApp();}
}

async function goToCameraPage(){el.setupWelcome.hidden=true;el.setupCamera.hidden=false;el.setupBack.hidden=false;el.setupNext.textContent="Get started";el.setupNext.disabled=true;try{await startCameraAndModel();}catch(error){el.setupFrame.querySelector("span").textContent="No camera image";el.setupStatus.className="camera-status error";el.setupStatus.textContent="Could not open webcam. Please connect a camera and restart.";}}
async function finishSetup(){localStorage.setItem(SETUP_KEY,"true");el.setup.hidden=true;try{await loadTrackingModel();resetCalibration();el.main.focus();}catch(error){showMessage(`Could not download the tracking model. Check your internet connection.\n\n${error.message}`);}}
function bindEvents(){
  el.setupNext.addEventListener("click",()=>el.setupWelcome.hidden?finishSetup():goToCameraPage());el.setupBack.addEventListener("click",()=>{el.setupWelcome.hidden=false;el.setupCamera.hidden=true;el.setupBack.hidden=true;el.setupNext.disabled=false;el.setupNext.textContent="Continue";});el.setupQuit.addEventListener("click",closeApp);
  el.messageOk.addEventListener("click",closeMessage);document.addEventListener("keydown",handleKey);el.studioCancel.addEventListener("click",closeStudio);el.studioUse.addEventListener("click",useProfile);el.studioGenerate.addEventListener("click",analyzeProfile);el.studioRecord.addEventListener("click",toggleStudioRecording);el.studioFiles.addEventListener("change",()=>loadStudioFiles(el.studioFiles.files));el.studioName.addEventListener("input",validateStudio);el.studioSamples.addEventListener("input",validateStudio);el.studioPicker.addEventListener("change",()=>{resetStudioResult();const value=el.studioPicker.value;if(value.startsWith("prebuilt:")){showPrebuiltProfile(value.slice(9));return;}if(value==="saved"&&state.profile){el.studioName.value=state.profile.name;el.studioSamples.value=(state.profile.samples||[]).join("\n\n");}else{el.studioName.value="";el.studioSamples.value="";}validateStudio();});
  el.closePreferences.addEventListener("click",()=>{el.preferences.hidden=true;el.main.focus();});el.clearPreferences.addEventListener("click",()=>{if(state.profile){state.profile.preferenceRules=[];delete state.profile.preferenceUpdatedAt;saveJSON(PROFILE_KEY,state.profile);}el.preferences.hidden=true;});
  window.addEventListener("pagehide",()=>{const duration=Math.max(1,(performance.now()-state.sessionStarted)/1000);navigator.sendBeacon?.("/api/telemetry",new Blob([JSON.stringify({duration_seconds:duration})],{type:"application/json"}));});
}

async function initialize(){buildSegments();bindEvents();updateModeLabel();setPhase("waiting");[...el.responses,el.none].forEach(item=>item.hidden=true);
  if(localStorage.getItem(SETUP_KEY)==="true"){el.setup.hidden=true;el.loading.hidden=false;try{await startCameraAndModel();await loadTrackingModel();resetCalibration();el.main.focus();}catch(error){el.loading.hidden=true;showMessage(`Cannot open webcam. ${error.message}`);}}else el.setupNext.focus();
}

initialize();
