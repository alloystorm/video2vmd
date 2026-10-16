/* Canvas projection, orbit/zoom and local pose-buffer rendering adapted from
 * motion_encoder/inspect_data.py. Playback here uses the video clock, not a
 * second animation timer. No network libraries or training code are required. */
"use strict";
const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("scene"), ctx = canvas.getContext("2d");
const timeline = $("timeline"), play = $("play"), status = $("status");
let meta, poses, body, count, stride, duration = 0, current = 0;
let az = .45, el = .18, zoom = 1, panX = 0, panY = 0;
let virtualPlaying = false, virtualStart = 0, virtualTime = 0, virtualRate = 1;
let presentedTime = 0, videoFrames = false, seeking = false, resumeAfterSeek = false;
let width = 1, height = 1;

function formatTime(t) {
  const ms = Math.round(Math.max(0, t) * 1000);
  return `${String(Math.floor(ms / 60000)).padStart(2,"0")}:${String(Math.floor(ms / 1000) % 60).padStart(2,"0")}.${String(ms % 1000).padStart(3,"0")}`;
}
function isPlaying() { return meta?.mediaKind === "video" ? !video.paused && !video.ended : virtualPlaying; }
function clock() {
  if (meta.mediaKind === "video") return video.paused || video.seeking || !videoFrames ? video.currentTime : presentedTime;
  return virtualPlaying ? virtualTime + (performance.now() - virtualStart) / 1000 * virtualRate : virtualTime;
}
function setPlayingUI() {
  play.textContent = isPlaying() ? "Pause" : "Play";
  status.textContent = meta.mediaKind === "video" ? "Video and motion synchronized" : "Motion timeline · source video unavailable";
  if (meta.mediaKind === "image") status.textContent = "Static source image · motion timeline";
}
function pause() {
  if (meta.mediaKind === "video") video.pause();
  else { virtualTime = Math.min(clock(), duration); virtualPlaying = false; }
  setPlayingUI();
}
async function start() {
  if (clock() >= duration - .001) seek(0);
  if (meta.mediaKind === "video") {
    try { await video.play(); }
    catch (err) { status.textContent = `Cannot play video: ${err.message}`; return; }
  } else { virtualStart = performance.now(); virtualPlaying = true; }
  setPlayingUI();
}
function seek(t) {
  t = Math.max(0, Math.min(duration, t));
  if (meta.mediaKind === "video") { video.currentTime = t; presentedTime = t; }
  else { virtualTime = t; virtualStart = performance.now(); }
  current = t;
  updateTimeline(t);
  draw(t);
}
function step(direction) {
  pause();
  const frame = Math.round(clock() * meta.fps) + direction;
  seek(Math.max(0, Math.min(meta.frames-1, frame)) / meta.fps);
}
function updateTimeline(t) {
  if (!seeking) timeline.value = t;
  $("time").textContent = formatTime(t);
  $("frame-badge").textContent = `FRAME ${String(Math.min(meta.frames-1, Math.floor(t * meta.fps + 1e-5))).padStart(4,"0")}`;
}

// Orthographic 3D projection with yaw, pitch and root-relative translation.
function project(x, y, z, origin, scale) {
  x -= origin[0]; y -= origin[1]; z -= origin[2];
  const X = Math.cos(az)*x + Math.sin(az)*z;
  const Z = -Math.sin(az)*x + Math.cos(az)*z;
  const Y = Math.cos(el)*y - Math.sin(el)*Z;
  return [width*.5 + X*scale + panX, height*.5 - Y*scale + panY];
}
function sample(variant, bone, frame) {
  const lo = Math.floor(frame), hi = Math.min(meta.frames-1, lo+1), blend = frame-lo;
  const base = variant*stride + lo*count*3 + bone*3;
  const next = variant*stride + hi*count*3 + bone*3;
  return [0,1,2].map(axis => poses[base+axis]*(1-blend) + poses[next+axis]*blend);
}
function line(a,b,color,lineWidth=1) {
  ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
  ctx.strokeStyle=color; ctx.lineWidth=lineWidth; ctx.stroke();
}
function draw(t) {
  if (!meta) return;
  ctx.clearRect(0,0,width,height);
  const frame = Math.max(0,Math.min(meta.frames-1,t*meta.fps));
  const selection = $("variant").value;
  const selected = Math.max(0,meta.variants.indexOf(selection === "overlay" ? "corrected" : selection));
  const root = sample(selected,meta.root,frame);
  const initialRoot = sample(selected,meta.root,0);
  const center = $("follow").checked ? root : initialRoot;
  const origin = [center[0], meta.floor + meta.height*.48, center[2]];
  const scale = Math.min(width*.58,height*.78) / meta.height * zoom;
  const projectPoint = p => project(...p,origin,scale);
  const gx = Math.floor(center[0]), gz = Math.floor(center[2]);
  for (let d=-8; d<=8; d++) {
    line(projectPoint([gx+d,meta.floor,gz-8]),projectPoint([gx+d,meta.floor,gz+8]),"#232e3b");
    line(projectPoint([gx-8,meta.floor,gz+d]),projectPoint([gx+8,meta.floor,gz+d]),"#232e3b");
  }
  const zero = projectPoint([0,meta.floor,0]);
  line(zero,projectPoint([1,meta.floor,0]),"#70444d",1.5);
  line(zero,projectPoint([0,meta.floor,1]),"#406b62",1.5);
  // Root trajectory gives a stable indication of travel while following.
  ctx.beginPath();
  for (let f=Math.max(0,Math.floor(frame)-180); f<=Math.min(meta.frames-1,Math.floor(frame)+60); f+=2) {
    const p=sample(selected,meta.root,f); p[1]=meta.floor;
    const q=projectPoint(p);
    if(f===Math.max(0,Math.floor(frame)-180)) ctx.moveTo(...q); else ctx.lineTo(...q);
  }
  ctx.strokeStyle="#4a6666"; ctx.lineWidth=1.5; ctx.stroke();
  function skeleton(variant, ghost=false) {
    const points=meta.names.map((_,i)=>projectPoint(sample(variant,i,frame)));
    ctx.globalAlpha=ghost ? .5 : 1;
    if(ghost) ctx.setLineDash([5,4]);
    for(let i=0;i<count;i++) {
      const parent=meta.parents[i];
      if(parent<0 || (!$("helpers").checked && (!body.has(i) || ["Master","Center","Groove"].includes(meta.names[parent])))) continue;
      const name=meta.names[i];
      const color=ghost?"#f5cd7d":name.startsWith("Left")?"#68aaff":name.startsWith("Right")?"#ff8586":"#c6d1df";
      line(points[parent],points[i],color,body.has(i)?2.7:1);
    }
    ctx.setLineDash([]);
    for(let i=0;i<count;i++) {
      if(!$("helpers").checked && (!body.has(i) || ["Master","Center","Groove"].includes(meta.names[i]))) continue;
      ctx.beginPath(); ctx.arc(...points[i],body.has(i)?3:1.5,0,Math.PI*2);
      ctx.fillStyle=ghost?"#f5cd7d":"#e4edf7"; ctx.fill();
    }
    ctx.globalAlpha=1;
  }
  if(selection==="overlay") skeleton(meta.variants.indexOf("raw"),true);
  skeleton(selected);
}
function resize() {
  const rect=canvas.getBoundingClientRect(), ratio=window.devicePixelRatio||1;
  width=rect.width; height=rect.height;
  canvas.width=Math.round(width*ratio); canvas.height=Math.round(height*ratio);
  ctx.setTransform(ratio,0,0,ratio,0,0);
  draw(current);
}
function tick() {
  if(meta) {
    let t=clock();
    if(t>=duration && isPlaying()) {
      if($("loop").checked) { seek(0); t=0; }
      else { pause(); seek(duration); t=duration; }
    }
    current=Math.min(t,duration);
    updateTimeline(current); draw(current);
  }
  requestAnimationFrame(tick);
}

async function init() {
  const [metadataResponse,poseResponse] = await Promise.all([fetch("/api/meta"),fetch("/api/poses.bin")]);
  if(!metadataResponse.ok || !poseResponse.ok) throw new Error("Cannot load motion from the local server");
  meta=await metadataResponse.json(); poses=new Float32Array(await poseResponse.arrayBuffer());
  count=meta.names.length; stride=meta.frames*count*3; body=new Set(meta.body);
  if(poses.length!==stride*meta.variants.length) throw new Error("Incomplete pose buffer");
  document.title=`${meta.title} · V2VMD inspector`;
  $("summary").textContent=`${meta.frames.toLocaleString()} frames · ${meta.fps} fps · ${count} bones`;
  $("source-name").textContent=meta.source||"No source media";
  for(const variant of meta.variants) $("variant").add(new Option(variant==="corrected"?"Corrected":"Raw",variant));
  if(meta.variants.includes("raw")) $("variant").add(new Option("Overlay · raw / corrected","overlay"));
  $("variant").onchange=()=>{ $("overlay-key").hidden=$("variant").value!=="overlay"; draw(current); };
  duration=meta.duration;
  if(meta.mediaKind==="video") {
    video.src="/media";
    await new Promise((resolve,reject)=>{
      video.onloadedmetadata=resolve;
      video.onerror=()=>reject(new Error("The browser cannot decode this video. Install ffmpeg and reopen the inspector to create an H.264 preview."));
    });
    if(Number.isFinite(video.duration)) duration=video.duration;
    video.onplay=video.onpause=setPlayingUI;
    video.onseeked=()=>{presentedTime=video.currentTime; draw(presentedTime);};
    video.onended=()=>{if($("loop").checked) {seek(0);start();} else setPlayingUI();};
    video.onwaiting=()=>{status.textContent="Buffering source video…";};
    video.onplaying=setPlayingUI;
    if("requestVideoFrameCallback" in video) {
      videoFrames=true;
      const onFrame=(_,info)=>{
        presentedTime=info.mediaTime;
        current=presentedTime;
        updateTimeline(current);draw(current);
        video.requestVideoFrameCallback(onFrame);
      };
      video.requestVideoFrameCallback(onFrame);
    }
  } else {
    video.hidden=true; $("sound").hidden=true;
    if(meta.mediaKind==="image") {$("image").src="/media";$("image").hidden=false;}
    else {$("source-note").hidden=false;$("source-note").textContent="Source media was not found. Reopen with v2vmd inspect RUN_DIR --video PATH to show it beside this motion.";}
  }
  timeline.max=duration; timeline.disabled=false; play.disabled=false;
  $("duration").textContent=formatTime(duration);
  play.onclick=()=>isPlaying()?pause():start();
  $("previous").onclick=()=>step(-1); $("next").onclick=()=>step(1);
  timeline.onpointerdown=()=>{resumeAfterSeek=isPlaying();pause();seeking=true;};
  timeline.oninput=()=>seek(Number(timeline.value));
  const finishSeek=()=>{if(!seeking)return;seeking=false;if(resumeAfterSeek)start();};
  window.addEventListener("pointerup",finishSeek); timeline.onchange=finishSeek;
  $("speed").onchange=()=>{
    if(meta.mediaKind==="video") video.playbackRate=Number($("speed").value);
    else {virtualTime=clock();virtualStart=performance.now();virtualRate=Number($("speed").value);}
  };
  $("sound").onclick=()=>{video.muted=!video.muted;$("sound").textContent=video.muted?"Unmute":"Mute";$("sound").setAttribute("aria-pressed",String(!video.muted));};
  const reset=()=>{az=.45;el=.18;zoom=1;panX=panY=0;draw(current);};
  $("reset").onclick=reset;
  $("front").onclick=()=>{az=0;el=0;panX=panY=0;draw(current);};
  $("side").onclick=()=>{az=Math.PI/2;el=0;panX=panY=0;draw(current);};
  $("follow").onchange=$("helpers").onchange=()=>draw(current);
  let drag=null;
  canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};if(e.shiftKey){panX+=dx;panY+=dy;}else{az+=dx*.01;el=Math.max(-1.3,Math.min(1.3,el+dy*.01));}draw(current);};
  canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
  canvas.addEventListener("wheel",e=>{e.preventDefault();zoom=Math.max(.2,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));draw(current);},{passive:false});
  document.addEventListener("keydown",e=>{
    if(["INPUT","SELECT"].includes(document.activeElement.tagName)) return;
    if(e.code==="Space"){e.preventDefault();isPlaying()?pause():start();}
    if(e.code==="ArrowLeft"||e.code==="ArrowRight"){e.preventDefault();step(e.code==="ArrowLeft"?-1:1);}
  });
  new ResizeObserver(resize).observe(canvas); resize(); setPlayingUI(); tick();
}
init().catch(err=>{status.textContent=err.message;$("summary").textContent="Inspector could not load";console.error(err);});
