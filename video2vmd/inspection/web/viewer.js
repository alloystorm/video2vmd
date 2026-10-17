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
let cameraPoses=null, reconstruction=null, worldReconstruction=null, handTips=null, cameraHandTips=null, overlap=false;
const alignment={yaw:0,pitch:0,fov:45,distance:3.5,pan:[0,0],opacity:.9};

function cameraSettings() {
  for(const [id,value,suffix] of [["fov",alignment.fov,"°"],["distance",alignment.distance," m"],
        ["pan-x",alignment.pan[0]*100,"%"],["pan-y",alignment.pan[1]*100,"%"],["opacity",alignment.opacity*100,"%"]]) {
    $(id).value=id==="opacity"?alignment.opacity:value;
    $(id+"-value").textContent=Number(value.toFixed(id==="distance"?2:1))+suffix;
  }
}
function resetAlignment() {
  alignment.yaw=alignment.pitch=0; alignment.pan=[0,0];
  alignment.fov=$("estimated").checked?meta.camera.fov:45;
  alignment.distance=$("estimated").checked?meta.camera.distance:meta.height/1.1/Math.tan(Math.PI/8);
  // Keep the control usable for unusual estimated camera distances/FOVs.
  $("distance").max=Math.max(30,alignment.distance*3);
  $("fov").min=Math.min(10,alignment.fov);
  $("fov").max=Math.max(120,alignment.fov);
  cameraSettings();draw(current);
}
function setLayout() {
  $("layout").querySelector('option[value="overlap"]').disabled=!meta.mediaKind || $("focus").value!=="body";
  overlap=$("layout").value==="overlap";
  document.body.classList.toggle("overlap",overlap);
  $("camera-controls").hidden=!overlap;
  const container=$(overlap?"source-view":"motion-view");
  for(const id of ["scene","legend","frame-badge"]) container.append($(id));
  const option=$("variant").querySelector('option[value="reconstruction"]');
  if(option) option.disabled=!overlap;
  if(!overlap && $("variant").value==="reconstruction") $("variant").value="corrected";
  $("camera-note").textContent=$("estimated").checked?"GVHMR estimate · adjustments stay editable":"Manual camera · FOV, distance and pan";
  $("manual-follow").hidden=$("estimated").checked;
  resize();
}

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
  const frame = motionFrame(clock()) + direction;
  seek(Math.max(0, Math.min(meta.frames-1, frame)) / meta.fps);
}
function motionFrame(t) {
  // Browser media time is rounded to microseconds. Keep frame-boundary seeks
  // stable without skipping a frame after a fractional timeline seek.
  return Math.max(0,Math.min(meta.frames-1,Math.floor(t*meta.fps+1e-3)));
}
function updateTimeline(t) {
  if (!seeking) timeline.value = t;
  $("time").textContent = formatTime(t);
  $("frame-badge").textContent = `FRAME ${String(motionFrame(t)).padStart(4,"0")}`;
}

// Orthographic 3D projection with yaw, pitch and root-relative translation.
function project(x, y, z, origin, scale) {
  x -= origin[0]; y -= origin[1]; z -= origin[2];
  const X = Math.cos(az)*x + Math.sin(az)*z;
  const Z = -Math.sin(az)*x + Math.cos(az)*z;
  const Y = Math.cos(el)*y - Math.sin(el)*Z;
  return [width*.5 + X*scale + panX, height*.5 - Y*scale + panY];
}
function sample(variant, bone, frame, buffer=poses, bones=count) {
  const lo = Math.floor(frame), hi = Math.min(meta.frames-1, lo+1), blend = frame-lo;
  const base = variant*meta.frames*bones*3 + lo*bones*3 + bone*3;
  const next = variant*meta.frames*bones*3 + hi*bones*3 + bone*3;
  return [0,1,2].map(axis => buffer[base+axis]*(1-blend) + buffer[next+axis]*blend);
}
function line(a,b,color,lineWidth=1) {
  if(!a || !b) return;
  ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
  ctx.strokeStyle=color; ctx.lineWidth=lineWidth; ctx.stroke();
}
function draw(t) {
  if (!meta) return;
  ctx.clearRect(0,0,width,height);
  const frame = Math.max(0,Math.min(meta.frames-1,t*meta.fps));
  const selection = $("variant").value;
  const focus = $("focus").value;
  const hand = meta.hands && focus!=="body" ? meta.hands.sides[focus] : null;
  $("hand-state").textContent=hand ? (selection==="body"?"Neutral fingers":["Unavailable","Observed","Interpolated","Held"][meta.hands.status[motionFrame(t)][focus==="left"?0:1]]) : "";
  if(hand && selection!=="body" && meta.hands.wrist_status) {
    const wrist=meta.hands.wrist_status[motionFrame(t)][focus==="left"?0:1];
    $("hand-state").textContent+=" · "+["Body wrist","Palm observed","Palm interpolated","Palm held"][wrist];
  }
  const selected = Math.max(0,meta.variants.indexOf(selection === "overlay" ? "corrected" : selection));
  const handCamera=hand && $("hand-view").value==="camera" && cameraHandTips;
  const useCamera=(handCamera || overlap && $("estimated").checked) && cameraPoses;
  const tipBuffer=handCamera?cameraHandTips:handTips;
  const sourceLayer=overlap && selection==="reconstruction";
  const boneNames=sourceLayer?meta.reconstruction.names:meta.names;
  const parents=sourceLayer?meta.reconstruction.parents:meta.parents;
  const activeBody=hand?new Set(hand.bones):sourceLayer?new Set(boneNames.map((_,i)=>i)):body;
  const activeBuffer=sourceLayer?(useCamera?reconstruction:worldReconstruction):useCamera?cameraPoses:poses;
  const activeCount=boneNames.length;
  const point=(variant,bone)=>sample(sourceLayer?0:variant,bone,frame,activeBuffer,activeCount);
  const manualSource=sourceLayer && !useCamera;
  const root = manualSource?sample(0,0,frame,worldReconstruction,activeCount):sample(selected,meta.root,frame);
  const initialRoot = manualSource?sample(0,0,0,worldReconstruction,activeCount):sample(selected,meta.root,0);
  const center = $("follow").checked ? root : initialRoot;
  const framing=manualSource?meta.reconstruction:meta;
  let origin = [center[0], framing.floor + framing.height*.48, center[2]];
  let scale = Math.min(width*.58,height*.78) / meta.height * zoom;
  if(hand) {
    const points=hand.bones.map(b=>point(selected,b));
    for(const tip of hand.tips) points.push(sample(selected,tip,frame,tipBuffer,meta.hands.tip_bones.length));
    origin=[0,1,2].map(axis=>(Math.min(...points.map(p=>p[axis]))+Math.max(...points.map(p=>p[axis])))/2);
    scale=Math.min(width*.75,height*.75)/hand.span*zoom;
  }
  let projectPoint = p => project(...p,origin,scale);
  if(handCamera) {
    const points=hand.bones.map(b=>point(selected,b));
    for(const tip of hand.tips) points.push(sample(selected,tip,frame,tipBuffer,meta.hands.tip_bones.length));
    const lo=Math.floor(frame),hi=Math.min(meta.frames-1,lo+1),a=frame-lo;
    const k=meta.camera.intrinsics[lo].map((v,i)=>v*(1-a)+meta.camera.intrinsics[hi][i]*a);
    projectPoint=Projection.handCloseup(points,k,meta.camera.image_aspect,width,height,hand.span,zoom,[panX,panY]);
  }
  let imageRect=null;
  if(overlap) {
    const media=meta.mediaKind==="image"?$("image"):video;
    const aspect=meta.mediaKind==="image"?media.naturalWidth/media.naturalHeight:video.videoWidth/video.videoHeight;
    if(!Number.isFinite(aspect)||aspect<=0) return;
    imageRect=Projection.contain(width,height,aspect);
    if(useCamera) {
      const cameraAnchor=sample(0,0,frame,reconstruction,meta.camera.names.length);
      const lo=Math.floor(frame),hi=Math.min(meta.frames-1,lo+1),a=frame-lo;
      const k=meta.camera.intrinsics[lo].map((v,i)=>v*(1-a)+meta.camera.intrinsics[hi][i]*a);
      projectPoint=p=>Projection.estimated(p,cameraAnchor,k,alignment,imageRect,meta.camera.fov,meta.camera.distance);
    } else projectPoint=p=>Projection.manual(p,origin,alignment,imageRect);
    ctx.save();ctx.beginPath();ctx.rect(imageRect.x,imageRect.y,imageRect.w,imageRect.h);ctx.clip();
  }
  if(!overlap && !hand) {
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
  }
  function skeleton(variant, ghost=false) {
    const points=boneNames.map((_,i)=>projectPoint(point(variant,i)));
    ctx.globalAlpha=(ghost ? .5 : 1)*(overlap?alignment.opacity:1);
    if(ghost) ctx.setLineDash([5,4]);
    for(let i=0;i<activeCount;i++) {
      const parent=parents[i];
      if(parent<0 || (hand && (!activeBody.has(i) || !activeBody.has(parent))) || (!$("helpers").checked && (!activeBody.has(i) || ["Master","Center","Groove"].includes(boneNames[parent])))) continue;
      const name=boneNames[i].toLowerCase();
      const color=ghost?"#f5cd7d":name.startsWith("left")?"#68aaff":name.startsWith("right")?"#ff8586":"#c6d1df";
      // A dark outline keeps limbs legible against bright source footage.
      if(overlap) line(points[parent],points[i],"#0b101a",activeBody.has(i)?6:3);
      line(points[parent],points[i],color,activeBody.has(i)?2.7:1);
    }
    ctx.setLineDash([]);
    for(let i=0;i<activeCount;i++) {
      if(!points[i] || (hand && !activeBody.has(i)) || (!$("helpers").checked && (!activeBody.has(i) || ["Master","Center","Groove"].includes(boneNames[i])))) continue;
      ctx.beginPath(); ctx.arc(...points[i],activeBody.has(i)?3:1.5,0,Math.PI*2);
      ctx.fillStyle=ghost?"#f5cd7d":"#e4edf7"; ctx.fill();
    }
    if(handTips && !overlap) for(let tip=0;tip<meta.hands.tip_bones.length;tip++) {
      if(hand && !hand.tips.includes(tip)) continue;
      const bone=meta.hands.tip_bones[tip], end=projectPoint(sample(variant,tip,frame,tipBuffer,meta.hands.tip_bones.length));
      const color=ghost?"#f5cd7d":meta.names[bone].startsWith("Left")?"#68aaff":"#ff8586";
      line(points[bone],end,color,2.7);
      if(end) {ctx.beginPath();ctx.arc(...end,2.5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
    }
    ctx.globalAlpha=1;
  }
  if(selection==="overlay") skeleton(meta.variants.indexOf("raw"),true);
  skeleton(selected);
  if(overlap) ctx.restore();
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
  if(meta.reconstruction) {
    const response=await fetch("/api/world-reconstruction.bin");
    if(!response.ok) throw new Error("Cannot load source reconstruction");
    worldReconstruction=new Float32Array(await response.arrayBuffer());
    if(worldReconstruction.length!==meta.frames*meta.reconstruction.names.length*3) throw new Error("Incomplete source reconstruction buffer");
  }
  if(meta.camera) {
    const responses=await Promise.all([fetch("/api/camera-poses.bin"),fetch("/api/reconstruction.bin")]);
    if(responses.some(r=>!r.ok)) throw new Error("Cannot load saved camera poses");
    cameraPoses=new Float32Array(await responses[0].arrayBuffer());
    reconstruction=new Float32Array(await responses[1].arrayBuffer());
    if(cameraPoses.length!==poses.length || reconstruction.length!==meta.frames*meta.camera.names.length*3) throw new Error("Incomplete camera pose buffer");
  }
  if(meta.hands) {
    const response=await fetch("/api/hand-tips.bin");
    if(!response.ok) throw new Error("Cannot load hand fingertip geometry");
    handTips=new Float32Array(await response.arrayBuffer());
    if(handTips.length!==meta.variants.length*meta.frames*meta.hands.tip_bones.length*3) throw new Error("Incomplete fingertip buffer");
    if(meta.camera) {
      const cameraResponse=await fetch("/api/camera-hand-tips.bin");
      if(!cameraResponse.ok) throw new Error("Cannot load camera fingertip geometry");
      cameraHandTips=new Float32Array(await cameraResponse.arrayBuffer());
      if(cameraHandTips.length!==handTips.length) throw new Error("Incomplete camera fingertip buffer");
    }
    $("hand-view").querySelector('option[value="camera"]').disabled=!cameraHandTips;
    $("hand-view").value=cameraHandTips?"camera":"orbit";
    $("focus").disabled=false;
    $("capture-note").textContent="Fingers from hand cache · Face neutral";
  }
  document.title=`${meta.title} · V2VMD inspector`;
  $("summary").textContent=`${meta.frames.toLocaleString()} frames · ${meta.fps} fps · ${count} bones`;
  $("source-name").textContent=meta.source||"No source media";
  for(const variant of meta.variants) $("variant").add(new Option(({corrected:"Corrected",raw:"Raw",body:"Body only"})[variant]||variant,variant));
  if(meta.variants.includes("raw")) $("variant").add(new Option("Overlay · raw / corrected","overlay"));
  if(meta.reconstruction) {const option=new Option("Source reconstruction","reconstruction");option.disabled=true;$("variant").add(option);}
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
  $("layout").querySelector('option[value="overlap"]').disabled=!meta.mediaKind;
  $("estimated").disabled=!meta.camera;$("estimated").checked=!!meta.camera;
  $("estimated").onchange=()=>{
    resetAlignment();setLayout();
  };
  $("layout").onchange=setLayout;
  $("focus").onchange=()=>{ $("layout").value="split";$("hand-view").hidden=$("focus").value==="body";panX=panY=0;zoom=1;setLayout(); };
  $("hand-view").onchange=()=>{panX=panY=0;zoom=1;draw(current);};
  $("overlap-follow").onchange=()=>{$("follow").checked=$("overlap-follow").checked;draw(current);};
  $("align-reset").onclick=resetAlignment;
  for(const id of ["fov","distance","pan-x","pan-y","opacity"]) $(id).oninput=()=>{
    alignment.fov=Number($("fov").value);alignment.distance=Number($("distance").value);
    alignment.pan=[Number($("pan-x").value)/100,Number($("pan-y").value)/100];
    alignment.opacity=Number($("opacity").value);cameraSettings();draw(current);
  };
  resetAlignment();
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
  $("front").onclick=()=>{$("hand-view").value="orbit";az=0;el=0;panX=panY=0;draw(current);};
  $("side").onclick=()=>{$("hand-view").value="orbit";az=Math.PI/2;el=0;panX=panY=0;draw(current);};
  $("follow").onchange=()=>{$("overlap-follow").checked=$("follow").checked;draw(current);};
  $("helpers").onchange=()=>draw(current);
  let drag=null;
  canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{
    if(!drag)return;
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};
    if(overlap) {
      if(e.shiftKey) {
        const media=meta.mediaKind==="image"?$("image"):video;
        const aspect=meta.mediaKind==="image"?media.naturalWidth/media.naturalHeight:video.videoWidth/video.videoHeight;
        const rect=Projection.contain(width,height,aspect);
        alignment.pan=alignment.pan.map((v,i)=>Math.max(-1,Math.min(1,v+[dx/rect.w,dy/rect.h][i])));
      } else {alignment.yaw+=dx*.01;alignment.pitch=Math.max(-1.3,Math.min(1.3,alignment.pitch+dy*.01));}
      cameraSettings();
    } else if(e.shiftKey){panX+=dx;panY+=dy;}else{
      $("hand-view").value="orbit";az+=dx*.01;el=Math.max(-1.3,Math.min(1.3,el+dy*.01));
    }
    draw(current);
  };
  canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
  canvas.addEventListener("wheel",e=>{
    e.preventDefault();
    if(overlap){alignment.distance=Math.max(.2,Math.min(Number($("distance").max),alignment.distance*Math.exp(e.deltaY*.001)));cameraSettings();}
    else zoom=Math.max(.2,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));
    draw(current);
  },{passive:false});
  document.addEventListener("keydown",e=>{
    if(["INPUT","SELECT"].includes(document.activeElement.tagName)) return;
    if(e.code==="Space"){e.preventDefault();isPlaying()?pause():start();}
    if(e.code==="ArrowLeft"||e.code==="ArrowRight"){e.preventDefault();step(e.code==="ArrowLeft"?-1:1);}
  });
  new ResizeObserver(resize).observe(canvas); setLayout();
  $("layout").disabled=$("variant").disabled=false;
  setPlayingUI(); tick();
}
init().catch(err=>{status.textContent=err.message;$("summary").textContent="Inspector could not load";console.error(err);});
