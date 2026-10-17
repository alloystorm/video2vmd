/* Pure projection math shared with the numeric regression checks. */
"use strict";
const Projection = (() => {
  function contain(width,height,aspect) {
    const w=Math.min(width,height*aspect), h=w/aspect;
    return {x:(width-w)/2,y:(height-h)/2,w,h};
  }
  function rotate(p,yaw,pitch) {
    const x=Math.cos(yaw)*p[0]+Math.sin(yaw)*p[2];
    const z=-Math.sin(yaw)*p[0]+Math.cos(yaw)*p[2];
    return [x,Math.cos(pitch)*p[1]-Math.sin(pitch)*z,Math.sin(pitch)*p[1]+Math.cos(pitch)*z];
  }
  function pinhole(p,k,rect,pan,near=.05) {
    if(p[2]<=near) return null;
    return [rect.x+(k[0]*p[0]/p[2]+k[2]+pan[0])*rect.w,
            rect.y+(k[1]*p[1]/p[2]+k[3]+pan[1])*rect.h];
  }
  function manual(p,origin,settings,rect) {
    const v=rotate(p.map((n,i)=>n-origin[i]),settings.yaw,settings.pitch);
    const f=.5/Math.tan(settings.fov*Math.PI/360);
    // Manual scene coordinates are Y-up, with the camera looking along +Z.
    return pinhole([v[0],-v[1],settings.distance+v[2]],
                   [f*rect.h/rect.w,f,.5,.5],rect,settings.pan);
  }
  function estimated(p,anchor,k,settings,rect,baseFov,baseDistance) {
    // Rotate about the subject, preserving the estimated camera placement.
    const relative=[p[0]-anchor[0],-(p[1]-anchor[1]),p[2]-anchor[2]];
    const v=rotate(relative,settings.yaw,settings.pitch);
    const factor=Math.tan(baseFov*Math.PI/360)/Math.tan(settings.fov*Math.PI/360);
    const adjusted=[k[0]*factor,k[1]*factor,k[2],k[3]];
    return pinhole([v[0]+anchor[0],-v[1]+anchor[1],v[2]+anchor[2]+settings.distance-baseDistance],adjusted,rect,settings.pan);
  }
  function handCloseup(points,k,aspect,width,height,span,zoom=1,pan=[0,0]) {
    // Crop the original pinhole view without changing hand orientation. Use
    // one pixel aspect for both axes; normalized fx and fy have different units.
    const image=p=>pinhole(p,k,{x:0,y:0,w:aspect,h:1},[0,0]);
    const projected=points.map(image).filter(p=>p);
    if(!projected.length) return ()=>null;
    const center=[0,1].map(a=>(Math.min(...projected.map(p=>p[a]))+Math.max(...projected.map(p=>p[a])))/2);
    const depth=points.reduce((sum,p)=>sum+p[2],0)/points.length;
    const scale=Math.min(width*.75,height*.75)*Math.max(.05,depth)/(span*k[1])*zoom;
    return p=>{const q=image(p);return q?[width/2+(q[0]-center[0])*scale+pan[0],height/2+(q[1]-center[1])*scale+pan[1]]:null;};
  }
  return {contain,pinhole,manual,estimated,handCloseup};
})();
if(typeof module!=="undefined") module.exports=Projection;
