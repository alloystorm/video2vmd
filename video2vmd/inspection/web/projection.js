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
  return {contain,pinhole,manual,estimated};
})();
if(typeof module!=="undefined") module.exports=Projection;
