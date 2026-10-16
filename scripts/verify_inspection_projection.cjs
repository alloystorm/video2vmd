/* Numeric projection checks: node scripts/verify_inspection_projection.cjs */
const assert=require('node:assert/strict');
const P=require('../video2vmd/inspection/web/projection.js');
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-7,`${a} != ${b}`);
// Portrait source letterboxes horizontally; resizing keeps normalized alignment.
const rect=P.contain(1000,600,9/16);
near(rect.w,337.5);near(rect.h,600);near(rect.x,331.25);
const k=[1,1,.5,.5], anchor=[0,0,4];
const settings={yaw:0,pitch:0,fov:2*Math.atan(.5)*180/Math.PI,distance:4,pan:[0,0]};
let point=P.estimated([1,.5,4],anchor,k,settings,rect,settings.fov,4);
near(point[0],rect.x+rect.w*.75);near(point[1],rect.h*.625);
const smaller=P.contain(500,300,9/16);
let scaled=P.estimated([1,.5,4],anchor,k,settings,smaller,settings.fov,4);
near(scaled[0],point[0]/2);near(scaled[1],point[1]/2);
// Pan is a fraction of the contained image, never of its black bars.
settings.pan=[.1,-.2];
let moved=P.estimated([1,.5,4],anchor,k,settings,rect,settings.fov,4);
near(moved[0]-point[0],rect.w*.1);near(moved[1]-point[1],-rect.h*.2);
settings.pan=[0,0];
const base=settings.fov;settings.fov=90;
let wider=P.estimated([1,.5,4],anchor,k,settings,rect,base,4);
assert.ok(wider[0]<point[0]);
assert.equal(P.pinhole([0,0,-1],k,rect,[0,0]),null);
// A Y-up manual camera places higher joints above its principal point.
let manual=P.manual([0,1,0],[0,0,0],{...settings,fov:60},rect);
assert.ok(manual[1]<rect.h*.5);
console.log('Projection checks passed: letterboxing, resizing, pan, FOV, near clipping and Y-up camera.');
