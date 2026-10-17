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
// Source-camera hand crop preserves screen directions, including camera Y down,
// perspective and pixel aspect. Principal point cancels when centering the crop.
const handPoints=[[.2,-.1,4],[-.1,.2,4.1],[.3,.2,3.8]];
const handK=[2,1.25,.5,.5],aspect=.625;
const crop=P.handCloseup(handPoints,handK,aspect,600,800,.25);
const a=crop(handPoints[0]),b=crop(handPoints[1]);
const originalA=P.pinhole(handPoints[0],handK,{x:0,y:0,w:1000,h:1600},[0,0]);
const originalB=P.pinhole(handPoints[1],handK,{x:0,y:0,w:1000,h:1600},[0,0]);
near((b[0]-a[0])/(b[1]-a[1]),(originalB[0]-originalA[0])/(originalB[1]-originalA[1]));
assert.ok(b[0]<a[0] && b[1]>a[1]);
const shifted=P.handCloseup(handPoints,[2,1.25,.1,.8],aspect,600,800,.25);
near(shifted(handPoints[0])[0],a[0]);near(shifted(handPoints[0])[1],a[1]);
const resized=P.handCloseup(handPoints,handK,aspect,300,400,.25);
near(resized(handPoints[0])[0],a[0]/2);near(resized(handPoints[0])[1],a[1]/2);
assert.equal(crop([0,0,-1]),null);
console.log('Projection checks passed: letterboxing, resizing, pan, FOV, near clipping, Y-up camera and source-camera hand crops.');
