import test from 'node:test';
import assert from 'node:assert/strict';
import {SaveQueue} from '../web/save-queue.mjs';
import {maskMoved} from '../web/editor.mjs';
import {encodeMask,decodeMask,validateShape,rotated,resizeRectangle,history} from '../web/shapes.mjs';

test('save queue serializes revisions and keeps edits made during an active save',async()=>{
  const calls=[],responses=[],states=[];
  const queue=new SaveQueue((id,payload)=>new Promise(resolve=>{calls.push({id,payload});responses.push(resolve)}),{delay:999999,onStatus:s=>states.push(s)});
  const asset={id:'image1',revision:'r0',shapes:[{label:'before'}],review_state:'approved'};queue.load(asset);
  asset.shapes[0].label='first';queue.changed();const saving=queue.flush();
  asset.shapes[0].label='second';queue.changed();responses.shift()({revision:'r1',review_state:'pending'});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls.length,2);assert.equal(calls[0].payload.shapes[0].label,'first');assert.equal(calls[1].payload.shapes[0].label,'second');
  assert.equal(calls[1].payload.revision,'r1');assert.equal(asset.shapes[0].label,'second');
  responses.shift()({revision:'r2',review_state:'pending'});await saving;
  assert.equal(queue.dirty,false);assert.equal(asset.revision,'r2');assert.equal(states.at(-1),'saved');clearTimeout(queue.timer);
});

test('failed save retains edits and blocks loading another image until retry succeeds',async()=>{
  let reject=true;const queue=new SaveQueue(async()=>{if(reject)throw Error('revision conflict');return {revision:'r2'}},{delay:999999});
  const asset={id:'a',revision:'r0',shapes:[{label:'source'}]};queue.load(asset);asset.shapes.push({label:'retained'});queue.changed();
  await assert.rejects(queue.flush(),/revision conflict/);assert.equal(queue.dirty,true);assert.equal(asset.shapes.length,2);assert.equal(asset.revision,'r0');
  assert.throws(()=>queue.load({id:'b',shapes:[]}),/尚未儲存/);reject=false;await queue.flush();assert.equal(queue.dirty,false);queue.load({id:'b',shapes:[]});clearTimeout(queue.timer);
});

test('column-major RLE preserves an internal hole through translation',()=>{
  const pixels=new Uint8Array(49);for(let y=1;y<=3;y++)for(let x=1;x<=3;x++)pixels[y*7+x]=1;pixels[2*7+2]=0;
  const shape={id:'stable',type:'mask',label:'part',counts:encodeMask(pixels,7,7),x:0,y:0,width:7,height:7};
  const moved=maskMoved(shape,2,1,7,7),result=decodeMask(moved.counts,7,7);
  assert.equal(moved.id,'stable');assert.equal(result[3*7+4],0);assert.equal(result.reduce((a,b)=>a+b),8);assert.equal(result[2*7+3],1);
});

test('rotated boxes keep IDs and reject out-of-image geometry',()=>{
  const rectangle={id:'stable',type:'rectangle',label:'part',x:30,y:30,width:20,height:10,metadata:{source:'fixture'}};
  const result=rotated(rectangle,30,100,100);assert.equal(result.id,'stable');assert.equal(result.type,'obb');assert.equal(result.points.length,4);assert.deepEqual(result.metadata,rectangle.metadata);
  assert.throws(()=>rotated({...rectangle,x:0,y:0},45,100,100),/無效/);
});

test('rectangle edge resizing keeps the opposite edge, unrelated axis, and metadata', () => {
  const rectangle = Object.freeze({id: 'stable', label: 'part', x: 20, y: 30, width: 15, height: 25,
    hidden: true, metadata: {source: 'fixture'}});
  const cases = [
    ['n', {x: 80, y: -5}, {y: 25, height: 30}],
    ['e', {x: 5, y: -80}, {width: 20}],
    ['s', {x: -80, y: 5}, {height: 30}],
    ['w', {x: -5, y: 80}, {x: 15, width: 20}],
  ];
  for (const [direction, delta, changes] of cases) {
    assert.deepEqual(resizeRectangle(rectangle, direction, delta, 100, 80), {...rectangle, ...changes}, direction);
  }
});

test('rectangle corners stay inside the image and cross the opposite corner with positive extents', () => {
  const rectangle = {type: 'rectangle', label: 'part', x: 20, y: 30, width: 15, height: 25};
  assert.deepEqual(resizeRectangle(rectangle, 'ne', {x: 100, y: -100}, 100, 80),
    {...rectangle, y: 0, width: 80, height: 55});
  assert.deepEqual(resizeRectangle(rectangle, 'sw', {x: -100, y: 100}, 100, 80),
    {...rectangle, x: 0, width: 35, height: 50});
  assert.deepEqual(resizeRectangle(rectangle, 'se', {x: 4, y: 6}, 100, 80),
    {...rectangle, width: 19, height: 31});
  assert.deepEqual(resizeRectangle(rectangle, 'nw', {x: 40, y: 40}, 100, 80),
    {...rectangle, x: 35, y: 55, width: 25, height: 15});
  assert.throws(() => resizeRectangle(rectangle, 'e', {x: -15, y: 0}, 100, 80), /尺寸無效/);
});

test('zero and clamped drag preserve fractional geometry exactly', () => {
  const rectangle = {type: 'rectangle', label: 'part', x: 0, y: 0.1, width: 0.2, height: 0.2};
  assert.deepEqual(resizeRectangle(rectangle, 'se', {x: 0, y: 0}, 100, 80), rectangle);
  assert.deepEqual(resizeRectangle(rectangle, 'w', {x: -10, y: 40}, 100, 80), rectangle);
  assert.equal(resizeRectangle(rectangle, 'e', {x: 5, y: 40}, 100, 80).height, rectangle.height);
});

test('mask translation stops at the image boundary without discarding foreground pixels',()=>{
  const pixels=new Uint8Array(25);pixels[1*5+1]=1;pixels[1*5+2]=1;pixels[2*5+1]=1;
  const shape={type:'mask',label:'object',counts:encodeMask(pixels,5,5)};
  const result=decodeMask(maskMoved(shape,100,-100,5,5).counts,5,5);
  assert.equal(result.reduce((a,b)=>a+b),3);assert.equal(result[0*5+4],1);assert.equal(result[1*5+3],1);
});

test('invalid and mismatched RLE is rejected, operation history is isolated',()=>{
  assert.throws(()=>validateShape({type:'mask',label:'x',counts:[3]},2,2),/RLE/);
  const a=history(2),b=history(2);a.push([{id:'x',label:'original'}]);const next=a.undo([{id:'x',label:'changed'}]);next[0].label='edited';
  assert.equal(b.canUndo,false);assert.equal(a.redo(next)[0].label,'changed');
});
