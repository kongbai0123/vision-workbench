import test from 'node:test';
import assert from 'node:assert/strict';
import {SaveQueue} from '../web/save-queue.mjs';
import {maskMoved} from '../web/editor.mjs';
import {encodeMask,decodeMask,validateShape,rotated,history} from '../web/shapes.mjs';

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
