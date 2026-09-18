import test from 'node:test';
import assert from 'node:assert/strict';
import {drawTrialLabel,trialLabelFontSize,trialLabelLayout,trialSourceSummary} from '../web/pages/training.mjs';

test('trial results name the model that produced them',()=>{
  const models=[{model_version_id:'M001',engine_name:'外部 YOLO · 偵測'},{model_version_id:'M002',engine_name:'YOLO Seg'}];
  assert.equal(trialSourceSummary(null,{selectedModel:'M001',models}),null);

  const trial=trialSourceSummary({model_version_id:'M001'},{selectedModel:'M001',models});
  assert.equal(trial.label,'此結果來自 M001 · 外部 YOLO · 偵測 · 圖片／影片試跑');
  assert.equal(trial.mismatch,'');

  const moved=trialSourceSummary({model_version_id:'M001'},{selectedModel:'M002',models});
  assert.equal(moved.label,trial.label);
  assert.match(moved.mismatch,/M002/);

  const comparison=trialSourceSummary({model_version_id:'M002',comparison:{split:'val'}},{selectedModel:'M002',models});
  assert.equal(comparison.label,'此結果來自 M002 · YOLO Seg · Validation 標註比對');

  const unknown=trialSourceSummary({model_version_id:'M009'},{selectedModel:'',models});
  assert.equal(unknown.label,'此結果來自 M009 · 圖片／影片試跑');
  assert.equal(unknown.mismatch,'');
});

test('model trial labels never render below 12 CSS pixels',()=>{
  assert.equal(trialLabelFontSize(0),12);
  assert.equal(trialLabelFontSize(640),12);
  assert.equal(trialLabelFontSize(960),12);
  assert.equal(trialLabelFontSize(1120),14);
  assert.equal(trialLabelFontSize(2000),16);
});

test('model trial label backgrounds stay inside the rendered image',()=>{
  const right=trialLabelLayout({canvasWidth:320,canvasHeight:180,anchorX:310,anchorY:100,textWidth:90,fontSize:12});
  assert.equal(right.x+right.width,320);
  assert.ok(right.height>12);
  assert.equal(right.textY,right.y+right.height/2);

  const top=trialLabelLayout({canvasWidth:320,canvasHeight:180,anchorX:-20,anchorY:4,textWidth:90,fontSize:12});
  assert.equal(top.x,0);
  assert.equal(top.y,4);
  assert.ok(top.y+top.height<=180);

  const oversized=trialLabelLayout({canvasWidth:80,canvasHeight:20,anchorX:40,anchorY:10,textWidth:200,fontSize:16});
  assert.deepEqual([oversized.x,oversized.y,oversized.width,oversized.height],[0,0,80,20]);
});

test('model trial label drawing uses CSS pixels on high density displays',()=>{
  const calls=[],context={font:'',textBaseline:'',fillStyle:'',save:()=>calls.push(['save']),restore:()=>calls.push(['restore']),setTransform:(...args)=>calls.push(['transform',...args]),measureText:()=>({width:72}),fillRect:(...args)=>calls.push(['rect',...args]),fillText:(...args)=>calls.push(['text',...args])};
  const layout=drawTrialLabel(context,{label:'part 92%',canvasWidth:640,canvasHeight:360,anchorX:120,anchorY:80,pixelRatio:2});
  assert.equal(layout.fontSize,12);
  assert.match(context.font,/600 12px/);
  assert.deepEqual(calls[1],['transform',2,0,0,2,0,0]);
  assert.equal(calls.at(-1)[0],'restore');
});
