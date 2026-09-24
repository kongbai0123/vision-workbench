import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {mergeProjectDelta} from '../web/state.mjs';
import {nativeCallbacks, installNativeCallbacks, connectNative} from '../web/native-bridge.mjs';
import {filterModelVersions} from '../web/pages/training.mjs';
import {filterReviewAssets} from '../web/pages/review.mjs';
import {visibleScarcityMessages} from '../web/split-manager.mjs';
import {augmentationProjection} from '../web/pages/preparation.mjs';

test('project deltas preserve selection ordering, replace and delete only changed IDs', () => {
  const current={id:'p',revision:1,assets:[{id:'b'},{id:'a'},{id:'c'}]};
  const update={id:'p',revision:2,delta:{base_revision:1,changed_ids:['a','c','d']},assets:[{id:'a',revision:3},{id:'d'}]};
  const merged=mergeProjectDelta(current,update);
  assert.deepEqual(merged.assets,[{id:'b'},{id:'a',revision:3},{id:'d'}]);
  assert.equal(current.assets.length,3);
  assert.equal(mergeProjectDelta({...current,revision:3},update),null);
  assert.equal(mergeProjectDelta({...current,id:'other'},update),null);
});

test('desktop bridge centralizes callbacks and supports browser-only mode', async () => {
  const target={};installNativeCallbacks(target);
  nativeCallbacks.state=()=>({dirty:true});
  assert.deepEqual(target.workbenchState(),{dirty:true});
  nativeCallbacks.flush=async()=>true;
  assert.equal(await target.workbenchFlush(),true);
  assert.equal(await connectNative(target),null);
});

test('split warnings only discuss source scarcity when source isolation is enabled', () => {
  const item={messages:['來源群組不足以在各集合提供獨立來源','Test 少於 5 張，評估可能不穩定']};
  assert.deepEqual(visibleScarcityMessages(item,false),['Test 少於 5 張，評估可能不穩定']);
  assert.deepEqual(visibleScarcityMessages(item,true),item.messages);
});

test('augmentation projection expands Train without changing evaluation splits', () => {
  assert.deepEqual(augmentationProjection({train:60,val:20,test:20},2),{
    rows:{
      train:{originals:60,expanded:120,events:180},
      val:{originals:20,expanded:0,events:20},
      test:{originals:20,expanded:0,events:20},
    },
    originals:100,expanded:120,trainEvents:180,
  });
});

test('web interface never declares text below 12px', async () => {
  const cssFiles=['style.css','acquisition.css','review.css','review-policy.css','training.css','split-manager.css'];
  const violations=[];
  for(const file of cssFiles) {
    const source=await readFile(new URL(`../web/${file}`,import.meta.url),'utf8');
    const declarations=[
      ...source.matchAll(/\bfont-size\s*:\s*([0-9]*\.?[0-9]+)px/gi),
      ...source.matchAll(/\bfont\s*:\s*[^;{}]*?\b([0-9]*\.?[0-9]+)px\b/gi),
    ];
    for(const match of declarations) {
      if(Number(match[1])<12) violations.push(`${file}: ${match[0]}`);
    }
  }
  assert.deepEqual(violations,[]);
});

test('page factories import without executing browser globals', async () => {
  for(const page of ['training','settings','review','export','preparation','camera']) {
    const module=await import(`../web/pages/${page}.mjs`);
    assert.equal(typeof Object.values(module)[0],'function');
  }
});

test('review controls and augmentation layout are static markup', async () => {
  const html=await readFile(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['reviewCorrection','reviewTrash','reviewDelete','reviewRestore','reviewQuality','reviewAnnotationFilter','reviewReasonFilter','augmentationSplitProjection','createPreparationDatasetVersion'])
    assert.equal(html.split(`id="${id}"`).length-1,1);
  assert.ok(html.indexOf('class="panel augmentation-panel"')<html.indexOf('id="splitFlowStats"'));
  assert.match(html,/原圖分割與每輪輸入/);
  assert.match(html,/原始圖片分割/);
  const app=await readFile(new URL('../web/app.mjs',import.meta.url),'utf8');
  assert.ok(!app.includes('window.workbench'));
  assert.ok(!app.includes("createElement('link')"));
});

test('review assets can be classified by whether they contain annotations',()=>{
  const assets=[
    {id:'a',name:'標註.png',review_state:'pending',shape_count:2,source:{}},
    {id:'b',name:'空白.png',review_state:'pending',shape_count:0,source:{}},
    {id:'c',name:'未載入.png',review_state:'approved',source:{}},
  ];
  assert.deepEqual(filterReviewAssets(assets,{status:'all',annotation:'annotated'}).map(a=>a.id),['a']);
  assert.deepEqual(filterReviewAssets(assets,{status:'all',annotation:'unannotated'}).map(a=>a.id),['b','c']);
  assert.deepEqual(filterReviewAssets(assets,{status:'pending',annotation:'unannotated'}).map(a=>a.id),['b']);
});

test('AI annotation and model trial have separate workflow locations',async()=>{
  const html=await readFile(new URL('../web/index.html',import.meta.url),'utf8');
  assert.match(html,/SAM2 使用官方預訓練權重，不需要先訓練專案模型/);
  assert.match(html,/id="generateCurrentPrediction"/);
  assert.match(html,/id="trialImages"/);assert.match(html,/id="trialVideo"/);
  assert.match(html,/影片逐幀推論，不跳幀/);
  assert.ok(html.indexOf('id="generateCurrentPrediction"')<html.indexOf('id="models"'));
  assert.ok(!html.includes('id="predictionTarget"'));
});

test('model versions can be narrowed by dataset, engine and search text',async()=>{
  const models=[
    {model_version_id:'M001',run_id:'R001',dataset_version_id:'D001',engine:'yolo',engine_name:'YOLO Detect'},
    {model_version_id:'M002',run_id:'R002',dataset_version_id:'D002',engine:'maskrcnn',engine_name:'Mask R-CNN'},
    {model_version_id:'M003',run_id:'R003',dataset_version_id:'D002',engine:'yolo',engine_name:'YOLO Detect'},
  ];
  assert.deepEqual(filterModelVersions(models,{dataset:'D002'}).map(item=>item.model_version_id),['M002','M003']);
  assert.deepEqual(filterModelVersions(models,{dataset:'D002',engine:'yolo'}).map(item=>item.model_version_id),['M003']);
  assert.deepEqual(filterModelVersions(models,{query:'r002'}).map(item=>item.model_version_id),['M002']);
  const html=await readFile(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['modelDatasetFilter','modelEngineFilter','modelSearch','modelFilterCount','clearModelFilters'])assert.match(html,new RegExp(`id="${id}"`));
  const training=await readFile(new URL('../web/pages/training.mjs',import.meta.url),'utf8');assert.match(training,/訓練設定與差異說明/);
});
