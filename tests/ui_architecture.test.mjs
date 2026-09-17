import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {mergeProjectDelta} from '../web/state.mjs';
import {nativeCallbacks, installNativeCallbacks, connectNative} from '../web/native-bridge.mjs';

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

test('page factories import without executing browser globals', async () => {
  for(const page of ['training','settings','review','export','preparation','camera']) {
    const module=await import(`../web/pages/${page}.mjs`);
    assert.equal(typeof Object.values(module)[0],'function');
  }
});

test('review controls and augmentation layout are static markup', async () => {
  const html=await readFile(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['reviewCorrection','reviewTrash','reviewRestore','reviewQuality','reviewReasonFilter'])
    assert.equal(html.split(`id="${id}"`).length-1,1);
  assert.ok(html.indexOf('class="panel augmentation-panel"')<html.indexOf('id="splitFlowStats"'));
  const app=await readFile(new URL('../web/app.mjs',import.meta.url),'utf8');
  assert.ok(!app.includes('window.workbench'));
  assert.ok(!app.includes("createElement('link')"));
});
