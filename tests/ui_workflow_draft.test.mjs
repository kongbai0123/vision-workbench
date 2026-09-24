import test from 'node:test';
import assert from 'node:assert/strict';
import {WorkflowDraft} from '../web/workflow-draft.mjs';

test('draft writes serialize edits made in flight and restore project isolation',async()=>{
  let value={engine:'a'},release;const writes=[];
  const draft=new WorkflowDraft({collect:()=>({...value}),restore:payload=>{value=payload},api:async(path,method,body)=>{
    if(!body.payload)return {revision:0,payload:{engine:path.includes('/B/')?'b':'a'}};
    writes.push({path,...body});if(writes.length===1)await new Promise(resolve=>{release=resolve});
    return {revision:body.revision+1};
  }});
  await draft.load('A');draft.changed();const pending=draft.flush();
  value={engine:'new'};draft.changed();release();await pending;
  assert.deepEqual(writes.map(w=>w.payload.engine),['a','new']);
  assert.deepEqual(writes.map(w=>w.revision),[0,1]);assert.equal(draft.dirty,false);
  await draft.load('B');assert.equal(value.engine,'b');assert.equal(draft.revision,0);
});

test('failed draft save retains input and prevents switching project',async()=>{
  const draft=new WorkflowDraft({collect:()=>({engine:'a'}),restore:()=>{},api:async(path,method,body)=>{
    if(body.payload)throw Error('conflict');return {revision:0,payload:{}};
  }});
  await draft.load('A');draft.changed();await assert.rejects(draft.load('B'),/conflict/);
  assert.equal(draft.pid,'A');assert.equal(draft.dirty,true);
});
