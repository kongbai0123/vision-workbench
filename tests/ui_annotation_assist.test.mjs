import assert from 'node:assert/strict';
import test from 'node:test';
import {groupAssistCandidates} from '../web/pages/annotation-assist.mjs';

const candidate = (id,x,label='tool') => ({id,label,shape:{x,y:0,width:100,height:50,type:'rectangle',label}});
const confirmation = (item,run_id='batch') =>
  ({...item.shape,metadata:{source:'manual',annotation_assist:{candidate_id:item.id,run_id,human_confirmed:true}}});

test('suggestions stay available until an explicit human confirmation or rejection',()=>{
  const candidates=[candidate('first',0),candidate('nearby',3),candidate('other',120)];
  assert.deepEqual(groupAssistCandidates(candidates,[],'batch'),{active:candidates,archived:[]});
});

test('manual confirmation archives only direct alternatives and preserves decisions for other objects',()=>{
  const first=candidate('first',0), rival=candidate('rival',8,'misidentified');
  const throughRival=candidate('through-rival',16), other=candidate('other',150);
  const all=[first,rival,throughRival,other];
  const result=groupAssistCandidates(all,[confirmation(first)],'batch');
  assert.deepEqual(result.active,[throughRival,other]);
  assert.deepEqual(result.archived,[
    {candidate:first,reason:'adopted'},{candidate:rival,reason:'alternative'},
  ]);
  assert.deepEqual(groupAssistCandidates(all,[confirmation(first),confirmation(other)],'batch').active,[throughRival]);
  assert.deepEqual(all,[first,rival,throughRival,other]);
});

test('editing or undoing a human box recomputes archived candidates using native shapes',()=>{
  const first=candidate('first',0), rival=candidate('rival',3);
  const all=[first,rival], adopted=confirmation(first);
  assert.deepEqual(groupAssistCandidates(all,[{...adopted,width:60}],'batch').active,[]);
  assert.deepEqual(groupAssistCandidates(all,[{...adopted,x:260}],'batch').active,[rival]);
  assert.deepEqual(groupAssistCandidates(all,[],'batch').active,all);
  assert.deepEqual(groupAssistCandidates(all,[confirmation(first,'older')],'batch').archived,[]);
  assert.deepEqual(groupAssistCandidates(all,[],'batch',new Map([['first','rejected']])).archived,[
    {candidate:first,reason:'rejected'},
  ]);
});
