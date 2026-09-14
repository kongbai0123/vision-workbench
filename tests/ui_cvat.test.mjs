import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../web/app.mjs',import.meta.url),'utf8');
const launchSource=source.slice(source.indexOf('async function openCvat()'),source.indexOf('async function refreshCvat('));
function harness(pollJob) {
  const elements={};
  const context={
    state:{project:{id:'test'},editorMode:'cvat',nativeBridge:{openCvat:(_,done)=>done(true)}},
    $:id=>elements[id]??=( {hidden:false,textContent:''} ),
    api:async()=>({id:'job'}),pollJob,
    updateCvatSetupText(){elements.cvatSetupText={textContent:context.state.cvatPollBaseText}},
    clearTimeout(){},clearInterval(){},setInterval(){return 1},Date,toast(){},
  };
  vm.createContext(context);vm.runInContext(launchSource,context);
  return {context,elements};
}

test('launch failure is visible and can be retried successfully',async()=>{
  let fail=true;
  const {context:c,elements}=harness(async()=>{
    if(fail)throw Error('CVAT 專案同步失敗（HTTP 500）。');
    return {ticket:'ticket'};
  });
  await assert.rejects(c.openCvat(),/HTTP 500/);
  assert.match(elements.cvatSetupText.textContent,/開啟失敗.*HTTP 500/);
  assert.equal(elements.retryCvat.hidden,false);
  assert.equal(c.state.cvatLaunching,false);
  fail=false;await c.openCvat();
  assert.equal(elements.retryCvat.hidden,true);
  assert.equal(c.state.cvatWasOpened,true);
  assert.equal(c.state.cvatPollBusy,false);
});

test('duplicate launch is ignored and leaving CVAT prevents late window opening',async()=>{
  let resolveJob,calls=0,opened=0;
  const {context:c}=harness(()=>{calls++;return new Promise(resolve=>resolveJob=resolve)});
  c.state.nativeBridge.openCvat=()=>opened++;
  const pending=c.openCvat();await Promise.resolve();
  await c.openCvat();assert.equal(calls,1);
  c.state.editorMode='builtin';resolveJob({ticket:'ticket'});await pending;
  assert.equal(opened,0);assert.equal(c.state.cvatLaunching,false);
});
