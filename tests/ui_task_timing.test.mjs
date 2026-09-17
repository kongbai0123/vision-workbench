import test from 'node:test';
import assert from 'node:assert/strict';
import {TaskTiming,duration,taskStage} from '../web/task-timing.mjs';

function fixture(){let now=0;const timing=new TaskTiming(()=>now);return {timing,at(t,p,state='running',phase='default'){now=t;timing.update(p,state,phase);return timing.text()},time(t){now=t;return timing.text()}};}
test('duration is readable for seconds, minutes and hours',()=>{
  assert.equal(duration(4.2),'5 秒');assert.equal(duration(61),'1 分 1 秒');assert.equal(duration(3660),'1 小時 1 分');
});
test('ETA needs real advances, not repeated polling or elapsed time',()=>{
  const f=fixture();f.at(0,0);for(let n=1;n<10;n++)f.at(n,0);
  assert.match(f.time(10),/無法估算/);f.at(11,null);assert.match(f.time(12),/無法估算/);
});
test('measured throughput estimates remaining stage work',()=>{
  const f=fixture();f.at(0,0);f.at(2,10);f.at(4,20);
  assert.match(f.at(6,30),/預估剩餘約 14 秒/);
  assert.match(f.time(40),/進度暫未更新/);
});
test('pause excludes paused time and resume resamples',()=>{
  const f=fixture();f.at(0,0);f.at(2,10);f.at(4,20);f.at(6,30);
  assert.match(f.at(8,30,'paused'),/已暫停/);
  assert.match(f.at(108,30,'running'),/無法估算/);
  assert.equal(f.timing.elapsed,8);
});
test('regression, unknown progress and phase switches invalidate ETA',()=>{
  for(const change of ['regress','unknown','phase']){
    const f=fixture();f.at(0,0);f.at(2,10);f.at(4,20);f.at(6,30);
    const text=f.at(8,change==='regress'?1:change==='unknown'?null:40,'running',change==='phase'?'saving':'default');
    assert.doesNotMatch(text,/預估剩餘約/);
  }
});
test('terminal states freeze elapsed time and never imply early success',()=>{
  for(const state of ['succeeded','failed','cancelled','stopped']){
    const f=fixture();f.at(0,0);assert.match(f.at(5,100),/正在確認完成/);
    const ended=f.at(6,100,state);assert.doesNotMatch(ended,/\d/);assert.equal(f.time(100),ended);assert.equal(f.timing.elapsed,6);
  }
});
test('queued time is not counted and stalled recovery needs fresh samples',()=>{
  const f=fixture();assert.match(f.at(0,null,'queued'),/排隊等待/);
  f.at(100,0);f.at(102,10);f.at(104,20);f.at(106,30);
  assert.doesNotMatch(f.time(107),/已耗時/);
  assert.match(f.at(150,40),/無法估算/);
});
test('task messages omit item counters and elapsed-time substitutes',()=>{
  assert.equal(taskStage('保存原圖 12 / 66'),'保存原圖');
  assert.equal(taskStage('影片推論第 981 幀'),'影片推論');
  assert.equal(taskStage('正在處理（已進行 12.3 秒）'),'正在處理');
});
