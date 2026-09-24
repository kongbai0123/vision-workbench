import test from 'node:test';
import assert from 'node:assert/strict';
import {trainingTimeText,trainingTimeDetail,trainingProgress,quickEstimateText} from '../web/training-time.mjs';

const run={status:'running',timing:{phase:'training',phase_label:'訓練',completed:1,total:10,elapsed_seconds:2,
  remaining_seconds:18,lower_seconds:9,upper_seconds:27,samples:1,source:'live',updated_at:100,stale_after_seconds:30,seconds_since_advance:0}};
test('first measured batch shows quick ETA and a truthful range',()=>{
  assert.match(trainingTimeText(run,100),/快速初估.*18 秒/);
  assert.match(trainingTimeDetail(run,100),/波動範圍 9 秒～27 秒/);
  assert.deepEqual(trainingProgress(run),{label:'訓練',value:10});
});
test('stale observations cannot continue a false countdown',()=>{
  assert.match(trainingTimeText(run,140),/等待進度回報/);
  assert.doesNotMatch(trainingTimeText(run,140),/剩餘|無法估算/);
  assert.doesNotMatch(trainingTimeDetail(run,140),/波動範圍/);
  assert.doesNotMatch(trainingTimeText({...run,status:'stopped'},101),/剩餘/);
  assert.doesNotMatch(trainingTimeText({...run,status:'failed'},101),/剩餘/);
});
test('partial estimate is labelled as phase only and old runs remain readable',()=>{
  const partial={...run,timing:{...run.timing,remaining_seconds:null,phase_remaining_seconds:18}};
  assert.match(trainingTimeText(partial,100),/訓練剩餘/);
  assert.doesNotMatch(trainingTimeText(partial,100),/全程/);
  assert.match(trainingTimeDetail(partial,100),/其他階段/);
  assert.equal(trainingTimeText({status:'running',message:'Epoch 1/10'}),'Epoch 1/10');
});
test('history is clearly initial and missing history is not a made-up number',()=>{
  assert.match(quickEstimateText({remaining_seconds:100,lower_seconds:80,upper_seconds:120,history_runs:3}),/歷史初估.*3 次/);
  assert.match(quickEstimateText({remaining_seconds:null}),/首批實測/);
  assert.doesNotMatch(quickEstimateText({remaining_seconds:null}),/0 秒|無法估算/);
});
