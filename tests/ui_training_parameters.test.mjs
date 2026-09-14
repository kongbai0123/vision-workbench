import test from 'node:test';
import assert from 'node:assert/strict';
import {parseParameter,parameterPayload} from '../web/training-parameters.mjs';

test('parameter entry validates integer range and required Ultralytics image multiple',()=>{
  const field={key:'image_size',label:'影像尺寸',type:'integer',min:128,max:2048,step:32};
  assert.equal(parseParameter(field,'640'),640);
  for(const value of ['', 'NaN','Infinity','127','2049','129','256.5'])assert.throws(()=>parseParameter(field,value));
});
test('numeric rates preserve user decimals and reject missing or nonfinite input',()=>{
  const field={label:'學習率',type:'number',min:1e-8,max:1};
  assert.equal(parseParameter(field,'0.003'),.003);
  for(const value of ['',null,'NaN','Infinity','0','-1','2'])assert.throws(()=>parseParameter(field,value));
});
test('payload contains only current model schema fields and validates linked thresholds',()=>{
  const schema=[{key:'epochs',label:'輪數',type:'integer',min:1,max:200},{key:'threshold_min',label:'下限',type:'number',min:1e-8,max:100},{key:'threshold_max',label:'上限',type:'number',min:1e-8,max:100}];
  assert.deepEqual(parameterPayload(schema,{epochs:'12',threshold_min:'1',threshold_max:'2',learning_rate:'.003'}),{epochs:12,threshold_min:1,threshold_max:2});
  assert.throws(()=>parameterPayload(schema,{epochs:'12',threshold_min:'2',threshold_max:'1'}));
  assert.throws(()=>parameterPayload([],{epochs:'12'}));
});
test('optimizer options reject unsupported choices',()=>{
  const field={label:'優化器',type:'select',options:[{value:'AdamW',label:'AdamW'},{value:'SGD',label:'SGD'}]};
  assert.equal(parseParameter(field,'SGD'),'SGD');
  assert.throws(()=>parseParameter(field,'auto'));
});
