// The backend schema is the source of truth for editable model parameters.
export function parseParameter(field, raw) {
  if(field.type==='select'){
    if(!field.options?.some(option=>option.value===raw))throw Error(`${field.label}：請選擇有效選項。`);
    return raw;
  }
  if(typeof raw!=='string'||raw.trim()==='')throw Error(`${field.label}：請填入數值。`);
  const value=Number(raw);
  if(!Number.isFinite(value)||(field.type==='integer'&&!Number.isInteger(value)))throw Error(`${field.label}：請填入有效${field.type==='integer'?'整數':'數值'}。`);
  if(field.min!==undefined&&value<field.min)throw Error(`${field.label}不得小於 ${field.min}。`);
  if(field.max!==undefined&&value>field.max)throw Error(`${field.label}不得大於 ${field.max}。`);
  if(field.type==='integer'&&field.step>1&&value%field.step!==0)throw Error(`${field.label}必須是 ${field.step} 的倍數。`);
  return value;
}

export function parameterPayload(schema, values) {
  if(!Array.isArray(schema)||!schema.length)throw Error('此模型尚無可用的訓練參數；請確認模型已準備完成。');
  const result=Object.fromEntries(schema.map(field=>[field.key,parseParameter(field,values[field.key])]));
  if(result.threshold_min!==undefined&&result.threshold_max!==undefined&&result.threshold_min>=result.threshold_max)throw Error('閾值下限必須小於閾值上限。');
  return result;
}

export class TrainingParameters {
  constructor({preferredDevice=()=> 'auto'}={}){
    this.preferredDevice=preferredDevice;this.drafts=new Map();this.engine=null;this.signature='';this.fields=[];this.bound=new WeakSet();
    this.advanced=document.getElementById('trainingAdvancedFields');this.hint=document.getElementById('trainingParameterHint');
    this.resetButton=document.getElementById('resetTrainingParameters');
    this.resetButton.onclick=()=>{if(!this.engine)return;this.drafts.delete(this.engine.key);this.signature='';this.render(this.engine,{discardCurrent:true})};
  }
  reset(){this.drafts.clear();this.engine=null;this.signature='';this.fields=[]}
  inputs(){return [...document.querySelectorAll('[data-training-param]')]}
  remember(){if(!this.engine)return;this.drafts.set(this.engine.key,Object.fromEntries(this.inputs().filter(input=>!input.disabled).map(input=>[input.dataset.trainingParam,input.value])))}
  render(engine,{discardCurrent=false}={}){
    const fields=engine?.parameters||[],signature=JSON.stringify([engine?.key,fields]);
    if(signature===this.signature)return;
    if(!discardCurrent)this.remember();
    this.engine=engine;this.fields=fields;this.signature=signature;
    const draft=this.drafts.get(engine?.key)||{},byKey=new Map(fields.map(field=>[field.key,field]));
    for(const key of ['epochs','seed','device']){
      const wrapper=document.querySelector(`[data-core-param="${key}"]`),input=wrapper.querySelector('input,select'),field=byKey.get(key);
      wrapper.hidden=!field;input.disabled=!field;
      if(field){wrapper.querySelector('label').textContent=field.label;this.configure(input,field,draft[key])}
    }
    this.advanced.replaceChildren();
    for(const field of fields.filter(field=>field.advanced)){
      const wrap=document.createElement('div'),label=document.createElement('label'),input=document.createElement(field.type==='select'?'select':'input');
      input.id=`trainingParam-${field.key}`;label.htmlFor=input.id;label.textContent=field.label;
      this.configure(input,field,draft[field.key]);wrap.append(label,input);
      if(field.description){const note=document.createElement('p');note.className='field-note';note.id=`${input.id}-hint`;note.textContent=field.description;input.setAttribute('aria-describedby',note.id);wrap.append(note)}
      this.advanced.append(wrap);
    }
    document.getElementById('trainingAdvancedSection').hidden=!this.advanced.childElementCount;
    this.resetButton.disabled=!fields.length;
    this.hint.textContent=fields.length?(engine.component==='builtin'?'內建基準在 CPU 調整分割閾值，不使用梯度優化參數。':'參數會隨本次訓練固定保存；影像尺寸與批次大小會影響記憶體需求。'):Array.isArray(engine?.parameters)?'此模型的訓練整合尚未完成。':'請更新並重新開啟工作台，以取得此模型的參數設定。';
  }
  configure(input,field,remembered){
    input.dataset.trainingParam=field.key;input.disabled=false;
    if(field.type==='select'){
      input.replaceChildren();for(const option of field.options||[])input.append(new Option(option.label,option.value));
    }else{
      input.type='number';input.required=true;
      for(const name of ['min','max']){if(field[name]!==undefined)input.setAttribute(name,field[name]);else input.removeAttribute(name)}
      input.step=field.type==='integer'?String(field.step||1):'any';
    }
    input.value=remembered??String(field.key==='device'?this.preferredDevice():field.default);
    if(!this.bound.has(input)){input.addEventListener('input',()=>this.remember());if(input.tagName==='SELECT')input.addEventListener('change',()=>this.remember());this.bound.add(input)}
  }
  collect(){
    const values=Object.fromEntries(this.inputs().filter(input=>!input.disabled).map(input=>[input.dataset.trainingParam,input.value]));
    return parameterPayload(this.fields,values);
  }
}
