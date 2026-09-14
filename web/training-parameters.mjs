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
  const result=Object.fromEntries(schema.filter(field=>!field.when||field.when.includes(values.scheduler)).map(field=>[field.key,parseParameter(field,values[field.key])]));
  if(result.threshold_min!==undefined&&result.threshold_max!==undefined&&result.threshold_min>=result.threshold_max)throw Error('閾值下限必須小於閾值上限。');
  if(result.min_learning_rate>result.learning_rate)throw Error('最低學習率不得大於初始學習率。');
  if(result.warmup_epochs>=result.epochs)throw Error('暖身輪數必須小於總訓練輪數。');
  return result;
}

export function plannedRates(config){
  const n=Number(config.epochs),initial=Number(config.learning_rate),minimum=Number(config.min_learning_rate),warmup=Number(config.warmup_epochs||0);
  if(!Number.isInteger(n)||n<1||n>200||!Number.isFinite(initial)||initial<=0||config.scheduler==='plateau')return [];
  if(config.scheduler!=='fixed'&&(!Number.isFinite(minimum)||minimum<=0||minimum>initial||warmup>=n))return [];
  return Array.from({length:n},(_,i)=>{
    if(config.scheduler==='fixed')return initial;
    if(warmup&&i<warmup)return initial*(i+1)/warmup;
    const t=Math.max(0,(i-warmup)/Math.max(1,n-warmup-1)),factor=config.scheduler==='linear'?1-t:(1+Math.cos(Math.PI*t))/2;
    return minimum+(initial-minimum)*factor;
  });
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
    const wasOpen=this.advanced.querySelector('details')?.open||false;
    this.advanced.replaceChildren();
    const schedule=document.createElement('details'),summary=document.createElement('summary'),scheduleFields=document.createElement('div');
    schedule.id='trainingSchedule';schedule.className='training-schedule';schedule.open=wasOpen;summary.textContent='學習率策略';scheduleFields.className='training-form-grid';schedule.append(summary,scheduleFields);
    for(const field of fields.filter(field=>field.advanced)){
      const wrap=document.createElement('div'),label=document.createElement('label'),input=document.createElement(field.type==='select'?'select':'input');
      input.id=`trainingParam-${field.key}`;label.htmlFor=input.id;label.textContent=field.label;
      this.configure(input,field,draft[field.key]);wrap.append(label,input);
      if(field.description){const note=document.createElement('p');note.className='field-note';note.id=`${input.id}-hint`;note.textContent=field.description;input.setAttribute('aria-describedby',note.id);wrap.append(note)}
      (field.section==='schedule'?scheduleFields:this.advanced).append(wrap);
    }
    if(scheduleFields.childElementCount){const preview=document.createElement('div');preview.id='learningRatePreview';schedule.append(preview);this.advanced.append(schedule)}
    this.updateVisibility();
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
    if(!this.bound.has(input)){const change=()=>{this.remember();this.updateVisibility()};input.addEventListener('input',change);if(input.tagName==='SELECT')input.addEventListener('change',change);this.bound.add(input)}
  }
  updateVisibility(){
    const strategy=document.getElementById('trainingParam-scheduler');
    for(const field of this.fields)if(field.when){const input=document.getElementById(`trainingParam-${field.key}`);if(input)input.parentElement.hidden=!field.when.includes(strategy?.value)}
    const summary=this.advanced.querySelector('details summary');if(summary)summary.textContent=`學習率策略 · ${strategy?.selectedOptions[0]?.textContent||''}`;
    const preview=document.getElementById('learningRatePreview');if(!preview)return;
    const config=Object.fromEntries(this.inputs().map(input=>[input.dataset.trainingParam,input.value]));const rates=plannedRates(config);preview.replaceChildren();
    const note=document.createElement('p');note.className='field-note';note.textContent=config.scheduler==='plateau'?'依 Validation 分數決定下降時機，無預定曲線。':this.engine?.component==='ultralytics'?'排程趨勢示意；套件逐步暖身，實際值以訓練紀錄為準。':'預定每輪學習率；正式訓練另記錄實際使用值。';preview.append(note);
    if(!rates.length)return;
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 500 150');svg.setAttribute('role','img');svg.setAttribute('aria-label',`預定學習率，${rates.length} 輪，起始 ${rates[0]}，最後 ${rates.at(-1)}`);svg.style.width='100%';
    const part=(name,attrs,text)=>{const n=document.createElementNS(svg.namespaceURI,name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;svg.append(n);return n};const max=Math.max(...rates);
    part('path',{d:'M72 12 V116 H484',fill:'none',stroke:'#66818e'});
    part('polyline',{points:rates.map((v,i)=>`${72+i/Math.max(1,rates.length-1)*412},${116-v/max*98}`).join(' '),fill:'none',stroke:'#54cfb7','stroke-width':2});
    for(const[x,y,text]of [[2,22,max.toExponential(1)],[48,118,'0'],[72,138,'1'],[450,138,String(rates.length)],[252,138,'Epoch']])part('text',{x,y,fill:'#b9d0db','font-size':12},text);
    preview.append(svg);
  }
  collect(){
    const values=Object.fromEntries(this.inputs().filter(input=>!input.disabled).map(input=>[input.dataset.trainingParam,input.value]));
    return parameterPayload(this.fields,values);
  }
}
