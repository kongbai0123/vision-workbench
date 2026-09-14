import {createTrainingCharts, metricDescriptors, defaultMetricKeys, runAppearance, comparisonWarnings} from './training-charts.mjs';

const el=(tag,text,className)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n};
const action=(text,fn,className='secondary')=>{const n=el('button',text,className);n.type='button';n.onclick=fn;return n};
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const fmt=value=>finite(value)?value.toFixed(4):'—';
const active=new Set(['queued','preparing','running','stopping']);
const statusName=value=>({queued:'等待中',preparing:'準備資料',running:'訓練中',stopping:'正在停止',stopped:'已停止',completed:'已完成',failed:'失敗'}[value]||value);
const evaluationNames={mean_iou:'Mask IoU',box_mean_iou:'Box IoU',mean_dice:'Dice',accuracy:'Accuracy',macro_f1:'Macro F1',macro_recall:'Macro Recall',mask_map50_95:'Mask mAP50–95',mask_map50:'Mask mAP50',box_map50_95:'Box mAP50–95',box_map50:'Box mAP50',recall_50:'Recall@0.5'};

export class TrainingMonitor {
  constructor({loadReport,onSelect,onModel}) {
    this.loadReport=loadReport;this.onSelect=onSelect;this.onModel=onModel;
    this.root=document.getElementById('trainingRunDetail');this.dialog=document.getElementById('trainingHistoryDialog');
    this.openButton=document.getElementById('openTrainingHistory');
    this.openButton.onclick=()=>{this.renderHistory();this.dialog.showModal()};
    document.getElementById('closeTrainingHistory').onclick=()=>this.dialog.close();
    this.dialog.addEventListener('click',e=>{if(e.target===this.dialog){const r=this.dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)this.dialog.close()}});
    document.getElementById('trainingViewSingle').onclick=()=>{this.mode='single';this.refresh()};
    document.getElementById('trainingViewCompare').onclick=()=>{
      this.mode='compare';if(!this.compared.length){const first=this.runs.find(r=>r.run_id===this.selected)||this.runs[0];this.compared=this.runs.filter(r=>r.dataset_version_id===first?.dataset_version_id).slice(0,2).map(r=>r.run_id)}this.refresh();
    };
    this.reset();
  }
  reset(){
    this.chart?.destroy();this.chart=null;this.generation=(this.generation||0)+1;
    this.projectId=null;this.overview=null;this.runs=[];this.selected=null;this.mode='single';this.compared=[];
    this.hidden=new Set();this.reports=new Map();this.errors=new Map();this.domains=new Map();this.metricChoices=new Map();this.detailOpen=new Map();this.appearances=new Map();this.interaction={};
    if(this.dialog.open)this.dialog.close();
  }
  update(overview,projectId,selected){
    if(this.projectId!==projectId){this.reset();this.projectId=projectId}
    this.overview=overview;this.runs=overview.runs||[];
    this.selected=this.runs.some(r=>r.run_id===selected)?selected:this.runs[0]?.run_id;
    this.compared=this.compared.filter(id=>this.runs.some(r=>r.run_id===id));
    return this.refresh();
  }
  selectedRuns(){
    const ids=this.mode==='compare'?this.compared:[this.selected];
    const used=new Set();
    return ids.map(id=>this.runs.find(r=>r.run_id===id)).filter(Boolean).map(run=>{
      const dataset=this.overview?.datasets?.find(d=>d.id===run.dataset_version_id);
      let appearanceIndex=this.appearances.get(run.run_id)??0;while(used.has(appearanceIndex))appearanceIndex=(appearanceIndex+1)%6;used.add(appearanceIndex);this.appearances.set(run.run_id,appearanceIndex);
      const metricSplit=dataset?.splits?(dataset.splits.val?'val':'test'):(run.validation_split||run.evaluation?.validation?.split||'unknown');
      return {...run,appearanceIndex,metricSplit,datasetHash:dataset?.manifest_sha256};
    });
  }
  async refresh(){
    const token=++this.generation,project=this.projectId,runs=this.selectedRuns();
    this.render();
    const results=await Promise.allSettled(runs.map(run=>this.loadReport(project,run.run_id)));
    if(token!==this.generation||project!==this.projectId)return;
    results.forEach((result,i)=>{const id=runs[i].run_id;if(result.status==='fulfilled'){this.reports.set(id,result.value);this.errors.delete(id)}else this.errors.set(id,result.reason?.message||'指標暫時無法載入')});
    this.render();
  }
  selectRun(id){this.selected=id;this.mode='single';this.onSelect(id);this.dialog.close();this.refresh()}
  toggleComparison(id){
    const index=this.compared.indexOf(id);
    if(index>=0)this.compared.splice(index,1);else{
      if(this.compared.length>=4)return;
      this.compared.push(id);this.hidden.delete(id);
    }
    this.mode='compare';this.refresh();
  }
  renderHistory(){
    const list=document.getElementById('trainingRuns');
    const focus=document.activeElement?.dataset;const focusedId=focus?.compareRun||focus?.selectRun;
    list.replaceChildren();
    if(!this.runs.length)list.append(el('p','尚無訓練紀錄。','empty-list'));
    const anchor=this.runs.find(r=>r.run_id===this.compared[0]);
    for(const run of this.runs){
      const row=el('div',undefined,'training-history-row');
      const open=action('',()=>this.selectRun(run.run_id),'run-row'+(run.run_id===this.selected?' active':''));open.dataset.selectRun=run.run_id;
      const top=el('div',undefined,'run-row-top');top.append(el('b',`${run.run_id} · ${run.engine_name}`),el('span',statusName(run.status),'run-status '+run.status));
      open.append(top,el('small',`${run.dataset_version_id} · ${new Date(run.created_at*1000||run.created_at).toLocaleString('zh-TW')}`));
      const included=this.compared.includes(run.run_id),add=action(included?'移出比較':'加入比較',()=>this.toggleComparison(run.run_id));add.dataset.compareRun=run.run_id;add.setAttribute('aria-pressed',String(included));
      const incompatible=anchor&&run.dataset_version_id!==anchor.dataset_version_id;
      add.disabled=!included&&(this.compared.length>=4||!!incompatible);
      add.title=incompatible?'比較使用相同資料版本；先移除已選 Run，再選擇另一個版本。':this.compared.length>=4&&!included?'最多同時比較 4 個 Run':'';
      row.append(open,add);list.append(row);
    }
    document.getElementById('trainingHistoryCount').textContent=`已選 ${this.compared.length} / 4 · 相同資料版本可比較`;
    if(this.dialog.open&&focusedId){const node=[...list.querySelectorAll('[data-compare-run],[data-select-run]')].find(n=>(focus.compareRun?n.dataset.compareRun:n.dataset.selectRun)===focusedId);node?.focus()}
  }
  render(){
    const focus=document.activeElement;const focusId=focus?.id,focusRun=focus?.dataset?.toggleRun,focusMetric=focus?.dataset?.metricKey;
    this.root.querySelectorAll('details[data-detail-key]').forEach(d=>this.detailOpen.set(d.dataset.detailKey,d.open));
    this.chart?.destroy();this.chart=null;this.root.replaceChildren();
    this.openButton.textContent=`執行紀錄 · ${this.runs.length}`;
    for(const mode of ['single','compare']){const b=document.getElementById(mode==='single'?'trainingViewSingle':'trainingViewCompare');b.setAttribute('aria-pressed',String(this.mode===mode));b.disabled=!this.runs.length}
    this.renderHistory();
    const runs=this.selectedRuns();
    if(!runs.length){const empty=el('div',undefined,'report-empty');empty.append(el('h3',this.mode==='compare'?'選擇要比較的 Run':'尚無訓練紀錄'),el('p',this.mode==='compare'?'在執行紀錄中加入最多 4 個相同資料版本的 Run。':'開始訓練後在此查看進度、指標與固定設定。'));this.root.append(empty);return}
    if(this.mode==='single')this.renderSingleSummary(runs[0]);else this.renderComparisonSummary(runs);
    for(const run of runs){if(this.errors.has(run.run_id))this.root.append(el('p',`${run.run_id}：${this.errors.get(run.run_id)}；${this.reports.has(run.run_id)?'顯示上次成功取得的指標。':'可按重新整理重試。'}`,'readiness-item warning'))}
    const descriptors=metricDescriptors(runs,this.reports),signature=this.mode+':'+runs.map(r=>r.run_id).join('|');
    const keys=this.metricChoices.get(signature)||new Set(defaultMetricKeys(descriptors));
    const options=el('div',undefined,'training-metric-options');options.id='trainingMetricOptions';
    if(descriptors.length){options.append(el('b','顯示圖表'));for(const metric of descriptors){const label=el('label'),check=el('input');check.type='checkbox';check.checked=keys.has(metric.key);check.dataset.metricKey=metric.key;check.onchange=()=>{const selected=new Set(keys);if(check.checked)selected.add(metric.key);else selected.delete(metric.key);this.metricChoices.set(signature,selected);this.render()};label.append(check,el('span',metric.label));options.append(label)}}
    this.root.append(options);
    const chosen=[...keys].filter(k=>descriptors.some(d=>d.key===k));
    const warnings=this.mode==='compare'?chosen.flatMap(key=>comparisonWarnings(runs.filter(run=>descriptors.find(d=>d.key===key)?.availableRunIds.includes(run.run_id)),[key])):[];
    for(const warning of new Set(warnings))this.root.append(el('p',warning,'readiness-item warning'));
    const plot=el('div');plot.id='trainingPlots';this.root.append(plot);
    const visible=new Set(runs.filter(r=>!this.hidden.has(r.run_id)||this.mode==='single').map(r=>r.run_id));
    this.chart=createTrainingCharts(plot,{runs,reports:this.reports,visibleRunIds:visible,comparison:this.mode==='compare',metricKeys:chosen,domains:this.domains,interaction:this.interaction});
    if(chosen.length)this.root.append(el('p','X 軸涵蓋完整 Run，曲線不會左右掃視；隱藏 Run 不改變座標。移到圖表、點選或使用方向鍵查看同一 Epoch 數值。','metric-chart-note'));
    this.renderEpochTable(runs,descriptors,visible);this.renderEvaluation(runs);this.renderConfig(runs);
    if(focusRun)[...this.root.querySelectorAll('[data-toggle-run]')].find(n=>n.dataset.toggleRun===focusRun)?.focus({preventScroll:true});
    else if(focusMetric)[...options.querySelectorAll('input')].find(n=>n.dataset.metricKey===focusMetric)?.focus({preventScroll:true});
    else if(focusId&&document.getElementById(focusId)!==focus)document.getElementById(focusId)?.focus({preventScroll:true});
  }
  evaluation(run){
    const dataset=this.overview?.datasets?.find(d=>d.id===run.dataset_version_id);
    const test=run.evaluation?.test,val=run.evaluation?.validation;
    const score=dataset?.splits?.test?test||val:val||test;
    const split=score?.split==='test'||(!score?.split&&dataset?.splits?.test&&score===test)?'Test':'Validation';
    return {score,split};
  }
  scoreEntries(run,score){
    if(!score)return [];
    const priorities=['accuracy','mask_map50_95','box_map50_95','box_mean_iou','mean_iou'];
    const primary=priorities.find(key=>finite(score[key]));
    const names={...evaluationNames,mean_iou:String(run.engine||run.config?.engine).includes('deeplab')?'mIoU':'Mask IoU'};
    return Object.entries(names).filter(([key])=>finite(score[key])&&!(key==='mean_iou'&&finite(score.box_mean_iou))).sort(([a],[b])=>(a===primary?-1:b===primary?1:0)).map(([key,label])=>({key,label,value:score[key]}));
  }
  renderSingleSummary(run){
    const heading=el('div',undefined,'run-progress-head'),copy=el('div');copy.append(el('span','TRAINING RUN','eyebrow'),el('h2',`${run.run_id} · ${statusName(run.status)}`));heading.append(copy);
    if(active.has(run.status))heading.append(el('strong',`${Math.round(run.progress||0)}%`));
    const hasModel=this.overview?.models?.some(model=>model.model_version_id===run.model_version_id);
    if(hasModel)heading.append(action('評估與模型',()=>this.onModel(run.model_version_id)));
    this.root.append(heading);
    if(active.has(run.status)){const p=el('progress');p.className='run-progress';p.max=100;p.value=Number(run.progress||0);p.setAttribute('aria-label',`${run.run_id} 訓練進度`);this.root.append(p)}
    if(run.message)this.root.append(el('p',run.message,'muted'));
    if(run.error)this.root.append(el('p',run.error,'readiness-item error'));
    const {score,split}=this.evaluation(run),best=this.scoreEntries(run,score)[0];
    const cards=el('div',undefined,'run-metrics');
    for(const [label,value] of [[best?`${split} · ${best.label}`:'評估指標',best?fmt(best.value):'等待評估'],[`${score?split:'評估'} 圖片`,score&&finite(score.images)?`${score.images} 張`:'—'],['模型版本',hasModel?run.model_version_id:'尚未產生']]){const card=el('div',undefined,'run-metric');card.append(el('span',label),el('b',value));cards.append(card)}this.root.append(cards);
    this.root.append(this.configRow(run));
  }
  configRow(run){
    const row=el('div',undefined,'run-config');
    for(const [label,value]of [['資料版本',run.dataset_version_id],['引擎',run.engine_name],['完成／總 Epoch',`${run.epoch??this.reports.get(run.run_id)?.metrics?.at(-1)?.epoch??'—'} / ${run.config?.epochs??'—'}`],['裝置設定',String(run.config?.device||'auto').toUpperCase()]]){const item=el('div');item.append(el('span',label),el('b',value));row.append(item)}return row;
  }
  renderComparisonSummary(runs){
    this.root.append(el('p',`${runs[0].dataset_version_id} · 曲線來源：${runs[0].metricSplit==='test'?'Test（未設定 Validation）':runs[0].metricSplit==='val'?'Validation':'評估分割未知'} · 點選 Run 可同步顯示／隱藏所有曲線`,'muted'));
    const list=el('div',undefined,'training-comparison-list');
    runs.forEach((run,index)=>{const card=el('section',undefined,'training-comparison-card'),shown=!this.hidden.has(run.run_id),appearance=runAppearance(run,index),head=el('div',undefined,'training-comparison-heading');
      const toggle=action('',()=>{if(shown)this.hidden.add(run.run_id);else this.hidden.delete(run.run_id);this.render()},'training-run-toggle');toggle.dataset.toggleRun=run.run_id;toggle.setAttribute('aria-pressed',String(shown));
      const swatch=el('i');swatch.style.borderColor=appearance.color;swatch.style.borderTopStyle=appearance.dash?'dashed':'solid';toggle.append(swatch,el('span',`${run.run_id} · ${shown?'已顯示':'已隱藏'}`));head.append(toggle,el('span',statusName(run.status),'run-status '+run.status));card.append(head,el('p',run.engine_name,'muted'));
      const {score,split}=this.evaluation(run),metric=this.scoreEntries(run,score)[0];card.append(el('p',metric?`${split} ${metric.label} ${fmt(metric.value)} · ${run.model_version_id||'尚無模型版本'}`:'等待評估結果','training-comparison-score'),el('small',`Epoch ${run.epoch??'—'} / ${run.config?.epochs??'—'}`));
      if(run.error)card.append(el('p',run.error,'readiness-item error'));card.classList.toggle('curve-hidden',!shown);list.append(card);
    });this.root.append(list);
  }
  details(title,key){const d=el('details',undefined,'training-monitor-details');d.dataset.detailKey=key;d.open=this.detailOpen.get(key)||false;d.append(el('summary',title));this.root.append(d);return d}
  table(parent,headers,rows){const wrap=el('div',undefined,'training-table-wrap'),table=el('table'),head=el('thead'),tr=el('tr');headers.forEach(h=>tr.append(el('th',h)));head.append(tr);table.append(head);const body=el('tbody');rows.forEach(values=>{const row=el('tr');values.forEach(v=>row.append(el('td',String(v??'—'))));body.append(row)});table.append(body);wrap.append(table);parent.append(wrap)}
  renderEpochTable(runs,descriptors,visible){
    const d=this.details('Epoch 數值明細','epochs'),rows=[];
    for(const run of runs.filter(r=>visible.has(r.run_id)))for(const row of this.reports.get(run.run_id)?.metrics||[])rows.push([run.run_id,row.epoch,...descriptors.map(metric=>fmt(row[metric.key]))]);
    if(rows.length)this.table(d,['Run','Epoch',...descriptors.map(m=>m.label)],rows);else d.append(el('p','沒有可顯示的 Epoch 數值。','muted'));
  }
  renderEvaluation(runs){
    const d=this.details('各類別與詳細評估','evaluation');
    for(const run of runs){const {score,split}=this.evaluation(run);d.append(el('h3',`${run.run_id} · ${split}`));if(!score){d.append(el('p','尚無評估結果。','muted'));continue}
      this.table(d,['指標','數值'],this.scoreEntries(run,score).map(m=>[m.label,fmt(m.value)]));
      const iou=score.per_class_iou||{},dice=score.per_class_dice||{};
      const labels=[...new Set([...Object.keys(iou),...Object.keys(dice)])];
      if(labels.length)this.table(d,['類別',finite(score.box_mean_iou)?'Box IoU':'IoU',...(Object.keys(dice).length?['Dice']:[])],labels.map(label=>[label,fmt(iou[label]),...(Object.keys(dice).length?[fmt(dice[label])]:[])]));
      if(score.per_class_recall)this.table(d,['類別','Recall'],Object.entries(score.per_class_recall).map(([label,value])=>[label,fmt(value)]));
      const matrix=score.confusion_matrix,classes=score.classes||this.overview?.datasets?.find(ds=>ds.id===run.dataset_version_id)?.classes;
      if(Array.isArray(matrix)&&matrix.every(Array.isArray)){d.append(el('p','混淆矩陣：列為真實類別，欄為預測類別。','muted'));this.table(d,['真實／預測',...(classes||matrix.map((_,i)=>String(i)))],matrix.map((row,i)=>[classes?.[i]??String(i),...row]))}
      if(score.per_class&&typeof score.per_class==='object'){const entries=Object.entries(score.per_class);this.table(d,['類別','Precision','Recall','F1'],entries.map(([label,value])=>[label,fmt(value.precision),fmt(value.recall),fmt(value.f1)]))}
    }
  }
  renderConfig(runs){const d=this.details('完整執行設定','config');for(const run of runs){d.append(el('h3',run.run_id),this.configRow(run));const p=el('pre',JSON.stringify(run.config||{},null,2),'training-config-json');d.append(p)}}
}
