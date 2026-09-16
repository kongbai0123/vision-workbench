import {createTrainingCharts, metricDescriptors, normalizedMetricRows, runAppearance, comparisonWarnings, formatMetric as fmt} from './training-charts.mjs';

const el=(tag,text,className)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n};
const action=(text,fn,className='secondary')=>{const n=el('button',text,className);n.type='button';n.onclick=fn;return n};
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const active=new Set(['queued','preparing','running','stopping']);
const statusName=value=>({queued:'等待中',preparing:'準備資料',running:'訓練中',stopping:'正在停止',stopped:'已停止',completed:'已完成',failed:'失敗'}[value]||value);
const evaluationNames={mean_iou:'Macro IoU',micro_iou:'Micro IoU',box_mean_iou:'Box IoU',mean_dice:'Dice',accuracy:'Accuracy',macro_f1:'Macro F1',macro_recall:'Macro Recall',mask_map50_95:'Mask mAP50–95',mask_map50:'Mask mAP50',box_map50_95:'Box mAP50–95',box_map50:'Box mAP50',precision_50:'Precision@0.5',recall_50:'Recall@0.5'};
const executionLabels={batch_size:'每次前向批次大小',gradient_accumulation:'梯度累積步數',effective_batch_size:'有效批次大小',optimizer_step_measurement:'更新次數量測方式',optimizer_attempts:'最佳化器累計更新嘗試',optimizer_steps:'最佳化器累計成功更新',optimizer_skipped_updates:'AMP 累計跳過更新'};
const chartGroups=[['performance','效果指標'],['loss','損失曲線'],['learning','學習率'],['all','全部圖表']];

export function chartGroupForMetric(key){
  if(key==='train/learning_rate'||key.startsWith('lr/'))return 'learning';
  if(key.endsWith('/loss')||key.endsWith('_loss'))return 'loss';
  return 'performance';
}

export function runAuditSections(run,report){
  const recorded=report?.run?.run_id===run.run_id?report.run:run,sections=[];
  const initialization=recorded.initialization||run.initialization;
  if(initialization){
    const mode={pretrained:'預訓練權重',random:'從零開始（隨機初始化）',scratch:'從零開始（隨機初始化）'}[initialization.mode]||initialization.mode;
    const rows=[['初始化方式',mode],['權重來源',initialization.source],['權重檔案',initialization.weights_path],['權重 SHA-256',initialization.weights_sha256]].filter(([,value])=>value!==undefined&&value!==null&&value!=='');
    if(rows.length)sections.push({title:'模型初始化',rows});
  }
  const execution=recorded.execution||run.execution;
  if(execution){
    const rows=Object.entries(executionLabels).filter(([key])=>execution[key]!==undefined&&execution[key]!==null).map(([key,label])=>[label,key==='optimizer_step_measurement'?({post_step_hook:'每次實際權重更新後計數',unavailable:'未提供；不以 Batch 數估算'}[execution[key]]||execution[key]):execution[key]]);
    if(rows.length)sections.push({title:'實際執行資訊',rows});
  }
  const protocol=recorded.evaluation?.protocol||recorded.evaluation_protocol||run.evaluation?.protocol;
  if(protocol){
    const checkpoint={best_validation:'Validation 最佳權重',final_epoch:'最後一輪權重',validation_selected_thresholds:'Validation 選出的門檻'}[protocol.selection_checkpoint]||protocol.selection_checkpoint;
    const independent=protocol.test_independent_sources===true?'是':protocol.test_independent_sources===false?'否（僅流程驗證）':'未確認';
    const rows=[['選模資料','Validation'],['評估權重',checkpoint],['獨立 Test',protocol.test_present?'已執行':'未執行'],['Test 來源獨立',protocol.test_present?independent:'不適用']];
    if(protocol.test_present&&protocol.test_source_overlap_groups?.length)rows.push(['跨集合來源群組',protocol.test_source_overlap_groups.join('、')]);
    sections.push({title:'評估來源',rows});
  }
  return sections;
}

export class TrainingMonitor {
  constructor({loadReport,onSelect,onModel}) {
    this.loadReport=loadReport;this.onSelect=onSelect;this.onModel=onModel;
    this.root=document.getElementById('trainingRunDetail');this.dialog=document.getElementById('trainingHistoryDialog');
    this.workspace=document.getElementById('trainingMonitorWorkspace');this.fullscreenButton=document.getElementById('toggleTrainingFullscreen');
    this.openButton=document.getElementById('openTrainingHistory');
    this.openButton.onclick=()=>{this.renderHistory();this.dialog.showModal()};
    document.getElementById('closeTrainingHistory').onclick=()=>this.dialog.close();
    this.dialog.addEventListener('click',e=>{if(e.target===this.dialog){const r=this.dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)this.dialog.close()}});
    document.getElementById('trainingViewSingle').onclick=()=>{this.mode='single';this.refresh()};
    document.getElementById('trainingViewCompare').onclick=()=>this.beginComparison();
    this.fullscreenButton.onclick=()=>this.toggleFullscreen();
    document.addEventListener('keydown',event=>{if(event.key==='Escape'&&this.workspace.classList.contains('expanded'))this.toggleFullscreen(false)});
    this.reset();
  }
  reset(){
    this.chart?.destroy();this.chart=null;this.generation=(this.generation||0)+1;
    this.projectId=null;this.overview=null;this.runs=[];this.selected=null;this.mode='single';this.compared=[];this.chartGroup='performance';
    this.comparisonDataset=null;this.comparisonInitialized=false;
    this.reports=new Map();this.errors=new Map();this.domains=new Map();this.detailOpen=new Map();this.scrollPositions=new Map();this.appearances=new Map();this.interaction={};
    if(this.dialog.open)this.dialog.close();
  }
  toggleFullscreen(force){
    const expanded=force??!this.workspace.classList.contains('expanded');
    this.workspace.classList.toggle('expanded',expanded);document.body.classList.toggle('training-monitor-expanded',expanded);
    this.fullscreenButton.setAttribute('aria-pressed',String(expanded));this.fullscreenButton.textContent=expanded?'結束全螢幕':'全螢幕檢視';
    requestAnimationFrame(()=>window.dispatchEvent(new Event('resize')));
  }
  update(overview,projectId,selected){
    if(this.projectId!==projectId){this.reset();this.projectId=projectId}
    this.overview=overview;this.runs=overview.runs||[];
    this.selected=this.runs.some(r=>r.run_id===selected)?selected:this.runs[0]?.run_id;
    this.compared=this.compared.filter(id=>this.comparisonCandidates().some(r=>r.run_id===id));
    return this.refresh();
  }
  enrichRuns(runs){
    return runs.map(run=>{
      const dataset=this.overview?.datasets?.find(d=>d.id===run.dataset_version_id);
      if(!this.appearances.has(run.run_id))this.appearances.set(run.run_id,this.appearances.size);
      const appearanceIndex=this.appearances.get(run.run_id);
      const metricSplit=dataset?.splits?(dataset.splits.val?'val':'test'):(run.validation_split||run.evaluation?.validation?.split||'unknown');
      return {...run,appearanceIndex,metricSplit,datasetHash:dataset?.manifest_sha256};
    });
  }
  comparisonCandidates(){
    const dataset=this.comparisonDataset||this.runs.find(r=>r.run_id===this.selected)?.dataset_version_id;
    const models=new Set((this.overview?.models||[]).map(model=>model.model_version_id));
    return this.enrichRuns(this.runs.filter(run=>run.status==='completed'&&run.dataset_version_id===dataset&&models.has(run.model_version_id))
      .sort((a,b)=>String(a.model_version_id).localeCompare(String(b.model_version_id),undefined,{numeric:true})));
  }
  selectedRuns(){
    if(this.mode==='compare')return this.comparisonCandidates().filter(run=>this.compared.includes(run.run_id));
    return this.enrichRuns(this.runs.filter(run=>run.run_id===this.selected));
  }
  beginComparison(){
    const selected=this.runs.find(run=>run.run_id===this.selected),dataset=selected?.dataset_version_id;
    if(!this.comparisonInitialized||this.comparisonDataset!==dataset){
      this.comparisonDataset=dataset;this.comparisonInitialized=true;
      const candidates=this.comparisonCandidates(),first=candidates.find(run=>run.run_id===this.selected)||candidates[0];
      const next=candidates.find(run=>run.run_id!==first?.run_id&&run.engine===first?.engine)||candidates.find(run=>run.run_id!==first?.run_id);
      this.compared=[first,next].filter(Boolean).map(run=>run.run_id);
    }
    this.mode='compare';this.refresh();
  }
  async refresh(){
    const token=++this.generation,project=this.projectId,runs=this.selectedRuns();
    const fetchRuns=this.mode==='compare'?[...runs,...this.comparisonCandidates().filter(run=>!this.reports.has(run.run_id)&&!runs.some(selected=>selected.run_id===run.run_id))]:runs;
    this.render();
    const results=await Promise.allSettled(fetchRuns.map(run=>this.loadReport(project,run.run_id)));
    if(token!==this.generation||project!==this.projectId)return;
    results.forEach((result,i)=>{const id=fetchRuns[i].run_id;if(result.status==='fulfilled'){this.reports.set(id,result.value);this.errors.delete(id)}else this.errors.set(id,result.reason?.message||'指標暫時無法載入')});
    this.render();
  }
  selectRun(id){this.selected=id;this.mode='single';this.onSelect(id);this.dialog.close();this.refresh()}
  toggleComparison(id,checked){
    if(!this.comparisonCandidates().some(run=>run.run_id===id))return;
    this.compared=this.compared.filter(runId=>runId!==id);
    if(checked)this.compared.push(id);
    this.mode='compare';this.refresh();
  }
  renderHistory(){
    const list=document.getElementById('trainingRuns');
    const focusedId=document.activeElement?.dataset?.selectRun;
    list.replaceChildren();
    if(!this.runs.length)list.append(el('p','尚無訓練紀錄。','empty-list'));
    for(const run of this.runs){
      const row=el('div',undefined,'training-history-row');
      const open=action('',()=>this.selectRun(run.run_id),'run-row'+(run.run_id===this.selected?' active':''));open.dataset.selectRun=run.run_id;
      const top=el('div',undefined,'run-row-top');top.append(el('b',`${run.run_id} · ${run.engine_name}`),el('span',statusName(run.status),'run-status '+run.status));
      open.append(top,el('small',`${run.dataset_version_id} · ${new Date(run.created_at*1000||run.created_at).toLocaleString('zh-TW')}`));
      row.append(open);list.append(row);
    }
    document.getElementById('trainingHistoryCount').textContent=`共 ${this.runs.length} 次執行 · 選擇一筆查看訓練內容`;
    if(this.dialog.open&&focusedId)[...list.querySelectorAll('[data-select-run]')].find(n=>n.dataset.selectRun===focusedId)?.focus({preventScroll:true});
  }
  render(){
    const focus=document.activeElement;const focusId=focus?.id,focusRun=focus?.dataset?.modelRun,focusScroll=focus?.dataset?.scrollKey;
    this.root.querySelectorAll('details[data-detail-key]').forEach(d=>this.detailOpen.set(d.dataset.detailKey,d.open));
    this.root.querySelectorAll('[data-scroll-key]').forEach(wrap=>this.rememberScroll(wrap));
    this.chart?.destroy();this.chart=null;this.root.replaceChildren();
    this.openButton.textContent=`執行紀錄 · ${this.runs.length}`;
    for(const mode of ['single','compare']){const b=document.getElementById(mode==='single'?'trainingViewSingle':'trainingViewCompare');b.setAttribute('aria-pressed',String(this.mode===mode));b.disabled=!this.runs.length}
    this.renderHistory();
    const runs=this.selectedRuns();
    if(this.mode==='compare')this.renderModelPicker();
    if(!runs.length){const empty=el('div',undefined,'report-empty');empty.append(el('h3',this.mode==='compare'?'選擇要比較的模型':'尚無訓練紀錄'),el('p',this.mode==='compare'?'勾選上方模型後，圖表、數值與評估內容會一起顯示。':'開始訓練後在此查看進度、指標與固定設定。'));this.root.append(empty);this.restoreView(focusId,focusRun,focusScroll);return}
    if(this.mode==='single')this.renderSingleSummary(runs[0]);else this.renderComparisonSummary(runs);
    for(const run of runs){if(this.errors.has(run.run_id))this.root.append(el('p',`${run.run_id}：${this.errors.get(run.run_id)}；${this.reports.has(run.run_id)?'顯示上次成功取得的指標。':'可按重新整理重試。'}`,'readiness-item warning'))}
    const descriptors=metricDescriptors(runs,this.reports);
    this.renderChartToolbar(descriptors);
    const chosen=descriptors.filter(metric=>this.chartGroup==='all'||chartGroupForMetric(metric.key)===this.chartGroup).map(metric=>metric.key);
    const warnings=this.mode==='compare'?chosen.flatMap(key=>comparisonWarnings(runs.filter(run=>descriptors.find(d=>d.key===key)?.availableRunIds.includes(run.run_id)),[key])):[];
    const uniqueWarnings=[...new Set(warnings)];
    if(uniqueWarnings.length){const notice=el('details',undefined,'training-alert-summary');notice.append(el('summary',`比較提示 · ${uniqueWarnings.length}`));uniqueWarnings.forEach(warning=>notice.append(el('p',warning,'readiness-item warning')));this.root.append(notice)}
    const plot=el('div');plot.id='trainingPlots';this.root.append(plot);
    this.chart=createTrainingCharts(plot,{runs,domainRuns:this.mode==='compare'?this.comparisonCandidates():runs,reports:this.reports,comparison:this.mode==='compare',metricKeys:chosen,domains:this.domains,interaction:this.interaction});
    if(chosen.length)this.root.append(el('p','X 軸涵蓋完整 Run；切換圖表群組不會隱藏原始數值。移到圖表、點選或使用方向鍵可同步查看同一 Epoch。','metric-chart-note'));
    this.renderEpochTable(runs,descriptors);
    const columns=el('div',undefined,'training-detail-columns');this.root.append(columns);
    this.renderEvaluation(runs,columns);this.renderConfig(runs,columns);
    this.restoreView(focusId,focusRun,focusScroll);
  }
  restoreView(focusId,focusRun,focusScroll){
    const restore=()=>this.root.querySelectorAll('[data-scroll-key]').forEach(wrap=>{const position=this.scrollPositions.get(wrap.dataset.scrollKey);if(position&&wrap.getClientRects().length){wrap.scrollTop=position.top;wrap.scrollLeft=position.left;wrap.restoredScroll={top:wrap.scrollTop,left:wrap.scrollLeft}}});
    restore();
    if(focusRun)[...this.root.querySelectorAll('[data-model-run]')].find(node=>node.dataset.modelRun===focusRun)?.focus({preventScroll:true});
    else if(focusScroll)[...this.root.querySelectorAll('[data-scroll-key]')].find(node=>node.dataset.scrollKey===focusScroll)?.focus({preventScroll:true});
    else if(focusId)document.getElementById(focusId)?.focus({preventScroll:true});
  }
  rememberScroll(wrap){
    if(!wrap.isConnected||!wrap.getClientRects().length)return;
    if(wrap.restoredScroll?.top===wrap.scrollTop&&wrap.restoredScroll?.left===wrap.scrollLeft)return;
    wrap.restoredScroll=null;
    this.scrollPositions.set(wrap.dataset.scrollKey,{top:wrap.scrollTop,left:wrap.scrollLeft});
  }
  renderChartToolbar(descriptors){
    const bar=el('div',undefined,'training-chart-toolbar'),copy=el('div');copy.append(el('span','METRIC VIEWS','eyebrow'),el('h3','訓練曲線'));
    const tabs=el('div',undefined,'training-chart-tabs');tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','圖表群組');
    for(const [key,label] of chartGroups){const count=key==='all'?descriptors.length:descriptors.filter(metric=>chartGroupForMetric(metric.key)===key).length,button=action(`${label} · ${count}`,()=>{this.chartGroup=key;this.render()},'secondary');button.dataset.chartGroup=key;button.setAttribute('role','tab');button.setAttribute('aria-selected',String(this.chartGroup===key));button.disabled=!count;tabs.append(button)}
    bar.append(copy,tabs);this.root.append(bar);
  }
  renderModelPicker(){
    const candidates=this.comparisonCandidates(),field=el('fieldset',undefined,'training-model-picker');field.id='trainingModelPicker';
    field.append(el('legend','比較模型'));
    field.append(el('p',`${this.comparisonDataset||'尚無資料版本'} · 勾選模型後，同步顯示其曲線、摘要與詳細內容。`,'muted'));
    const list=el('div',undefined,'training-model-choices');
    for(const run of candidates){
      const label=el('label',undefined,'training-model-choice'),check=el('input'),appearance=runAppearance(run);
      check.type='checkbox';check.checked=this.compared.includes(run.run_id);check.dataset.modelRun=run.run_id;
      check.onchange=()=>this.toggleComparison(run.run_id,check.checked);
      const swatch=el('i');swatch.style.borderColor=appearance.color;swatch.style.borderTopStyle=appearance.dash?'dashed':'solid';swatch.setAttribute('aria-hidden','true');
      const copy=el('span');copy.append(el('b',`${run.model_version_id} · ${run.run_id}`),el('span',run.engine_name));
      label.append(check,swatch,copy);list.append(label);
    }
    if(!candidates.length)list.append(el('p','此資料版本尚無已完成並保存的模型。可從執行紀錄選擇其他資料版本。','muted'));
    field.append(list);this.root.append(field);
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
    this.renderDataQuality(run,this.root);
    if(active.has(run.status)){const p=el('progress');p.className='run-progress';p.max=100;p.value=Number(run.progress||0);p.setAttribute('aria-label',`${run.run_id} 訓練進度`);this.root.append(p)}
    if(run.message)this.root.append(el('p',run.message,'muted'));
    if(run.error)this.root.append(el('p',run.error,'readiness-item error'));
    const {score,split}=this.evaluation(run),best=this.scoreEntries(run,score)[0],invalid=run.evaluation?.valid===false;
    const currentEpoch=run.epoch??this.reports.get(run.run_id)?.metrics?.at(-1)?.epoch??'—';
    const cards=el('div',undefined,'run-metrics training-summary-cards');
    for(const [label,value] of [['狀態／進度',active.has(run.status)?`${statusName(run.status)} · ${Math.round(run.progress||0)}%`:statusName(run.status)],[best?`${split} · ${best.label}`:'主要評估指標',invalid?'不可用':best?fmt(best.value):'等待評估'],['完成／總 Epoch',`${currentEpoch} / ${run.config?.epochs??'—'}`],['模型版本',hasModel?run.model_version_id:'尚未產生']]){const card=el('div',undefined,'run-metric');card.append(el('span',label),el('b',value));cards.append(card)}this.root.append(cards);if(invalid)this.root.append(el('p',run.evaluation.reason||'歷史評估資料不符合目前規範。','readiness-item error'));
    this.root.append(this.configRow(run));
  }
  configRow(run){
    const row=el('div',undefined,'run-config');
    for(const [label,value]of [['資料版本',run.dataset_version_id],['引擎',run.engine_name],['完成／總 Epoch',`${run.epoch??this.reports.get(run.run_id)?.metrics?.at(-1)?.epoch??'—'} / ${run.config?.epochs??'—'}`],['裝置設定',String(run.config?.device||'auto').toUpperCase()]]){const item=el('div');item.append(el('span',label),el('b',value));row.append(item)}return row;
  }
  renderComparisonSummary(runs){
    this.root.append(el('p',`${runs[0].dataset_version_id} · 已選 ${runs.length} 個模型 · 曲線來源：${runs[0].metricSplit==='test'?'Test（未設定 Validation）':runs[0].metricSplit==='val'?'Validation':'評估分割未知'}`,'muted'));
    const list=el('div',undefined,'training-comparison-list');
    runs.forEach((run,index)=>{const card=el('section',undefined,'training-comparison-card'),appearance=runAppearance(run,index),head=el('div',undefined,'training-comparison-heading');card.dataset.runId=run.run_id;
      const name=el('b',`${run.model_version_id} · ${run.run_id}`);name.style.color=appearance.color;
      head.append(name,el('span',statusName(run.status),'run-status '+run.status));card.append(head,el('p',run.engine_name,'muted'));
      this.renderDataQuality(run,card);
      const {score,split}=this.evaluation(run),metric=this.scoreEntries(run,score)[0];card.append(el('p',metric?`${split} ${metric.label} ${fmt(metric.value)} · ${run.model_version_id||'尚無模型版本'}`:'等待評估結果','training-comparison-score'),el('small',`Epoch ${run.epoch??'—'} / ${run.config?.epochs??'—'}`));
      if(run.error)card.append(el('p',run.error,'readiness-item error'));list.append(card);
    });this.root.append(list);
  }
  renderDataQuality(run,parent){
    const report=this.reports.get(run.run_id),quality=report?.run?.run_id===run.run_id?report.run.data_quality||run.data_quality:run.data_quality;
    if(quality?.purpose==='diagnostic')parent.append(el('p','流程驗證 · 同拍攝批次跨集合，非獨立泛化評估','readiness-item warning'));
  }
  details(title,key,parent=this.root){const d=el('details',undefined,'training-monitor-details');d.dataset.detailKey=key;d.open=this.detailOpen.get(key)||false;d.append(el('summary',title));d.addEventListener('toggle',()=>{if(d.isConnected){this.detailOpen.set(key,d.open);if(d.open)this.restoreView()}});parent.append(d);return d}
  table(parent,headers,rows,scrollKey){const wrap=el('div',undefined,'training-table-wrap'),table=el('table'),head=el('thead'),tr=el('tr');headers.forEach(h=>{const th=el('th',h);th.scope='col';tr.append(th)});head.append(tr);table.append(head);const body=el('tbody');rows.forEach(values=>{const row=el('tr');values.forEach(v=>row.append(el('td',String(v??'—'))));body.append(row)});table.append(body);wrap.append(table);if(scrollKey){wrap.dataset.scrollKey=scrollKey;wrap.addEventListener('scroll',()=>this.rememberScroll(wrap))}parent.append(wrap);return wrap}
  renderEpochTable(runs,descriptors){
    const d=this.details('Epoch 數值明細','epochs'),rows=[];
    const counts=[['train/optimizer_attempts_epoch','本輪更新嘗試'],['train/optimizer_steps_epoch','本輪成功更新'],['train/optimizer_skipped_epoch','本輪 AMP 跳過'],['train/optimizer_steps','累計成功更新']].filter(([key])=>runs.some(run=>normalizedMetricRows(this.reports.get(run.run_id)?.metrics||[]).some(row=>finite(row[key]))));
    for(const run of runs)for(const row of normalizedMetricRows(this.reports.get(run.run_id)?.metrics||[]))rows.push([run.run_id,row.epoch,...descriptors.map(metric=>fmt(row[metric.key])),...counts.map(([key])=>finite(row[key])?String(row[key]):'—')]);
    if(rows.length){d.append(el('p',`共 ${rows.length} 筆 · 一次顯示最多 10 列，捲動查看全部 Epoch。`,'muted'));const wrap=this.table(d,['Run','Epoch',...descriptors.map(m=>m.label),...counts.map(([,label])=>label)],rows,'epochs');wrap.classList.add('training-epoch-scroll');wrap.tabIndex=0;wrap.setAttribute('role','region');wrap.setAttribute('aria-label','Epoch 數值明細，可捲動查看全部資料')}
    else d.append(el('p','沒有可顯示的 Epoch 數值。','muted'));
  }
  renderEvaluation(runs,parent){
    const d=this.details('各類別與詳細評估','evaluation',parent);
    for(const run of runs){const {score,split}=this.evaluation(run);d.append(el('h3',`${run.run_id} · ${split}`));if(!score){d.append(el('p',run.evaluation?.valid===false?(run.evaluation.reason||'歷史評估資料不符合目前規範。'):'尚無評估結果。',run.evaluation?.valid===false?'readiness-item error':'muted'));continue}
      this.table(d,['指標','數值'],this.scoreEntries(run,score).map(m=>[m.label,fmt(m.value)]));
      const iou=score.per_class_iou||{},dice=score.per_class_dice||{};
      const labels=[...new Set([...Object.keys(iou),...Object.keys(dice)])];
      if(labels.length)this.table(d,['類別',finite(score.box_mean_iou)?'Box IoU':'IoU',...(Object.keys(dice).length?['Dice']:[])],labels.map(label=>[label,fmt(iou[label]),...(Object.keys(dice).length?[fmt(dice[label])]:[])]));
      if(score.per_class_recall)this.table(d,['類別','Recall'],Object.entries(score.per_class_recall).map(([label,value])=>[label,fmt(value)]));
      const matrix=score.confusion_matrix,classes=score.classes||this.overview?.datasets?.find(ds=>ds.id===run.dataset_version_id)?.classes;
      if(Array.isArray(matrix)&&matrix.every(Array.isArray)){d.append(el('p','混淆矩陣：列為真實類別，欄為預測類別。','muted'));this.table(d,['真實／預測',...(classes||matrix.map((_,i)=>String(i)))],matrix.map((row,i)=>[classes?.[i]??String(i),...row]))}
      if(score.per_class&&typeof score.per_class==='object'){
        const entries=Object.entries(score.per_class);
        if(entries.some(([,value])=>'precision' in value||'recall' in value||'f1' in value))this.table(d,['類別','樣本／像素','預測','TP','FP','FN','Precision','Recall','F1'],entries.map(([label,value])=>[label,value.support??value.ground_truth_pixels??'—',value.predictions??value.predicted_pixels??'—',value.tp??'—',value.fp??'—',value.fn??'—',fmt(value.precision),fmt(value.recall),fmt(value.f1)]));
      }
    }
  }
  renderConfig(runs,parent){
    const d=this.details('完整執行設定','config',parent);
    const labels={epochs:'訓練輪數',device:'裝置設定',seed:'隨機種子',image_size:'輸入影像尺寸',batch_size:'批次大小',learning_rate:'初始學習率',scheduler:'學習率策略',min_learning_rate:'最低學習率',warmup_epochs:'暖身輪數',lr_patience:'降率耐心輪數',lr_factor:'降率倍率',engine:'訓練模型',weight_decay:'權重衰減',momentum:'動量',optimizer:'最佳化器',threshold_min:'門檻搜尋下限',threshold_max:'門檻搜尋上限',patience:'提前停止耐心輪數',workers:'資料載入程序數',pretrained:'使用預訓練權重'};
    Object.assign(labels,{initialization:'模型初始權重',gradient_accumulation:'梯度累積批次數'});
    const entries=(value,path=[])=>{if(value&&typeof value==='object')return Object.entries(value).flatMap(([key,item])=>entries(item,[...path,key]));const shown=path.at(-1)==='initialization'?({pretrained:'預訓練權重微調',scratch:'從零開始（隨機初始化）'}[value]||value):value;return [[path.map(key=>labels[key]||key).join(' / '),shown===null?'未設定':typeof shown==='boolean'?(shown?'開啟':'關閉'):String(shown??'—')]]};
    for(const run of runs){
      d.append(el('h3',`${run.model_version_id||'尚未產生模型'} · ${run.run_id}`));
      labels.epochs=run.engine==='pixel_prototype_v1'?'門檻搜尋次數':'訓練輪數';
      const list=el('dl',undefined,'training-parameter-list');
      for(const [name,value]of [['資料版本',run.dataset_version_id],['訓練模型',run.engine_name],...entries(run.config||{})]){list.append(el('dt',name),el('dd',value))}
      d.append(list);
      for(const section of runAuditSections(run,this.reports.get(run.run_id))){
        d.append(el('h4',section.title));const observed=el('dl',undefined,'training-parameter-list');
        for(const [name,value]of section.rows)observed.append(el('dt',name),el('dd',String(value)));
        d.append(observed);
      }
    }
  }
}
