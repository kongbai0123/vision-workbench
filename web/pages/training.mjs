// Training and model pages share state through an explicit application context.
export function createTrainingPage(context) {
  const {$, state, toast, status, api, projectPath, number, stats, date, button, element, settingValue, editor, flushAllEdits, safe, switchStage, formDialog, renderAssetList, loadAsset, selectAsset, pollJob, augmentationProfile, TrainingParameters, TrainingMonitor} = context;
const activeRunStates=new Set(['queued','preparing','running','stopping']);
function invalidateYoloCompatibility(){state.yoloCompatibility=null;state.yoloCompatibilitySignature='';if(state.stage==='train'&&state.training)renderTraining()}
const trainingParameters=new TrainingParameters({preferredDevice:()=>settingValue('device'),onChange:invalidateYoloCompatibility});
const trainingMonitor=new TrainingMonitor({loadReport:(projectId,runId)=>api(`/api/projects/${projectId}/training-runs/${runId}/metrics`),onSelect:id=>{state.selectedRun=id},onModel:id=>{state.selectedModel=id;safe(()=>switchStage('models'))}});
function applyTrainingConfigTab(){
  const tab=state.trainingConfigTab||'basic',basic=document.querySelector('[data-training-pane="basic"]'),advanced=$('trainingAdvancedSection'),fields=$('trainingAdvancedFields');
  if(basic)basic.hidden=tab!=='basic';if(advanced)advanced.hidden=tab==='basic';
  document.querySelectorAll('[data-training-tab]').forEach(button=>button.setAttribute('aria-selected',String(button.dataset.trainingTab===tab)));
  if(fields&&tab!=='basic')for(const child of fields.children){const schedule=child.dataset.section==='schedule';child.style.display=tab==='schedule'?(schedule?'':'none'):(schedule?'none':'');}
  const schedule=document.getElementById('trainingSchedule');if(schedule&&tab==='schedule')schedule.open=true;
  const heading=advanced?.querySelector('.training-parameter-heading h3');if(heading)heading.textContent=tab==='schedule'?'學習率排程':'進階設定';
}
window.addEventListener('training-parameters-rendered',applyTrainingConfigTab);
let trainingLoadGeneration=0;
function trainingStatusName(value){return {queued:'等待中',preparing:'準備資料',running:'訓練中',stopping:'正在停止',stopped:'已停止',completed:'已完成',failed:'失敗'}[value]||value||'未知'}
function selectedDataset(){const id=$('trainingDataset')?.value||state.training?.datasets?.[0]?.id;return state.training?.datasets?.find(item=>item.id===id)||null}
function activeTrainingRun(){return state.training?.runs?.find(run=>activeRunStates.has(run.status))||null}
function trainingProjectIsCurrent(projectId){return state.project?.id===projectId}
async function loadTraining({quiet=false}={}) {
  if(!state.project)return;
  const projectId=state.project.id,generation=++trainingLoadGeneration;
  try{
    const overview=await api(projectPath('/training'));
    if(!trainingProjectIsCurrent(projectId)||generation!==trainingLoadGeneration)return;
    state.training=overview;
    if(!overview.runs?.some(run=>run.run_id===state.selectedRun))state.selectedRun=overview.runs?.[0]?.run_id||null;
    if(!overview.models?.some(model=>model.model_version_id===state.selectedModel))state.selectedModel=overview.models?.[0]?.model_version_id||null;
    await trainingMonitor.update(overview,projectId,state.selectedRun);
    if(!trainingProjectIsCurrent(projectId)||generation!==trainingLoadGeneration)return;
    updateBackgroundTraining();
    if(state.stage==='train')renderTraining();
    if(state.stage==='models')renderModels();
    scheduleTrainingPoll();
  }catch(error){
    if(!quiet)throw error;
    if(trainingProjectIsCurrent(projectId)&&generation===trainingLoadGeneration){updateBackgroundTraining(error.message);scheduleTrainingPoll()}
  }
}
function scheduleTrainingPoll(){
  clearTimeout(state.trainingTimer);state.trainingTimer=null;
  if(!state.project||!activeTrainingRun())return;
  const projectId=state.project.id;
  state.trainingTimer=setTimeout(()=>{if(trainingProjectIsCurrent(projectId))pollTrainingStatus(projectId)},900);
}
async function pollTrainingStatus(projectId){
  const generation=++trainingLoadGeneration;
  try{
    const status=await api(`/api/projects/${projectId}/training/status`);
    if(!trainingProjectIsCurrent(projectId)||generation!==trainingLoadGeneration)return;
    const previous=state.training?.runs||[];
    const finished=previous.some(run=>activeRunStates.has(run.status)&&!status.runs.some(next=>next.run_id===run.run_id&&activeRunStates.has(next.status)));
    if(finished){await loadTraining({quiet:true});return}
    state.training={...state.training,runs:status.runs};
    updateBackgroundTraining();
    if(state.stage==='train')await trainingMonitor.update(state.training,projectId,state.selectedRun);
  }catch(error){
    if(trainingProjectIsCurrent(projectId)&&generation===trainingLoadGeneration)updateBackgroundTraining(error.message);
  }finally{
    if(trainingProjectIsCurrent(projectId)&&generation===trainingLoadGeneration)scheduleTrainingPoll();
  }
}
function updateBackgroundTraining(error=''){
  const run=activeTrainingRun(),button=$('backgroundTraining');
  button.hidden=!run&&!error;
  if(error){$('backgroundTrainingText').textContent=`訓練狀態暫時無法更新：${error}`;$('backgroundTrainingProgress').removeAttribute('value');return}
  if(!run)return;
  const epoch=run.epoch?` · Epoch ${number(run.epoch)}/${number(run.config?.epochs)}`:'';
  const batch=run.batch&&run.batches_per_epoch?` · Batch ${number(run.batch)}/${number(run.batches_per_epoch)}`:'';
  const updates=Number(run.execution?.optimizer_steps);const step=Number.isFinite(updates)?` · 權重更新 ${number(updates)}`:'';
  $('backgroundTrainingText').textContent=`${run.run_id} ${trainingStatusName(run.status)}${epoch}${batch}${step}`;
  const progress=Number(run.progress);if(Number.isFinite(progress))$('backgroundTrainingProgress').value=progress;else $('backgroundTrainingProgress').removeAttribute('value');
}
function renderReadiness(report){
  const root=$('trainingReadiness');root.replaceChildren();
  const stats=report?.stats||{},splits=stats.splits||{};
  if(report?.ready)root.append(element('div',`檢查通過 · 已核准 ${number(stats.approved)} 張 · Train ${number(splits.train)} / Val ${number(splits.val)} / Test ${number(splits.test)}`,'readiness-item'));
  for(const item of report?.blockers||[]){const row=element('div',undefined,'readiness-item error');row.append(element('b',item.message),element('div',`處理方式：${item.action}`));root.append(row)}
  for(const item of report?.warnings||[]){const row=element('div',undefined,'readiness-item warning');row.append(element('b',item.message),element('div',item.action));if(item.code==='source_group_leak'){const fix=element('button','前往資料分割','text-button');fix.onclick=()=>safe(()=>switchStage('split'));row.append(fix)}root.append(row)}
}
function yoloCompatibilitySignature(dataset,config){return JSON.stringify([state.project?.id,dataset?.id,$('trainingEngine').value,config])}
function renderYoloCompatibility(){
  const panel=$('yoloCompatibilityPanel');if(!panel)return;
  const report=state.reviewYoloCompatibility,badge=$('yoloCompatibilityBadge'),root=$('yoloCompatibilityReport'),current=report?.project_revision===state.project?.revision;root.replaceChildren();
  badge.className='badge';
  if(state.reviewYoloCompatibilityLoading){badge.textContent='掃描中…';return}
  if(!report||!current){badge.textContent=report?'標註已變動':'尚未檢查';if(report)badge.classList.add('warning');return}
  const summary=report.summary||{};badge.textContent=report.compatible?'檢查通過':'需要處理';badge.classList.add(report.compatible?'approved':'error');
  const scope=report.review_scope||{};
  const message=report.compatible
    ?`專案 revision ${number(report.project_revision)} 已掃描 ${number(summary.assets_scanned)} 張（待審核 ${number(scope.pending)}／已核准 ${number(scope.approved)}）；${summary.pixels_repaired?`YOLO Run 副本可安全修補 ${number(summary.affected_assets)} 張中的 ${number(summary.holes_repaired)} 個微孔洞，共 ${number(summary.pixels_repaired)} px。`:'不需要相容修補。'} 原始標註不會變更。`
    :`專案 revision ${number(report.project_revision)} 已掃描 ${number(summary.assets_scanned)} 張；${number(summary.blocked_assets)} 張含有 YOLO Seg 無法安全表示的內容。可在此處修正，或改用能保留孔洞的模型。`;
  root.append(element('div',message,`yolo-compatibility-summary${report.compatible?'':' error'}`));
  const issues=[...(report.blockers||[]),...(report.repairs||[])];if(!issues.length)return;
  const list=element('div',undefined,'yolo-issue-list');
  for(const issue of issues){const row=element('div',undefined,'yolo-issue'),copy=element('div');copy.append(element('b',`${issue.name} · ${issue.label||'未命名標註'}`),element('p',issue.message));const locate=button('放大位置','text-button',()=>safe(()=>showYoloLocation(issue)));row.append(copy,locate);const coords=(issue.holes||[]).map(hole=>{const [x,y,w,h]=hole.bbox;return `X=${x}${w>1?`～${x+w-1}`:''}、Y=${y}${h>1?`～${y+h-1}`:''} · ${hole.pixels} px`}).join('；');if(coords)row.append(element('p',coords));list.append(row)}
  root.append(list);
}
async function checkReviewYoloCompatibility(){
  if(!state.project||state.reviewYoloCompatibilityLoading)return state.reviewYoloCompatibility;
  state.reviewYoloCompatibilityLoading=true;renderYoloCompatibility();$('checkYoloCompatibility').disabled=true;
  try{const report=await api(projectPath('/review-compatibility'),'POST',{config:{yolo_mask_policy:'repair_tiny_holes'}});state.reviewYoloCompatibility=report;renderYoloCompatibility();return report}
  finally{state.reviewYoloCompatibilityLoading=false;$('checkYoloCompatibility').disabled=false;renderYoloCompatibility()}
}
async function showYoloLocation(issue){
  const holes=issue.holes||[];if(!holes.length)throw Error('這筆問題沒有可定位的孔洞座標。');
  const body=element('div',undefined,'yolo-diagnostic'),lead=element('p',`${issue.name}／${issue.label||'未命名標註'}。每個孔洞獨立裁切與放大；切換編號不必關閉視窗。`);
  const canvas=document.createElement('canvas');canvas.width=760;canvas.height=420;canvas.className='yolo-location-canvas';canvas.dataset.ready='false';
  const statusLine=element('div','準備裁切預覽…','yolo-preview-status'),counter=element('b','','yolo-hole-counter');
  const previous=button('← 上一個','secondary'),next=button('下一個 →','secondary'),controls=element('div',undefined,'yolo-hole-controls');controls.append(previous,counter,next);
  const gallery=element('div',undefined,'yolo-hole-gallery');let active=0,request=0;
  const cropFor=hole=>{const [x,y,w,h]=hole.bbox,aspect=760/420,pad=Math.max(48,Math.ceil(Math.max(w,h)*.8));let cw=Math.max(192,w+pad*2),ch=Math.max(108,h+pad*2);if(cw/ch<aspect)cw=ch*aspect;else ch=cw/aspect;cw=Math.min(issue.width,cw);ch=Math.min(issue.height,ch);const left=Math.max(0,Math.min(issue.width-cw,x+w/2-cw/2)),top=Math.max(0,Math.min(issue.height-ch,y+h/2-ch/2));return [Math.floor(left),Math.floor(top),Math.max(1,Math.ceil(cw)),Math.max(1,Math.ceil(ch))]};
  const drawMarker=(context,hole,crop,index)=>{const [x,y,w,h]=hole.bbox,[sx,sy,sw,sh]=crop,scaleX=canvas.width/sw,scaleY=canvas.height/sh,cx=(x+w/2-sx)*scaleX,cy=(y+h/2-sy)*scaleY,dw=Math.max(18,w*scaleX),dh=Math.max(18,h*scaleY);context.fillStyle='#ff2f4266';context.strokeStyle='#ffedf0';context.lineWidth=3;context.fillRect(cx-dw/2,cy-dh/2,dw,dh);context.strokeRect(cx-dw/2,cy-dh/2,dw,dh);context.beginPath();context.moveTo(cx-28,cy);context.lineTo(cx+28,cy);context.moveTo(cx,cy-28);context.lineTo(cx,cy+28);context.stroke();context.fillStyle='#fff';context.font='bold 18px sans-serif';context.fillText(String(index+1),cx+14,cy-14)};
  const show=index=>{active=(index+holes.length)%holes.length;const token=++request,hole=holes[active],crop=cropFor(hole),[x,y,w,h]=hole.bbox,context=canvas.getContext('2d');canvas.dataset.ready='false';context.fillStyle='#080d11';context.fillRect(0,0,canvas.width,canvas.height);context.fillStyle='#a9bbc4';context.font='16px sans-serif';context.fillText('正在載入精確裁切位置…',24,44);statusLine.textContent=`孔洞 ${active+1}：X=${x}${w>1?`～${x+w-1}`:''}、Y=${y}${h>1?`～${y+h-1}`:''} · ${hole.pixels} px`;counter.textContent=`${active+1} / ${holes.length}`;previous.disabled=holes.length===1;next.disabled=holes.length===1;[...gallery.children].forEach((item,i)=>item.classList.toggle('active',i===active));const image=new Image();image.decoding='async';const timer=setTimeout(()=>{if(token===request)statusLine.textContent+=' · 載入時間較長，仍在等待裁切影像…'},2500);image.onload=()=>{clearTimeout(timer);if(token!==request)return;context.imageSmoothingEnabled=false;context.drawImage(image,0,0,canvas.width,canvas.height);drawMarker(context,hole,crop,active);canvas.dataset.ready='true';const following=holes[(active+1)%holes.length];if(following!==hole){const ahead=cropFor(following),preload=new Image();preload.src=`/api/projects/${state.project.id}/assets/${issue.asset_id}/image?crop=${ahead.join(',')}&max=760,420`}};image.onerror=()=>{clearTimeout(timer);if(token!==request)return;context.fillStyle='#080d11';context.fillRect(0,0,canvas.width,canvas.height);context.fillStyle='#ffb5b5';context.fillText('裁切預覽載入失敗；可重試或直接定位到編輯器。',24,44);canvas.dataset.ready='error'};image.src=`/api/projects/${state.project.id}/assets/${issue.asset_id}/image?crop=${crop.join(',')}&max=760,420`};
  holes.forEach((hole,index)=>{const [x,y,w,h]=hole.bbox,item=button('', 'yolo-hole-card',()=>show(index));item.append(element('b',`孔洞 ${index+1} · ${hole.pixels} px`),element('span',`X=${x}${w>1?`～${x+w-1}`:''} · Y=${y}${h>1?`～${y+h-1}`:''}`));gallery.append(item)});
  previous.onclick=()=>show(active-1);next.onclick=()=>show(active+1);
  const jump=button('在內建編輯器精確定位目前孔洞','primary',()=>safe(async()=>{const [x,y,w,h]=holes[active].bbox;$('cancelDialog').click();await selectAsset(issue.asset_id);await switchStage('annotate');requestAnimationFrame(()=>requestAnimationFrame(()=>editor.focusPoint(x+w/2,y+h/2)))}));
  body.append(lead,canvas,statusLine,controls,gallery,jump);const showing=formDialog({title:'YOLO Seg 孔洞位置',body,eyebrow:'MASK DIAGNOSTIC',wide:true});show(0);await showing;
}
async function checkYoloCompatibility(){
  const dataset=selectedDataset();if(!dataset)throw Error('請先建立或選擇訓練資料版本。');
  const config=trainingParameters.collect(),signature=yoloCompatibilitySignature(dataset,config);
  const report=await api(projectPath('/training-compatibility'),'POST',{dataset_version_id:dataset.id,config:{engine:$('trainingEngine').value,...config}});state.yoloCompatibility=report;state.yoloCompatibilitySignature=signature;return report
}
function renderTraining(){
  if(!state.training)return;
  const {readiness,datasets=[],runs=[],capabilities}=state.training,active=activeTrainingRun();
  const datasetSelect=$('trainingDataset'),remember=datasetSelect.value;datasetSelect.replaceChildren();
  if(!datasets.length)datasetSelect.append(new Option('尚無資料版本',''));
  for(const item of datasets)datasetSelect.append(new Option(`${item.id}${item.data_quality?.label?' · '+item.data_quality.label:item.data_quality?.purpose==='diagnostic'?' · 流程驗證':''} · ${number(item.asset_count)} 張 · ${date(item.created_at)}`,item.id));
  datasetSelect.value=datasets.some(item=>item.id===remember)?remember:datasets[0]?.id||'';
  const dataset=selectedDataset();$('trainingDatasetCurrent').textContent=dataset?.id||'尚未建立';
  renderReadiness(dataset?.readiness||readiness);
  const purposeMessages={formal:'正式獨立來源評估：Train／Validation／Test 依來源隔離。',reviewed_independent:'人工確認獨立：依圖片進行多類別平衡；請依類別覆蓋解讀指標。',experimental:'寬鬆實驗：來源可能跨集合，分數不代表新場景泛化能力。',diagnostic:'流程驗證：同拍攝批次跨集合，分數不代表新場景泛化能力。',all_train:'全資料最終訓練：沒有獨立 Validation／Test，不提供可比較的泛化評估。'};
  if(purposeMessages[dataset?.data_quality?.purpose])$('trainingReadiness').append(element('div',purposeMessages[dataset.data_quality.purpose],['experimental','diagnostic','all_train'].includes(dataset.data_quality.purpose)?'readiness-item warning':'readiness-item'));
  if(dataset&&!readiness.ready)$('trainingReadiness').append(element('div','目前專案的分割尚需處理；建立新版本前請前往資料分割。此處檢查結果屬於所選固定版本。','readiness-item warning'));
  $('trainingDatasetHint').textContent=dataset?`${number(dataset.asset_count)} 張 · Train ${number(dataset.splits.train)} / Val ${number(dataset.splits.val)} / Test ${number(dataset.splits.test)}`:'只會固定已核准且完成分割的資料。';
  if(dataset&&dataset.project_revision!==readiness.project_revision)$('trainingDatasetHint').textContent+=' · 目前專案已變動，建立新版本才會套用。';
  $('prepareAutoSplit').hidden=false;$('prepareAutoSplit').disabled=false;
  $('createDatasetVersion').disabled=!readiness.ready||!!active;
  const engineSelect=$('trainingEngine'),engineRemember=engineSelect.value;engineSelect.replaceChildren();
  let currentTask='';for(const engine of capabilities.engines||[]){
    const task=engine.task_name||capabilities.tasks?.[engine.task]||engine.task;if(task!==currentTask){const group=document.createElement('optgroup');group.label=task;engineSelect.append(group);currentTask=task}
    const suffix=engine.train?' · 可使用':engine.integration==='ready'?' · 尚未安裝':' · 整合開發中';engineSelect.lastElementChild.append(new Option(`${engine.name}${suffix}`,engine.key));
  }
  if([...engineSelect.options].some(option=>option.value===engineRemember))engineSelect.value=engineRemember;
  else engineSelect.value=(capabilities.engines||[]).find(engine=>engine.train)?.key||(capabilities.engines||[])[0]?.key||'';
  const engine=(capabilities.engines||[]).find(item=>item.key===engineSelect.value);$('trainingEngineHint').textContent=engine?.description||'沒有可用的訓練引擎。';
  trainingParameters.render(engine);
  $('trainingTaskLabel').textContent=engine?`${engine.task_name||capabilities.tasks?.[engine.task]||engine.task} · ${engine.annotation||'依模型需求'}`:'尚未選擇模型';
  $('trainingActivationLabel').textContent=engine?.family?.includes('YOLO')?'SiLU／由 YOLO 架構固定':engine?.component==='builtin'?'不使用神經網路啟用函數':'ReLU／SiLU 等由模型架構固定';
  const notice=$('trainingModelNotice');notice.hidden=!!engine?.train;if(!notice.hidden){$('trainingModelNoticeTitle').textContent=engine?.integration==='ready'?'此模型尚未準備完成':'此模型已納入開發待辦';$('trainingModelNoticeText').textContent=engine?.unavailable_reason||'完成必要元件與 Workbench adapter 後即可使用。'}
  const summary=$('trainingSummary');summary.replaceChildren();
  const deviceName=engine?.component==='builtin'?'CPU':({auto:'自動選擇',cuda:'NVIDIA CUDA',cpu:'CPU'}[$('trainingDevice').value]||'自動選擇');
  let config={};try{config=trainingParameters.collect()}catch{}
  const trainCount=Number(dataset?.splits?.train||0),trainingEvents=Number(dataset?.training_events?.events||trainCount),batchSize=Number(config.batch_size||1),accumulation=Number(config.gradient_accumulation||1),batches=trainingEvents?Math.ceil(trainingEvents/batchSize):0;
  for(const [label,value]of [['資料版本',dataset?.id||'—'],['資料增強',dataset?.augmentation?.preset||'—'],['圖片',dataset?`${number(dataset.asset_count)} 張`:'—'],['每輪訓練事件',dataset?`${number(trainingEvents)}（Train 原圖 ${number(trainCount)}）`:'—'],['任務',engine?.task_name||capabilities.tasks?.[engine?.task]||'—'],['引擎',engine?.name||'—'],['每輪 Batch',batches?`${number(batches)}（${number(trainingEvents)} ÷ ${number(batchSize)}）`:'—'],['有效批次',config.batch_size?`${number(batchSize*accumulation)} 張`:'—'],['裝置',`${deviceName} · 獨立程序`]]){summary.append(element('dt',label),element('dd',value))}
  applyTrainingConfigTab();
  $('startTraining').disabled=!dataset||dataset.readiness?.ready===false||!engine?.train||!engine?.parameters?.length||!!active;
  if(engine?.key?.endsWith('_seg')&&state.yoloCompatibility){const signature=yoloCompatibilitySignature(dataset,trainingParameters.collect());if(state.yoloCompatibilitySignature===signature&&!state.yoloCompatibility.compatible)$('startTraining').disabled=true}
  $('stopTraining').hidden=!active;$('startTraining').hidden=!!active;
  $('trainingActionHint').textContent=active?`${active.run_id} ${trainingStatusName(active.status)}；切換頁面後仍在背景執行。`:!engine?.train?(engine?.unavailable_reason||'請先到設定中心準備模型。'):dataset?'開始時會固定目前顯示的資料、引擎與參數。':'先建立或選擇固定資料版本。';
  trainingMonitor.render();
}
async function createDatasetVersion(){await flushAllEdits();const created=await api(projectPath('/dataset-versions'),'POST',{augmentation:augmentationProfile()});state.yoloCompatibility=null;state.yoloCompatibilitySignature='';await loadTraining();$('trainingDataset').value=created.id;renderTraining();toast(`已建立固定訓練資料 ${created.id}，並固定資料增強配方。`);return created}
let pendingTrainingRequest=null;
async function startTrainingRun(){
  const dataset=selectedDataset();if(!dataset)throw Error('請先建立或選擇訓練資料版本。');
  if(dataset.readiness?.ready===false)throw Error(dataset.readiness.blockers.map(item=>item.message).join('；'));
  const config=trainingParameters.collect();
  if($('trainingEngine').value.endsWith('_seg')){const signature=yoloCompatibilitySignature(dataset,config),report=state.yoloCompatibilitySignature===signature?state.yoloCompatibility:await checkYoloCompatibility();if(!report?.compatible)throw Error('YOLO Seg 固定資料版本複查未通過；請回到「資料審核」查看標出的圖片與孔洞位置。')}
  const payload={dataset_version_id:dataset.id,config:{engine:$('trainingEngine').value,...config}},signature=JSON.stringify([state.project.id,payload]);
  if(pendingTrainingRequest?.signature!==signature)pendingTrainingRequest={signature,id:crypto.randomUUID()};
  const run=await api(projectPath('/training-runs'),'POST',{...payload,request_id:pendingTrainingRequest.id});
  pendingTrainingRequest=null;
  state.selectedRun=run.run_id;trainingMonitor.mode='single';await loadTraining();toast(`${run.run_id} 已啟動；可以切換到其他工作區。`)
}
async function stopTrainingRun(){const run=activeTrainingRun();if(!run)return;await api(projectPath(`/training-runs/${run.run_id}/stop`),'POST',{});await loadTraining();toast(`${run.run_id} 正在安全停止。`)}
function renderModels(){
  if(!state.training)return;const {models=[],predictions=[]}=state.training;
  const list=$('modelList');list.replaceChildren();if(!models.length)list.append(element('p','尚無模型版本。','empty-list'));
  for(const model of models){const definition=state.training?.capabilities?.engines?.find(item=>item.key===model.engine),row=button('','model-row'+(model.model_version_id===state.selectedModel?' active':''),()=>{state.selectedModel=model.model_version_id;renderModels()});const top=element('div',undefined,'model-row-top');top.append(element('b',`${model.model_version_id} · ${model.engine_name}`),element('span',definition?.predict?'可預標註':'訓練／匯出','run-status'));row.append(top,element('small',`${model.dataset_version_id} · ${date(model.created_at)}`));list.append(row)}
  renderModelDetail(models.find(model=>model.model_version_id===state.selectedModel)||models[0]);
  const selected=models.find(model=>model.model_version_id===state.selectedModel)||models[0],selectedDefinition=state.training?.capabilities?.engines?.find(item=>item.key===selected?.engine);$('generatePredictions').disabled=!selected||!selectedDefinition?.predict;$('generatePredictions').title=selected&&!selectedDefinition?.predict?'影像分類結果不會轉成整張圖片的 Bounding Box':'';
  const predictionsRoot=$('predictionList');predictionsRoot.replaceChildren();
  if(!predictions.length)predictionsRoot.append(element('p','尚無候選標註。','empty-list'));
  for(const candidate of predictions){const pending=candidate.assets.filter(asset=>asset.status==='candidate'),accepted=candidate.assets.filter(asset=>asset.status==='accepted'),empty=candidate.assets.filter(asset=>asset.status==='empty');const row=element('article',undefined,'prediction-row');row.append(element('b',`${candidate.model_version_id} · ${date(candidate.created_at)}`),element('small',`候選 ${number(pending.length)} 張 · 已接受 ${number(accepted.length)} 張${empty.length?` · 未偵測 ${number(empty.length)} 張`:''}`));if(pending.length){const accept=button('接受候選並送審','secondary',()=>safe(()=>acceptPredictions(candidate.candidate_id)));row.append(accept)}predictionsRoot.append(row)}
}
function renderModelDetail(model){const root=$('modelDetail');root.replaceChildren();if(!model){const empty=element('div',undefined,'report-empty');empty.append(element('span','◇'),element('h3','尚無可用模型'),element('p','完成一次訓練與評估後，模型會連同資料來源出現在這裡。'));root.append(empty);return}
  const definition=state.training?.capabilities?.engines?.find(item=>item.key===model.engine),title=element('div',undefined,'model-title'),copy=element('div');copy.append(element('span','MODEL VERSION','eyebrow'),element('h2',`${model.model_version_id} · ${model.engine_name}`));title.append(copy,element('span',definition?.task_name||'模型版本','badge approved'));root.append(title);
  const result=model.test||model.validation||{},split=model.test?'Test':'Validation',invalid=model.evaluation_reassessment?.valid===false,values=invalid?[['評估結果','不可用'],['原因',model.evaluation_reassessment.reason||'資料或評估流程不符合要求']]:result.accuracy!==undefined?[[`${split} Accuracy`,Number(result.accuracy).toFixed(3)],[`${split} Macro F1`,Number(result.macro_f1).toFixed(3)],[`${split} Macro Recall`,Number(result.macro_recall).toFixed(3)]]:result.mask_map50_95!==undefined?[[`${split} Mask mAP50–95`,Number(result.mask_map50_95).toFixed(3)],[`${split} Mask mAP50`,Number(result.mask_map50).toFixed(3)]]:result.box_map50_95!==undefined?[[`${split} Box mAP50–95`,Number(result.box_map50_95).toFixed(3)],[`${split} Box mAP50`,Number(result.box_map50).toFixed(3)]]:result.box_map50!==undefined?[[`${split} Box mAP50`,Number(result.box_map50).toFixed(3)],[`${split} Recall@0.5`,Number(result.recall_50).toFixed(3)]]:result.mean_dice!==undefined?[[`${split} mIoU`,Number(result.mean_iou).toFixed(3)],[`${split} Dice`,Number(result.mean_dice).toFixed(3)]]:[[`${split} Mean IoU`,Number(result.mean_iou||0).toFixed(3)]];if(!invalid){values.push([`${split} 圖片`,number(result.images)],['類別數',number(model.classes?.length)]);if(!model.test)values.push(['獨立 Test','尚未執行'])}const score=element('div',undefined,'run-metrics');for(const [label,value]of values){const item=element('div',undefined,'run-metric');item.append(element('span',label),element('b',value));score.append(item)}root.append(score);
  const abilities=['訓練','評估',...(definition?.predict?['預標註']:[]),'匯出'].join('／'),facts=element('dl',undefined,'model-facts');for(const [label,value]of [['來源 Run',model.run_id],['訓練資料',model.dataset_version_id],['類別',(model.classes||[]).join('、')],['模型能力',abilities],['建立時間',date(model.created_at)]])facts.append(element('dt',label),element('dd',value));root.append(facts,element('p',model.task==='image_classification'?'分類結果保留為圖片層級評估，不會建立覆蓋整張圖片的 Bounding Box。':'模型與資料版本、類別映射和評估結果一起保存；專案後續修改不會回寫此模型。','readiness-item'));
  if(model.yolo_compatibility){const report=model.yolo_compatibility,summary=report.summary||{},audit=element('details',undefined,'training-monitor-details'),heading=document.createElement('summary');heading.textContent=`YOLO Seg 相容稽核 · 修補 ${number(summary.pixels_repaired)} px`;const copy=element('p',`已掃描 ${number(summary.assets_scanned)} 張；${number(summary.affected_assets)} 張的 Run 副本修補 ${number(summary.holes_repaired)} 個微小孔洞。原始標註與固定資料版本未變更，模型內部評估使用相容副本。`,'readiness-item');audit.append(heading,copy);for(const issue of report.repairs||[]){const row=element('div',undefined,'yolo-issue');row.append(element('b',`${issue.name} · ${issue.label||'未命名標註'}`),button('放大位置','text-button',()=>safe(()=>showYoloLocation(issue))),element('p',issue.message));audit.append(row)}root.append(audit)}
  const actions=element('div',undefined,'model-export-actions'),exportButton=button('匯出模型封裝','primary',()=>safe(()=>exportSelectedModel(model.model_version_id)));actions.append(exportButton);const exports=(state.training?.model_exports||[]).filter(item=>item.model_version_id===model.model_version_id);if(exports.length){const latest=exports[0],open=button('開啟最近匯出資料夾','secondary',()=>safe(()=>api('/api/open-folder','POST',{project_id:state.project.id,model_export_id:latest.export_id})));actions.append(open);root.append(actions,element('p',`最近匯出：${latest.export_id} · ${date(latest.created_at)} · ${number(latest.bytes)} bytes`,'field-note'))}else root.append(actions,element('p','封裝包含模型、checkpoint、評估、Epoch 指標、來源 Run 與 SHA-256 manifest，不包含訓練圖片。','field-note'))}
async function exportSelectedModel(modelId){const job=await api(projectPath('/model-exports'),'POST',{model_version_id:modelId}),result=await pollJob(job);await loadTraining();toast(`${result.model_version_id} 已匯出為 ${result.export_id}。`);await api('/api/open-folder','POST',{project_id:state.project.id,model_export_id:result.export_id})}
async function generatePredictions(){const model=state.training?.models?.find(item=>item.model_version_id===state.selectedModel)||state.training?.models?.[0];if(!model)throw Error('請先完成一個模型版本。');let assetIds;if($('predictionTarget').value==='current'){if(!state.asset)throw Error('請先在標註頁選擇圖片。');assetIds=[state.asset.id]}const job=await api(projectPath('/predictions'),'POST',{model_version_id:model.model_version_id,...(assetIds?{asset_ids:assetIds}:{})});await pollJob(job);await loadTraining();toast('預標註候選已產生，接受後會進入待審核。')}
async function acceptPredictions(candidateId){const result=await api(`/api/predictions/${candidateId}/accept`,'POST',{});state.project=result.project;if(state.asset&&result.accepted.includes(state.asset.id))await loadAsset(state.asset.id);await loadTraining();renderAssetList();toast(`${number(result.accepted.length)} 張候選已寫入待審標註。`)}

  return {invalidateYoloCompatibility, trainingParameters, trainingMonitor, applyTrainingConfigTab, loadTraining, yoloCompatibilitySignature, renderYoloCompatibility, checkReviewYoloCompatibility, checkYoloCompatibility, renderTraining, createDatasetVersion, startTrainingRun, stopTrainingRun, generatePredictions};
}

