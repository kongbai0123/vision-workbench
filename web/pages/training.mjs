import {TaskTiming} from '../task-timing.mjs';
import {decodeMask} from '../shapes.mjs';

export function filterModelVersions(models,{dataset='',engine='',query=''}={}){
  const needle=String(query).trim().toLocaleLowerCase();
  return (models||[]).filter(model=>(!dataset||model.dataset_version_id===dataset)&&(!engine||model.engine===engine)&&(
    !needle||[model.model_version_id,model.run_id,model.dataset_version_id,model.engine,model.engine_name].some(value=>String(value||'').toLocaleLowerCase().includes(needle))
  ));
}

export function trialLabelFontSize(renderedWidth){
  const width=Number(renderedWidth);
  return Math.max(12,Math.min(16,Number.isFinite(width)&&width>0?width/80:12));
}

export function trialLabelLayout({canvasWidth,canvasHeight,anchorX,anchorY,textWidth,fontSize}){
  const width=Math.max(1,Number(canvasWidth)||1),height=Math.max(1,Number(canvasHeight)||1),size=Math.max(12,Number(fontSize)||12),paddingX=Math.max(5,size*.4),labelHeight=Math.min(height,Math.ceil(size+8)),labelWidth=Math.min(width,Math.max(1,Math.ceil((Number(textWidth)||0)+paddingX*2))),x=Math.min(Math.max(0,Number(anchorX)||0),Math.max(0,width-labelWidth)),anchor=Math.max(0,Number(anchorY)||0),y=anchor>=labelHeight?anchor-labelHeight:Math.min(anchor,Math.max(0,height-labelHeight));
  return {x,y,width:labelWidth,height:labelHeight,paddingX,textY:y+labelHeight/2};
}

export function drawTrialLabel(context,{label,canvasWidth,canvasHeight,anchorX,anchorY,pixelRatio=1}){
  const ratio=Math.max(1,Number(pixelRatio)||1),fontSize=trialLabelFontSize(canvasWidth);context.save();context.setTransform(ratio,0,0,ratio,0,0);context.font=`600 ${fontSize}px "Segoe UI","Microsoft JhengHei UI","Microsoft JhengHei",sans-serif`;context.textBaseline='middle';const layout=trialLabelLayout({canvasWidth,canvasHeight,anchorX,anchorY,textWidth:context.measureText(label).width,fontSize});context.fillStyle='#102c29';context.fillRect(layout.x,layout.y,layout.width,layout.height);context.fillStyle='#fff';context.fillText(label,layout.x+layout.paddingX,layout.textY,Math.max(1,layout.width-layout.paddingX*2));context.restore();return {fontSize,...layout};
}

const modelParameterDefinitions={
  epochs:['訓練輪數','完整走訪訓練事件的次數；較多輪不保證更好，需搭配 Validation 與早停判讀。'],
  image_size:['影像尺寸','模型實際接收的影像邊長；較大通常保留更多細節，但增加顯存與運算時間。'],
  batch_size:['批次大小','每次前向／反向處理的圖片數；受顯存限制，也會影響梯度穩定度。'],
  gradient_accumulation:['梯度累積','累積多少個 Batch 後才更新一次權重；可在顯存不足時提高有效批次。'],
  learning_rate:['初始學習率','每次更新權重的起始步幅；過大可能震盪，過小可能學習緩慢。'],
  min_learning_rate:['最低學習率','排程下降後的學習率下限。'],
  weight_decay:['權重衰減','正則化強度，用於抑制權重過度增長與過擬合。'],
  optimizer:['最佳化器','決定如何依梯度更新模型權重。'],
  scheduler:['學習率排程','決定學習率隨 Epoch 如何變化。'],
  warmup_epochs:['暖身輪數','訓練初期逐步提高學習率，降低剛開始更新不穩定的風險。'],
  initialization:['初始權重','預訓練權重通常收斂較快；隨機初始化需要更多資料與訓練。'],
  seed:['隨機種子','固定資料順序與隨機操作，方便重現；它不是 Epoch 或迭代次數。'],
  'augmentation.preset':['增強配方','Train 階段採用的資料增強組合。'],
  'augmentation.expansion_count':['每張擴充數','每張 Train 原圖額外產生的獨立增強事件數。'],
  'augmentation.brightness':['亮度強度','隨機亮度變化範圍。'],
  'augmentation.contrast':['對比強度','隨機對比變化範圍。'],
  'augmentation.fliplr':['水平翻轉率','每個訓練事件進行水平翻轉的機率。'],
  'augmentation.flipud':['垂直翻轉率','每個訓練事件進行垂直翻轉的機率。'],
  'augmentation.degrees':['旋轉角度','隨機旋轉的最大角度。'],
  'augmentation.translate':['平移比例','隨機水平／垂直平移的最大比例。'],
  'augmentation.scale':['縮放比例','隨機縮放變化範圍。'],
  'augmentation.mosaic':['Mosaic 機率','將多張影像拼接為一個訓練事件的機率。'],
};

function parameterText(value){if(value===undefined||value===null||value==='')return '—';if(typeof value==='boolean')return value?'啟用':'停用';if(typeof value==='number')return Number.isInteger(value)?String(value):String(Number(value.toPrecision(7)));return String(value)}
// Training and model pages share state through an explicit application context.
export function createTrainingPage(context) {
  const {$, state, toast, status, api, projectPath, number, stats, date, button, element, settingValue, editor, flushAllEdits, safe, switchStage, formDialog, renderAssetList, loadAsset, selectAsset, pollJob, nativeChoose, augmentationProfile, TrainingParameters, TrainingMonitor} = context;
const activeRunStates=new Set(['queued','preparing','running','stopping']);
let runTiming=null,runTimingKey=null;
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
let trialResult=null,trialIndex=0,trialTimer=null;
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
    renderAnnotationModels();
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
  $('backgroundTrainingProgress').hidden=true;
  button.hidden=!run&&!error;
  if(error){$('backgroundTrainingText').textContent=`訓練狀態暫時無法更新：${error}`;$('backgroundTrainingProgress').removeAttribute('value');return}
  if(!run)return;
  const epoch=run.epoch?` · Epoch ${number(run.epoch)}/${number(run.config?.epochs)}`:'';
  const batch=run.batch&&run.batches_per_epoch?` · Batch ${number(run.batch)}/${number(run.batches_per_epoch)}`:'';
  const updates=Number(run.execution?.optimizer_steps);const step=Number.isFinite(updates)?` · 權重更新 ${number(updates)}`:'';
  const timingKey=`${state.project?.id}/${run.run_id}`;
  if(runTimingKey!==timingKey){runTimingKey=timingKey;runTiming=new TaskTiming();}
  const measured=run.status==='running'&&Number(run.config?.epochs)>0?Number(run.epoch||0)/Number(run.config.epochs)*100:null;
  runTiming.update(measured,run.status==='preparing'?'running':run.status,run.status);
  $('backgroundTrainingText').textContent=`${run.run_id} ${trainingStatusName(run.status)}${epoch}${batch}${step} · ${runTiming.text('訓練階段')}（依目前觀測；不含後續評估，早停可能提前結束）`;
  button.title=$('backgroundTrainingText').textContent;
  $('backgroundTrainingText').textContent=`${run.run_id} · ${runTiming.text('訓練階段')}`;
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
  if(!state.training)return;const {models=[]}=state.training;
  const datasetFilter=$('modelDatasetFilter'),engineFilter=$('modelEngineFilter'),query=$('modelSearch').value,datasetRemember=datasetFilter.value,engineRemember=engineFilter.value;
  datasetFilter.onchange=renderModels;engineFilter.onchange=renderModels;$('modelSearch').oninput=renderModels;$('clearModelFilters').onclick=()=>{datasetFilter.value='';engineFilter.value='';$('modelSearch').value='';renderModels()};
  datasetFilter.replaceChildren(new Option('全部資料版本',''));for(const value of [...new Set(models.map(model=>model.dataset_version_id).filter(Boolean))].sort())datasetFilter.append(new Option(value,value));datasetFilter.value=[...datasetFilter.options].some(option=>option.value===datasetRemember)?datasetRemember:'';
  engineFilter.replaceChildren(new Option('全部模型引擎',''));for(const value of [...new Set(models.map(model=>model.engine).filter(Boolean))].sort()){const sample=models.find(model=>model.engine===value);engineFilter.append(new Option(sample?.engine_name||value,value))}engineFilter.value=[...engineFilter.options].some(option=>option.value===engineRemember)?engineRemember:'';
  const visible=filterModelVersions(models,{dataset:datasetFilter.value,engine:engineFilter.value,query});$('modelFilterCount').textContent=`顯示 ${visible.length}／${models.length} 個模型`;
  if(!visible.some(model=>model.model_version_id===state.selectedModel))state.selectedModel=visible[0]?.model_version_id||null;
  const list=$('modelList');list.replaceChildren();if(!visible.length)list.append(element('p',models.length?'沒有符合目前篩選條件的模型。':'尚無模型版本。','empty-list'));
  for(const model of visible){const definition=state.training?.capabilities?.engines?.find(item=>item.key===model.engine),row=button('','model-row'+(model.model_version_id===state.selectedModel?' active':''),()=>{state.selectedModel=model.model_version_id;renderModels()});const top=element('div',undefined,'model-row-top');top.append(element('b',`${model.model_version_id} · ${model.engine_name}`),element('span',definition?.predict?'可預標註':'訓練／匯出','run-status'));row.append(top,element('small',`${model.source?.kind==='external_import'?'外部匯入 · 尚未評估':`${model.dataset_version_id} · ${model.run_id}`} · ${date(model.created_at)}`));list.append(row)}
  const selected=visible.find(model=>model.model_version_id===state.selectedModel);renderModelDetail(selected);
  const selectedDefinition=state.training?.capabilities?.engines?.find(item=>item.key===selected?.engine);for(const id of ['trialImages','trialVideo','runModelComparison']){$(id).disabled=!selected||!selectedDefinition?.predict;$(id).title=selected&&!selectedDefinition?.predict?'此模型尚未提供推論介面':''}
  if(selected&&!selected.dataset_version_id){$('runModelComparison').disabled=true;$('runModelComparison').title='外部模型未綁定本專案資料版本，請使用圖片／影片試跑。'}
}
function renderAnnotationModels(){
  const select=$('annotationModel'),root=$('annotationPredictionList');if(!select||!root||!state.training)return;const models=state.training.models||[],predictable=models.filter(model=>state.training.capabilities?.engines?.find(item=>item.key===model.engine)?.predict),previous=select.value;select.replaceChildren();
  if(!predictable.length)select.append(new Option('尚無可用的專案模型',''));else for(const model of predictable)select.append(new Option(`${model.model_version_id} · ${model.engine_name}`,model.model_version_id));if(predictable.some(m=>m.model_version_id===previous))select.value=previous;
  $('generateCurrentPrediction').disabled=!predictable.length||!state.asset;root.replaceChildren();
  for(const candidate of state.training.predictions||[]){
    const row=candidate.assets.find(asset=>asset.asset_id===state.asset?.id);if(!row)continue;
    const item=element('article',undefined,'prediction-row');item.append(element('b',`${candidate.model_version_id} · ${date(candidate.created_at)}`));
    if(row.status==='candidate')item.append(
      element('small',`${row.shapes.length} 個候選；黃色預覽不會修改圖片`),
      button('預覽候選','secondary',()=>editor.setProposals(row.shapes)),
      button('接受目前圖片','secondary',()=>safe(()=>acceptPredictions(candidate.candidate_id,[row.asset_id])))
    );
    else item.append(element('small','此模型在目前圖片沒有偵測到符合門檻的物件。'));
    root.append(item);
  }
  if(!root.children.length)root.append(element('p','目前圖片尚無模型候選。','empty-list'));
}
function flattenModelConfig(config){const flat={};for(const [key,value]of Object.entries(config||{})){if(key==='augmentation'&&value&&typeof value==='object')for(const [child,nested]of Object.entries(value))flat[`augmentation.${child}`]=nested;else if(value===null||typeof value!=='object')flat[key]=value}return flat}
function renderModelParameters(model){
  const run=state.training?.runs?.find(item=>item.run_id===model.run_id),config=run?.config||{},flat=flattenModelConfig(config),models=state.training?.models||[];
  const previousModel=models.filter(item=>item.model_version_id!==model.model_version_id&&item.engine===model.engine&&Number(new Date(item.created_at||0))<=Number(new Date(model.created_at||0))).sort((a,b)=>Number(new Date(b.created_at||0))-Number(new Date(a.created_at||0)))[0];
  const previousRun=state.training?.runs?.find(item=>item.run_id===previousModel?.run_id),previous=flattenModelConfig(previousRun?.config||{}),keys=[...new Set([...Object.keys(modelParameterDefinitions).filter(key=>key in flat||key in previous),...Object.keys(flat).filter(key=>!(key in modelParameterDefinitions))])];
  const details=element('details',undefined,'model-parameters');details.open=true;const summary=document.createElement('summary');summary.append(element('span','TRAINING PARAMETERS','eyebrow'),element('b','訓練設定與差異說明'),element('small',previousModel?`比較基準：${previousModel.model_version_id} · ${previousModel.dataset_version_id}`:'第一個同引擎模型，沒有前版可比較'));details.append(summary);
  if(!keys.length){details.append(element('p','此舊模型沒有保存可顯示的訓練參數。','readiness-item'));return details}
  const note=element('p','此處讀取來源 Run 實際保存的設定。標示「已變更」表示與上一個同引擎模型不同，不代表設定一定較好或較差。','field-note'),table=document.createElement('table');table.className='model-parameter-table';const head=document.createElement('thead'),headRow=document.createElement('tr');for(const label of ['參數與用途','目前模型',previousModel?previousModel.model_version_id:'前一版'])headRow.append(element('th',label));head.append(headRow);const body=document.createElement('tbody');
  for(const key of keys){const definition=modelParameterDefinitions[key]||[key,'由所選模型引擎提供的專用設定。'],current=flat[key],before=previous[key],changed=previousModel&&parameterText(current)!==parameterText(before),row=document.createElement('tr');if(changed)row.className='parameter-changed';const name=document.createElement('td');name.append(element('b',definition[0]),element('small',definition[1]));const currentCell=document.createElement('td');currentCell.append(element('code',parameterText(current)));if(changed)currentCell.append(element('span','已變更','parameter-change-badge'));row.append(name,currentCell,element('td',previousModel?parameterText(before):'—'));body.append(row)}
  table.append(head,body);details.append(note,table);return details;
}
function renderModelDetail(model){const root=$('modelDetail');root.replaceChildren();
  const heading=element('div',undefined,'model-section-heading'),headingCopy=element('div');headingCopy.append(element('span','MODEL VERSION'),element('h2',model?`${model.model_version_id} · ${model.engine_name}`:'模型版本'));heading.append(headingCopy);root.append(heading);if(!model){const empty=element('div',undefined,'report-empty');empty.append(element('span','◇'),element('h3','尚無可用模型'),element('p','完成訓練，或匯入外部 YOLO／RT-DETR .pt 權重後，模型會出現在這裡。'));root.append(empty);return}
  const definition=state.training?.capabilities?.engines?.find(item=>item.key===model.engine);heading.append(element('span',definition?.task_name||'模型版本','badge approved'));
  const result=model.test||model.validation||{},split=model.test?'Test':'Validation',invalid=model.evaluation_reassessment?.valid===false,values=!model.test&&!model.validation?[['評估狀態','尚未評估']]:invalid?[['評估結果','不可用'],['原因',model.evaluation_reassessment.reason||'資料或評估流程不符合要求']]:result.accuracy!==undefined?[[`${split} Accuracy`,Number(result.accuracy).toFixed(3)],[`${split} Macro F1`,Number(result.macro_f1).toFixed(3)],[`${split} Macro Recall`,Number(result.macro_recall).toFixed(3)]]:result.mask_map50_95!==undefined?[[`${split} Mask mAP50–95`,Number(result.mask_map50_95).toFixed(3)],[`${split} Mask mAP50`,Number(result.mask_map50).toFixed(3)]]:result.box_map50_95!==undefined?[[`${split} Box mAP50–95`,Number(result.box_map50_95).toFixed(3)],[`${split} Box mAP50`,Number(result.box_map50).toFixed(3)]]:result.box_map50!==undefined?[[`${split} Box mAP50`,Number(result.box_map50).toFixed(3)],[`${split} Recall@0.5`,Number(result.recall_50).toFixed(3)]]:result.mean_dice!==undefined?[[`${split} mIoU`,Number(result.mean_iou).toFixed(3)],[`${split} Dice`,Number(result.mean_dice).toFixed(3)]]:[[`${split} Mean IoU`,Number(result.mean_iou||0).toFixed(3)]];if(!invalid&&(model.test||model.validation)){values.push([`${split} 圖片`,number(result.images)],['類別數',number(model.classes?.length)]);if(!model.test)values.push(['獨立 Test','尚未執行'])}const score=element('div',undefined,'run-metrics');for(const [label,value]of values){const item=element('div',undefined,'run-metric');item.append(element('span',label),element('b',value));score.append(item)}root.append(score);
  const imported=model.source?.kind==='external_import',abilities=[...(imported?['外部權重','試跑']:['訓練','評估']),...(definition?.predict?['預標註']:[]),'匯出'].join('／'),facts=element('dl',undefined,'model-facts');for(const [label,value]of [['來源 Run',imported?'外部匯入，無本機訓練紀錄':model.run_id],['訓練資料',model.dataset_version_id||'未提供本專案資料版本'],['類別',(model.classes||[]).join('、')],['模型能力',abilities],['建立時間',date(model.created_at)]])facts.append(element('dt',label),element('dd',value));root.append(facts,element('p',imported?'外部權重與類別保存在本專案；原訓練資料、訓練設定及評估成績未提供。':model.task==='image_classification'?'分類結果保留為圖片層級評估，不會建立覆蓋整張圖片的 Bounding Box。':'模型與資料版本、類別映射和評估結果一起保存；專案後續修改不會回寫此模型。','readiness-item'),imported?element('p',`來源：${model.source.filename} · SHA-256：${model.source.sha256}。已驗證載入與 CPU 推論；尚未量測準確率。`,'readiness-item imported-model-source'):renderModelParameters(model));
  if(model.yolo_compatibility){const report=model.yolo_compatibility,summary=report.summary||{},audit=element('details',undefined,'training-monitor-details'),heading=document.createElement('summary');heading.textContent=`YOLO Seg 相容稽核 · 修補 ${number(summary.pixels_repaired)} px`;const copy=element('p',`已掃描 ${number(summary.assets_scanned)} 張；${number(summary.affected_assets)} 張的 Run 副本修補 ${number(summary.holes_repaired)} 個微小孔洞。原始標註與固定資料版本未變更，模型內部評估使用相容副本。`,'readiness-item');audit.append(heading,copy);for(const issue of report.repairs||[]){const row=element('div',undefined,'yolo-issue');row.append(element('b',`${issue.name} · ${issue.label||'未命名標註'}`),button('放大位置','text-button',()=>safe(()=>showYoloLocation(issue))),element('p',issue.message));audit.append(row)}root.append(audit)}
  const actions=element('div',undefined,'model-export-actions'),exportButton=button('匯出模型封裝','primary',()=>safe(()=>exportSelectedModel(model.model_version_id)));actions.append(exportButton);const exports=(state.training?.model_exports||[]).filter(item=>item.model_version_id===model.model_version_id);if(exports.length){const latest=exports[0],open=button('開啟最近匯出資料夾','secondary',()=>safe(()=>api('/api/open-folder','POST',{project_id:state.project.id,model_export_id:latest.export_id})));actions.append(open);root.append(actions,element('p',`最近匯出：${latest.export_id} · ${date(latest.created_at)} · ${number(latest.bytes)} bytes`,'field-note'))}else root.append(actions,element('p',imported?'封裝包含匯入權重、來源資訊與 SHA-256 manifest；不包含原訓練資料或本機訓練紀錄。':'封裝包含模型、checkpoint、評估、Epoch 指標、來源 Run 與 SHA-256 manifest，不包含訓練圖片。','field-note'))}
async function importExternalModel(){
  const body=element('div'),fileInput=document.createElement('input'),dropZone=element('label',undefined,'model-import-dropzone'),dropIcon=element('span','⇧','model-import-drop-icon'),dropTitle=element('b','拖曳 .pt 權重至此'),dropHint=element('small','或點擊這裡瀏覽電腦中的檔案'),selection=element('div','尚未選擇檔案','model-import-selection'),nameLabel=element('label','模型名稱（選填）'),name=document.createElement('input'),trustLabel=element('label',undefined,'check-label'),trust=document.createElement('input');
  let selectedFile=null;
  fileInput.id='modelImportFile';fileInput.type='file';fileInput.accept='.pt';fileInput.hidden=true;dropZone.htmlFor=fileInput.id;dropZone.tabIndex=0;dropZone.setAttribute('role','button');dropZone.setAttribute('aria-label','選擇或拖曳 .pt 模型權重');selection.setAttribute('aria-live','polite');name.id='modelImportName';name.maxLength=100;nameLabel.htmlFor=name.id;trust.id='modelImportTrusted';trust.type='checkbox';trustLabel.append(trust,document.createTextNode('我確認權重來自可信任的來源'));
  const selectFile=file=>{if(!file)return;if(!file.name.toLowerCase().endsWith('.pt'))throw Error('請選擇副檔名為 .pt 的模型權重。');if(!file.size)throw Error('模型檔案是空的。');if(file.size>8*1024*1024*1024)throw Error('模型檔案超過 8 GiB 上限。');selectedFile=file;selection.textContent=`${file.name} · ${file.size<1024*1024?`${Math.max(1,Math.ceil(file.size/1024))} KB`:`${(file.size/1024/1024).toFixed(file.size<10*1024*1024?1:0)} MB`}`;selection.classList.add('selected');dropZone.classList.add('has-file');dropTitle.textContent='已選擇模型權重';dropHint.textContent='點擊或拖入另一個檔案即可更換'};
  fileInput.addEventListener('change',()=>{try{selectFile(fileInput.files?.[0])}catch(error){toast(error.message,true);fileInput.value=''}});
  dropZone.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();fileInput.click()}});
  for(const eventName of ['dragenter','dragover'])dropZone.addEventListener(eventName,event=>{event.preventDefault();event.stopPropagation();dropZone.classList.add('drag-active');event.dataTransfer.dropEffect='copy'});
  for(const eventName of ['dragleave','drop'])dropZone.addEventListener(eventName,event=>{event.preventDefault();event.stopPropagation();dropZone.classList.remove('drag-active')});
  dropZone.addEventListener('drop',event=>{try{const files=[...(event.dataTransfer?.files||[])];if(files.length!==1)throw Error('一次只能匯入一個 .pt 權重。');selectFile(files[0])}catch(error){toast(error.message,true)}});
  dropZone.append(dropIcon,dropTitle,dropHint);body.append(element('p','支援 Ultralytics 相容的 YOLO 偵測／實例分割與 RT-DETR 偵測權重。檔案會先安全上傳至本機暫存區，再複製至專案並驗證架構、類別與 CPU 推論。'),fileInput,dropZone,selection,nameLabel,name,element('p','載入 .pt 可能執行其中的 Python 程式碼。請只選擇自己訓練或信任來源的權重。','field-note'),trustLabel,element('p','模型類別沿用權重中的名稱；預標註前請在專案建立相同類別。匯入不會建立假的訓練紀錄或評估分數。','field-note'));
  const result=await formDialog({title:'匯入外部模型',body,confirm:'驗證並匯入',eyebrow:'IMPORT MODEL',onSubmit:async()=>{
    if(!selectedFile)throw Error('請選擇或拖入 .pt 權重檔案。');if(!trust.checked)throw Error('請先確認權重來源可信任。');
    const catalog=await api('/api/model-catalog?refresh=1'),component=(catalog.components||[]).find(item=>item.id==='ultralytics');
    if(component?.state!=='ready')throw Error(component?.message||'請先至「設定 → 模型與元件」安裝或修復 Ultralytics 執行環境。');
    const upload=await api('/api/model-uploads','POST',selectedFile);
    return pollJob(await api(projectPath('/model-import'),'POST',{upload_token:upload.upload_token,name:name.value.trim(),trusted:trust.checked}));
  }});
  if(result){$('modelDatasetFilter').value='';$('modelEngineFilter').value='';$('modelSearch').value='';state.selectedModel=result.model_version_id;await loadTraining();toast(`${result.model_version_id} 已匯入，可開始圖片／影片試跑。`)}
}
async function exportSelectedModel(modelId){const job=await api(projectPath('/model-exports'),'POST',{model_version_id:modelId}),result=await pollJob(job);await loadTraining();toast(`${result.model_version_id} 已匯出為 ${result.export_id}。`);await api('/api/open-folder','POST',{project_id:state.project.id,model_export_id:result.export_id})}
async function generatePredictions(){if(!state.asset)throw Error('請先在標註頁選擇圖片。');const modelId=$('annotationModel').value;if(!modelId)throw Error('尚無可用的專案模型。');const job=await api(projectPath('/predictions'),'POST',{model_version_id:modelId,asset_ids:[state.asset.id]});await pollJob(job);await loadTraining();toast('目前圖片的模型候選已產生，請預覽後再接受。')}
async function acceptPredictions(candidateId,assetIds){const result=await api(`/api/predictions/${candidateId}/accept`,'POST',{asset_ids:assetIds});editor.setProposals();state.project=result.project;if(state.asset&&result.accepted.includes(state.asset.id))await loadAsset(state.asset.id);await loadTraining();renderAssetList();toast('候選已寫入待審標註。')}
function confidence(shape){return Number(shape.metadata?.confidence??shape.confidence??1)}
function shapeBox(shape,frame){
  if(shape.type==='rectangle')return [Number(shape.x),Number(shape.y),Number(shape.width),Number(shape.height)];
  if(shape.points?.length){const xs=shape.points.map(point=>Number(point[0])),ys=shape.points.map(point=>Number(point[1]));const left=Math.min(...xs),top=Math.min(...ys);return [left,top,Math.max(0,Math.max(...xs)-left),Math.max(0,Math.max(...ys)-top)]}
  if(shape.type==='mask'&&shape.counts){const data=decodeMask(shape.counts,frame.width,frame.height);let left=frame.width,top=frame.height,right=-1,bottom=-1;for(let index=0;index<data.length;index++)if(data[index]){const x=index%frame.width,y=Math.floor(index/frame.width);left=Math.min(left,x);top=Math.min(top,y);right=Math.max(right,x);bottom=Math.max(bottom,y)}if(right>=left)return [left,top,right-left+1,bottom-top+1]}
  return [0,0,0,0];
}
function boxIou(left,right){const [ax,ay,aw,ah]=left,[bx,by,bw,bh]=right,x1=Math.max(ax,bx),y1=Math.max(ay,by),x2=Math.min(ax+aw,bx+bw),y2=Math.min(ay+ah,by+bh),intersection=Math.max(0,x2-x1)*Math.max(0,y2-y1),union=aw*ah+bw*bh-intersection;return union>0?intersection/union:0}
function comparisonAtThreshold(threshold){
  let tp=0,fp=0,fn=0;for(const frame of trialResult?.frames||[]){const predictions=(frame.shapes||[]).filter(shape=>confidence(shape)>=threshold),available=new Set(predictions.map((_,index)=>index));for(const truth of frame.ground_truth||[]){let best=-1,bestIou=0;for(const index of available){const candidate=predictions[index];if(candidate.label!==truth.label)continue;const overlap=boxIou(shapeBox(truth,frame),shapeBox(candidate,frame));if(overlap>bestIou){bestIou=overlap;best=index}}if(best>=0&&bestIou>=Number(trialResult.comparison?.iou_threshold||.5)){available.delete(best);tp++}else fn++}fp+=available.size}
  const precision=tp+fp?tp/(tp+fp):null,recall=tp+fn?tp/(tp+fn):null;return {tp,fp,fn,precision,recall};
}
function drawTrial(){
  if(!trialResult?.frames?.length)return;const frame=trialResult.frames[trialIndex],image=$('trialImage'),canvas=$('trialOverlay');$('trialTimeline').value=trialIndex;$('trialFrameLabel').textContent=frame.frame_index===null?`${trialIndex+1} / ${trialResult.frames.length}`:`Frame ${frame.frame_index+1} · ${Number(frame.time_seconds||0).toFixed(2)} s`;
  image.onload=()=>{const width=image.clientWidth,height=image.clientHeight,pixelRatio=Math.max(1,Number(globalThis.devicePixelRatio)||1);canvas.width=Math.max(1,Math.round(width*pixelRatio));canvas.height=Math.max(1,Math.round(height*pixelRatio));canvas.style.width=width+'px';canvas.style.height=height+'px';const ctx=canvas.getContext('2d');ctx.scale(pixelRatio*width/frame.width,pixelRatio*height/frame.height);const threshold=Number($('trialConfidence').value);let shown=0;
    for(const shape of frame.shapes||[]){if(confidence(shape)<threshold)continue;shown++;ctx.strokeStyle='#45c6b1';ctx.fillStyle='#45c6b166';ctx.lineWidth=Math.max(2,frame.width/700);if(shape.type==='rectangle'){ctx.strokeRect(shape.x,shape.y,shape.width,shape.height)}else if(shape.type==='polygon'&&shape.points?.length){ctx.beginPath();shape.points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fill();ctx.stroke()}else if(shape.type==='mask'&&shape.counts){const data=decodeMask(shape.counts,frame.width,frame.height),mask=document.createElement('canvas');mask.width=frame.width;mask.height=frame.height;const pixels=mask.getContext('2d').createImageData(frame.width,frame.height);for(let i=0;i<data.length;i++)if(data[i]){pixels.data[i*4]=69;pixels.data[i*4+1]=198;pixels.data[i*4+2]=177;pixels.data[i*4+3]=105}mask.getContext('2d').putImageData(pixels,0,0);ctx.drawImage(mask,0,0)}const label=`${shape.label||'object'} ${(confidence(shape)*100).toFixed(0)}%`,[left,top]=shapeBox(shape,frame);drawTrialLabel(ctx,{label,canvasWidth:width,canvasHeight:height,anchorX:left*width/frame.width,anchorY:top*height/frame.height,pixelRatio})}
    for(const truth of frame.ground_truth||[]){ctx.strokeStyle='#65a9ff';ctx.lineWidth=Math.max(2,frame.width/700);ctx.setLineDash([10,6]);if(truth.type==='rectangle')ctx.strokeRect(truth.x,truth.y,truth.width,truth.height);else if(truth.points?.length){ctx.beginPath();truth.points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke()}ctx.setLineDash([])}
    const summary=$('trialSummary');summary.replaceChildren();if(trialResult.comparison){const metrics=comparisonAtThreshold(threshold);for(const [label,value,ratio]of [[`Confidence ≥ ${threshold.toFixed(2)}`,'Box IoU ≥ 0.50',false],['Precision',metrics.precision??'—',true],['Recall',metrics.recall??'—',true],['TP',metrics.tp,false],['FP',metrics.fp,false],['FN',metrics.fn,false]]){const item=element('div',undefined,'run-metric'),display=ratio&&typeof value==='number'?value.toFixed(3):String(value);item.append(element('span',label),element('b',display));summary.append(item)}}else{const item=element('div',undefined,'run-metric');item.append(element('span','目前顯示'),element('b',`${shown} 個預測`));summary.append(item)}};
  image.onerror=()=>{$('modelTrialStatus').textContent='無法載入試跑影格，請重新執行模型試跑。'};image.src=`/api/model-trials/${trialResult.session_id}/frames/${trialIndex}`;
}
function stopTrialPlayback(){if(trialTimer)clearInterval(trialTimer);trialTimer=null;$('trialPlay').textContent='播放'}
async function startModelTrial(kind){const model=state.training?.models?.find(item=>item.model_version_id===state.selectedModel)||state.training?.models?.[0];if(!model)throw Error('請先選擇模型。');const paths=await nativeChoose(kind==='video'?'video':'images');if(!paths.length)return;stopTrialPlayback();$('modelTrialStatus').textContent='正在執行模型試跑；影片會逐幀處理。';trialResult=await pollJob(await api(projectPath('/model-trials'),'POST',{model_version_id:model.model_version_id,paths}));trialIndex=0;$('modelTrialViewer').hidden=false;$('trialTimeline').max=Math.max(0,trialResult.frames.length-1);$('modelTrialStatus').textContent=`試跑完成 · ${model.model_version_id} · ${trialResult.frames.length} 個影格。外部資料沒有人工標註，因此不計算準確率或 mAP。`;drawTrial()}
async function runModelComparison(){const model=state.training?.models?.find(item=>item.model_version_id===state.selectedModel)||state.training?.models?.[0];if(!model)throw Error('請先選擇模型。');stopTrialPlayback();const split=$('comparisonSplit').value;$('comparisonStatus').textContent='正在固定標註資料上重新推論…';trialResult=await pollJob(await api(projectPath('/model-comparisons'),'POST',{model_version_id:model.model_version_id,split}));trialIndex=0;$('modelTrialViewer').hidden=false;$('trialTimeline').max=Math.max(0,trialResult.frames.length-1);const metric=trialResult.comparison;$('comparisonStatus').textContent=`${split} 比對完成 · 調整信心門檻會同步重算 Precision／Recall；藍色虛線為人工標註。`;drawTrial();$('modelTrialViewer').scrollIntoView({block:'nearest'})}
function setTrialFrame(value){trialIndex=Math.max(0,Math.min(trialResult?.frames?.length-1||0,Number(value)));drawTrial()}
function toggleTrialPlayback(){if(trialTimer){stopTrialPlayback();return}if(!trialResult?.frames?.length)return;$('trialPlay').textContent='暫停';const frames=trialResult.frames,delay=Math.max(16,Math.round(((frames[1]?.time_seconds||1/10)-(frames[0]?.time_seconds||0))*1000));trialTimer=setInterval(()=>{if(trialIndex>=frames.length-1){stopTrialPlayback();return}setTrialFrame(trialIndex+1)},delay)}

  return {importExternalModel,invalidateYoloCompatibility, trainingParameters, trainingMonitor, applyTrainingConfigTab, loadTraining, yoloCompatibilitySignature, renderYoloCompatibility, checkReviewYoloCompatibility, checkYoloCompatibility, renderTraining, renderAnnotationModels, createDatasetVersion, startTrainingRun, stopTrainingRun, generatePredictions,startModelTrial,runModelComparison,setTrialFrame,toggleTrialPlayback,drawTrial};
}
