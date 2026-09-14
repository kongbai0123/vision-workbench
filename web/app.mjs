import {AnnotationEditor, colorFor, shapeNames} from './editor.mjs';
import {kind, decodeMask, encodeMask, brush} from './shapes.mjs';
import {SaveQueue} from './save-queue.mjs';
import {TrainingMonitor} from './training-monitor.mjs';
import {TrainingParameters} from './training-parameters.mjs';
import {SplitManager} from './split-manager.mjs';

const $ = id => document.getElementById(id);
const state = {projects:[],project:null,asset:null,stage:'library',acquireSource:'camera',busy:false,transitioning:false,
  system:null,editorMode:'builtin',cvatPoll:null,cvatElapsedTimer:null,cvatPollStartedAt:null,cvatPollBaseText:'',cvatPollBusy:false,cvatWasOpened:false,nativeBridge:null,reviewSelection:new Set(),acquireSelection:new Set(),reviewPage:0,reviewGeneration:0,previewRunning:false,previewTimer:null,camera:false,recording:false,cameraDetails:null,cameraPoll:0,autoCapture:null,cameraTarget:null,cameraTargetDraft:[],cameraTargetGesture:null,
  splitTab:'train',splitPage:0,validationResult:null,validationTab:null,validationPage:0,releaseDrawerFocus:null,
  training:null,trainingTimer:null,trainingMetrics:null,selectedRun:null,selectedModel:null,modelCatalog:null,selectedCatalogModel:null,yoloCompatibility:null,yoloCompatibilitySignature:'',
  settingsPage:'general',settingsFocus:null,settingsJob:null,desktopUpdate:null};
const reviewNames = {pending:'待審核',approved:'已核准',rejected:'已退回'};
const formatNames = {native:'原生專案',coco:'COCO',yolo_detection:'YOLO 偵測',yolo_segmentation:'YOLO 分割',labelme:'LabelMe',classification:'圖片分類',jsonl:'JSONL'};
const formatDescriptions = {
  native:'完整保留幾何形狀、遮罩、來源與修訂資訊，適合備份及再次匯入工作站。',
  coco:'保留物件偵測框、多邊形與遮罩。遮罩使用 RLE 保存內孔，適合精確分割資料交付。',
  yolo_detection:'將物件轉成正規化矩形框。多邊形、旋轉框或遮罩可能產生幾何損失，須閱讀驗證報告。',
  yolo_segmentation:'將物件輸出為多邊形。含內孔或分離區塊的遮罩可能無法無損表達，請檢查轉換報告。',
  labelme:'逐張輸出圖片及 JSON 標註。各形狀支援情形由驗證報告確認。',
  classification:'每張已核准圖片必須只有一種類別。依 train / val / test 及類別輸出圖片資料夾，用於影像分類。',
  jsonl:'每張圖片一列 JSON，保留完整原生標註、來源與修訂資訊；附帶原圖供資料處理流程使用。',
};
let toastTimer;
function toast(message,error=false) {
  clearTimeout(toastTimer);$('toast').textContent=String(message);$('toast').classList.toggle('error',error);$('toast').hidden=false;
  toastTimer=setTimeout(()=>$('toast').hidden=true,error?9500:4500);
}
function status(message,error=false) {$('statusText').textContent=message;$('connectionDot').classList.toggle('error',error);}
function setUpdateIndicators(count){
  const visible=Number.isInteger(count)&&count>0;
  for(const id of ['updateDot','settingsUpdateNavDot','settingsUpdateCardDot'])if($(id))$(id).hidden=!visible;
  $('desktopUpdateCard')?.classList.toggle('has-update',visible);
  $('settings').title=count===null?'開啟設定中心；暫時無法檢查更新':count>0?`開啟設定中心；有新更新：${count} 個程式檔案已修改`:'開啟設定中心';
}
window.workbenchUpdateStatus=count=>{
  setUpdateIndicators(count);
  if(state.desktopUpdate?.state==='updating'||state.desktopUpdate?.state==='restarting')return;
  if(count===null)renderDesktopUpdate({state:'unavailable',count:null,changes:[],blockers:[],message:'暫時無法讀取程式檔案，請稍後再試。'});
  else if(count>0&&state.desktopUpdate?.count!==count)renderDesktopUpdate({state:'available',count,changes:[],blockers:[],message:`偵測到 ${count} 個程式檔案有新修改。`});
  else if(count===0&&state.desktopUpdate?.state==='available')renderDesktopUpdate({state:'current',count:0,changes:[],blockers:[],message:'目前執行中的程式已是最新狀態。'});
};
async function api(path,method='GET',body) {
  let response;
  try {response=await fetch(path,{method,cache:'no-store',headers:body===undefined?{}:{'Content-Type':'application/json','X-Workbench':'1'},...(body===undefined?{}:{body:JSON.stringify(body)})});}
  catch {throw Error('無法連線到本機服務。編輯內容仍保留在畫面，請恢復服務後儲存。');}
  let value;try{value=await response.json()}catch{throw Error(`本機服務回應格式無效（${response.status}）。`)}
  if(!response.ok){const e=Error(value.message||value.error||`操作失敗（${response.status}）`);e.status=response.status;e.details=value;throw e;}
  return value;
}
function projectPath(tail='') {if(!state.project)throw Error('請先開啟或建立一份專案。');return `/api/projects/${state.project.id}${tail}`;}
function imageURL(asset,pid=state.project?.id) {return asset.url||`/api/projects/${pid}/assets/${asset.id}/image`;}
function thumbnailURL(asset) {const url=imageURL(asset);return `${url}${url.includes('?')?'&':'?'}thumbnail=1`;}
function number(value) {return Number(value||0).toLocaleString('zh-TW');}
function stats(project) {
  if(project.assets) return {total:project.assets.length,pending:project.assets.filter(a=>a.review_state==='pending').length,approved:project.assets.filter(a=>a.review_state==='approved').length,rejected:project.assets.filter(a=>a.review_state==='rejected').length};
  return {...{total:0,pending:0,approved:0,rejected:0},...project.stats,...Object.fromEntries(['total','pending','approved','rejected'].filter(k=>project[k]!==undefined).map(k=>[k,project[k]]))};
}
function date(value) {
  if(!value)return '尚無紀錄';
  const parsed=new Date(typeof value==='number'&&value<1e12?value*1000:value);
  return Number.isNaN(parsed.getTime())?String(value):parsed.toLocaleString('zh-TW',{year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
}
function formatDuration(totalSeconds) {
  const total=Math.max(0,Math.floor(totalSeconds));
  const minutes=Math.floor(total/60),seconds=total%60;
  return `${minutes}分 ${String(seconds).padStart(2,'0')}秒`;
}
function currentCvatText() { return state.cvatPollBaseText || 'CVAT 尚未準備。'; }
function updateCvatSetupText({busy=state.cvatPollBusy,startedAt=state.cvatPollStartedAt}={}) {
  const base=currentCvatText();
  const isBusy=busy===undefined?state.cvatPollBusy:busy;
  const started=isNaN(startedAt)?state.cvatPollStartedAt:Number(startedAt);
  if(!isBusy||!Number.isFinite(started)||started<=0||state.editorMode!=='cvat'){ $('cvatSetupText').textContent=base;return; }
  const elapsed=formatDuration((Date.now()-started*1000)/1000);
  $('cvatSetupText').textContent=`${base}（已進行 ${elapsed}）`;
}
function button(text,className='secondary',click) {const b=document.createElement('button');b.type='button';b.textContent=text;b.className=className;if(click)b.onclick=click;return b;}
function element(tag,text,className) {const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}

const settingDefaults={scale:'1',device:'auto'};
function settingValue(name){try{return localStorage.getItem(`vision-workbench.${name}`)||settingDefaults[name]}catch{return settingDefaults[name]}}
function saveSetting(name,value){try{localStorage.setItem(`vision-workbench.${name}`,value)}catch{}applySettings()}
function applySettings(){document.body.style.zoom=settingValue('scale');if($('settingScale'))$('settingScale').value=settingValue('scale');if($('settingDevice'))$('settingDevice').value=settingValue('device')}

function showSettingsPage(page='general'){
  state.settingsPage=page;document.querySelectorAll('[data-settings-page]').forEach(item=>item.classList.toggle('active',item.dataset.settingsPage===page));
  document.querySelectorAll('[data-settings-panel]').forEach(item=>item.hidden=item.dataset.settingsPanel!==page);
  if(page==='updates')void refreshDesktopUpdate().catch(error=>renderDesktopUpdate({state:'unavailable',count:null,changes:[],message:error.message}));
}
async function openSettings(page='general',modelKey=null){
  state.settingsFocus=document.activeElement;$('settingsShell').hidden=false;showSettingsPage(page);applySettings();
  if(modelKey)state.selectedCatalogModel=modelKey;
  if(!state.system)state.system=await api('/api/system');
  await loadModelCatalog();renderSettingsSummary();$('closeSettings').focus();
}
function closeSettings(){if($('settingsShell').hidden)return;$('settingsShell').hidden=true;state.settingsFocus?.focus?.();state.settingsFocus=null}
function renderSettingsSummary(){
  if(!state.system)return;
  $('settingsVersion').textContent=`Vision Workbench ${state.system.version||'—'}`;
  $('settingsLabelmeState').textContent=state.system.desktop?'桌面版可使用':'需要桌面版';
  const facts=[['資料位置',state.system.data_root||'—'],['匯出位置',state.system.default_export_path||'—'],['訓練 Python',state.modelCatalog?.worker_python||'—'],['執行模式',state.system.desktop?'Windows 桌面版':'瀏覽器開發模式']];
  $('computeSummary').replaceChildren(...facts.map(([label,value])=>{const row=element('div',undefined,'settings-fact');row.append(element('span',label),element('b',value));return row}));
  $('diagnosticSummary').textContent=[`Vision Workbench ${state.system.version||'—'}`,`模式：${state.system.desktop?'Windows 桌面版':'瀏覽器'}`,`資料：${state.system.data_root||'—'}`,`訓練 Python：${state.modelCatalog?.worker_python||'—'}`,
    ...((state.modelCatalog?.components||[]).map(component=>`${component.name}：${component.state} · ${component.message}`))].join('\n');
}
function parseNativeResult(value){
  if(typeof value==='string'){try{return JSON.parse(value)}catch{return {state:'unavailable',count:null,changes:[],message:'桌面更新服務回應格式無效。'}}}
  return value&&typeof value==='object'?value:{state:'unavailable',count:null,changes:[],message:'桌面更新服務沒有回應。'};
}
function invokeNativeUpdate(method){return new Promise(resolve=>state.nativeBridge[method](value=>resolve(parseNativeResult(value))))}
function renderDesktopUpdate(snapshot){
  if(!snapshot)return;
  state.desktopUpdate={...(state.desktopUpdate||{}),...snapshot};const update=state.desktopUpdate;
  setUpdateIndicators(update.count);
  const statusNames={available:'有可用更新',blocked:'等待處理',updating:'正在更新',restarting:'正在重啟',current:'已是最新',unavailable:'無法檢查',error:'更新失敗'};
  const badge=$('desktopUpdateBadge');badge.textContent=statusNames[update.state]||'尚未檢查';badge.className='settings-state '+(update.state==='current'?'ready':update.state==='available'||update.state==='blocked'?'warning':update.state==='unavailable'||update.state==='error'?'error':'ready');
  $('desktopUpdateStatus').textContent=update.message||'尚未檢查更新。';
  const changes=Array.isArray(update.changes)?update.changes:[],details=$('desktopUpdateDetails');
  details.hidden=!(Number.isInteger(update.count)&&update.count>0);$('desktopUpdateCount').textContent=update.count>0?`${update.count} 個待套用程式檔案`:'';
  $('desktopUpdateFiles').replaceChildren(...changes.map(path=>element('li',path)));
  if(update.count>0&&!changes.length)$('desktopUpdateFiles').append(element('li','按下更新後會重新確認並列出檔案。'));
  const blockers=Array.isArray(update.blockers)?update.blockers:[],blocker=$('desktopUpdateBlockers');blocker.hidden=!blockers.length;blocker.textContent=blockers.join('\n');
  const action=$('openDesktopUpdater'),running=['updating','restarting'].includes(update.state);action.disabled=running;
  action.textContent=update.state==='available'?'套用更新並重新啟動':update.state==='blocked'?'重新檢查':running?'更新中…':update.state==='current'?'再次檢查':'檢查更新';
}
async function refreshDesktopUpdate(){
  if(!state.nativeBridge?.updateStatus){renderDesktopUpdate({state:'unavailable',count:null,changes:[],message:'本機程式更新只在 Windows 桌面版提供。'});return state.desktopUpdate}
  const snapshot=await invokeNativeUpdate('updateStatus');renderDesktopUpdate(snapshot);return snapshot;
}
async function runDesktopUpdate(){
  if(!state.nativeBridge?.applyUpdate)throw Error('本機程式更新只在 Windows 桌面版提供。');
  const snapshot=await refreshDesktopUpdate();
  if(snapshot.state!=='available'){
    if(snapshot.state==='current')toast('目前執行中的程式已是最新狀態。');
    else if(snapshot.state==='blocked')toast(snapshot.blockers?.[0]||snapshot.message,true);
    else if(snapshot.state!=='updating')toast(snapshot.message||'目前無法開始更新。',true);
    return;
  }
  renderDesktopUpdate(await invokeNativeUpdate('applyUpdate'));
}
window.workbenchUpdateProgress=snapshot=>{renderDesktopUpdate(snapshot);if(snapshot?.state==='error')toast(snapshot.message||'更新未完成。',true)};
async function loadModelCatalog(refresh=false){
  state.modelCatalog=await api(`/api/model-catalog${refresh?'?refresh=1':''}`);renderModelCatalog();renderSettingsSummary();return state.modelCatalog;
}
function catalogState(model){if(model.train)return ['可使用','ready'];if(model.integration!=='ready')return ['整合開發中','warning'];if(model.runtime_state==='broken')return ['需要修復','error'];return ['尚未安裝','warning']}
function renderModelCatalog(){
  const catalog=state.modelCatalog;if(!catalog)return;const search=$('modelCatalogSearch').value.trim().toLowerCase(),filter=$('modelCatalogFilter').value;
  const models=(catalog.engines||catalog.models||[]).filter(model=>{
    if(filter==='ready'&&!model.train)return false;if(filter==='unavailable'&&model.train)return false;
    return !search||`${model.name} ${model.description} ${model.task_name} ${model.family}`.toLowerCase().includes(search);
  });
  if(!state.selectedCatalogModel||!(catalog.engines||catalog.models||[]).some(model=>model.key===state.selectedCatalogModel))state.selectedCatalogModel=models[0]?.key||(catalog.engines||catalog.models||[])[0]?.key;
  const list=$('settingsModelList');list.replaceChildren();
  if(!models.length)list.append(element('p','沒有符合條件的模型。','empty-list'));
  for(const model of models){const [label,tone]=catalogState(model),row=button('','settings-model-row'+(model.key===state.selectedCatalogModel?' active':''),()=>{state.selectedCatalogModel=model.key;renderModelCatalog()});const top=element('div');top.append(element('b',model.name),element('i',undefined,'model-state-dot '+tone));row.append(top,element('small',`${model.task_name||model.task} · ${label}`));list.append(row)}
  const model=(catalog.engines||catalog.models||[]).find(item=>item.key===state.selectedCatalogModel);renderCatalogDetail(model);
}
function renderCatalogDetail(model){
  const root=$('settingsModelDetail');root.replaceChildren();if(!model){root.append(element('p','選擇一個模型查看詳情。'));return}
  const [statusLabel,tone]=catalogState(model),head=element('div',undefined,'model-detail-status'),title=element('div');title.append(element('span',model.family||'MODEL','eyebrow'),element('h3',model.name));head.append(title,element('span',statusLabel,'settings-state '+tone));root.append(head,element('p',model.description));
  const component=(state.modelCatalog.components||[]).find(item=>item.id===model.component),stateNames={ready:'可使用',not_installed:'尚未安裝',broken:'需要修復',planned:'整合開發中'};
  const capability=model.train?`訓練可使用 · ${model.predict?'可產生預標註':'不產生物件預標註'} · 可匯出`:model.unavailable_reason||'尚未準備',facts=element('dl',undefined,'catalog-facts');for(const [label,value]of [['任務',model.task_name||model.task],['所需標註',model.annotation||'依模型說明'],['評估指標',(model.metrics||[]).join('、')||'待定'],['執行元件',component?.name||model.component],['元件狀態',stateNames[model.runtime_state]||model.runtime_state],['模型能力',capability]])facts.append(element('dt',label),element('dd',value));root.append(facts,element('p',component?.description||'','field-note'),element('div',`授權：${model.license||'依來源'}`,'catalog-license'));
  const actions=element('div',undefined,'catalog-actions');
  if(model.train){const use=button('回到訓練並選用','primary',()=>{if(!state.project){toast('請先開啟專案。',true);return}closeSettings();switchStage('train').then(()=>{$('trainingEngine').value=model.key;renderTraining()})});actions.append(use)}
  else if(model.integration==='ready'&&component?.installable){const install=button(component.state==='broken'?'修復必要元件':'安裝必要元件','primary',()=>safe(()=>installModelComponent(component.id)));actions.append(install)}
  else {const planned=button('查看開發狀態','secondary',()=>toast('此模型已納入交付待辦；完成資料、評估與推論 adapter 後才會開放安裝。'));actions.append(planned)}
  root.append(actions);
}
async function installModelComponent(componentId){
  const component=(state.modelCatalog.components||[]).find(item=>item.id===componentId);if(!component)throw Error('找不到模型元件');
  const job=await api(`/api/model-components/${componentId}/install`,'POST',{});state.settingsJob=job.id;$('settingsJobText').textContent=`正在準備 ${component.name}`;$('settingsJobProgress').hidden=false;
  let current=job;
  while(!['succeeded','failed','cancelled'].includes(current.state)){await new Promise(resolve=>setTimeout(resolve,650));current=await api(`/api/jobs/${job.id}`);$('settingsJobText').textContent=current.message||`正在準備 ${component.name}`;$('settingsJobProgress').value=Number(current.progress||0)}
  state.settingsJob=null;$('settingsJobProgress').hidden=true;if(current.state!=='succeeded')throw Error(current.error||current.message||'安裝未完成');
  await loadModelCatalog(true);if(state.project)await loadTraining({quiet:true});$('settingsJobText').textContent=`${component.name} 已安裝並通過檢查`;toast(`${component.name} 已可使用。`);
}

let savePhase='saved';
function updateSaveControl() {
  const control=$('saveNow'),available=!!state.asset&&!state.busy&&!state.transitioning;
  const labels={saved:'已儲存',dirty:'立即儲存',saving:'儲存中…',error:'重試儲存'};
  control.replaceChildren(document.createTextNode(labels[savePhase]||'立即儲存'));
  if(['dirty','error'].includes(savePhase))control.append(element('kbd','Ctrl S'));
  control.hidden=!available||savePhase==='saved';
  control.disabled=!available||!['dirty','error'].includes(savePhase);
  control.title=savePhase==='saved'?'目前影像沒有未儲存的修改':savePhase==='dirty'?'自動儲存將於短暫停頓後執行；按此立即儲存':'儲存目前影像的標註修改';
}

const saver = new SaveQueue((id,payload)=>api(projectPath(`/assets/${id}`),'PUT',payload),{
  onStatus: (phase,error) => {
    savePhase=phase;updateSaveControl();
    const labels={saved:'已儲存',dirty:'有修改 · 等待儲存',saving:'正在儲存…',error:'儲存失敗 · 內容保留'};
    $('saveState').textContent=labels[phase];$('saveState').className='save-state '+(phase==='error'?'error':phase==='dirty'?'dirty':'');
    if(error){status('儲存尚未完成；切換工作前請先處理錯誤。',true);toast(error.status===409?'圖片已由其他視窗修改。你的編輯仍保留，請匯出救援副本後重新開啟，避免覆蓋較新修訂。':error.message,true);}
  },
  onSaved:(asset,saved)=>{
    const meta=state.project?.assets.find(a=>a.id===asset.id);
    if(meta)Object.assign(meta,{revision:saved.revision,review_state:saved.review_state||'pending',shape_count:asset.shapes.length});
    updateAssetHeader();renderAssetList();
    status(`已儲存 ${asset.name} · 修訂 ${String(saved.revision).slice(0,8)}`);
  },
});
const editor = new AnnotationEditor({
  onChange:()=>{saver.changed();updateAssetHeader();},onNotice:toast,onSelection:selected=>updateObjectClassUI(selected),
  onCreated:()=>{$('shapeLabel').value='';updateObjectClassUI();},
  onAssetNavigation:direction=>safe(()=>navigateAsset(direction)),
});

async function flushAllEdits() {
  if(editor.gesture)throw Error('請先完成畫布上的拖曳操作。');
  if(editor.draft.length)throw Error('目前仍有尚未完成的頂點。請按 Enter 完成，或按 Esc 取消。');
  if(editor.candidate)throw Error('目前仍有 AI 候選遮罩。請先接受或捨棄，再切換工作或關閉。');
  await saver.flush();
}
window.workbenchFlush=async()=>{if(state.autoCapture?.active)stopAutoCapture('軟體正在結束目前工作，已停止自動擷取。');await flushAllEdits();return true;};
window.workbenchState=()=>({projectId:state.project?.id||null,assetId:state.asset?.id||null,dirty:saver.dirty||editor.hasUncommittedWork,stage:state.stage,busy:state.busy||!!state.autoCapture?.active,transitioning:state.transitioning,loading:state.transitioning,saving:!!saver.inFlight,locked:editor.locked});
window.addEventListener('beforeunload',event=>{if(saver.dirty||editor.hasUncommittedWork||state.busy||state.autoCapture?.active){event.preventDefault();event.returnValue='';}});

async function safe(work) {try{return await work()}catch(error){toast(error.message,true);status(error.message,true);return null;}}
function bind(id,work,{busy=false,task=null}={}) {
  $(id).onclick=async()=>{
    if($(id).disabled||state.busy||state.transitioning)return;
    $(id).disabled=true;
    if(busy){state.busy=true;editor.locked=true;updateNavigation();}
    try{if(task)await runTimedTask(typeof task==='function'?task():task,work);else await work()}catch(error){if(!error.silent){toast(error.message,true);status(error.message,true)}}
    finally{if(busy){state.busy=false;editor.locked=false;updateNavigation();editor.render();editor.updateCandidate()}$(id).disabled=false;updateCameraControls();}
  };
}
function updateNavigation() {
  document.querySelectorAll('[data-stage]').forEach(b=>{b.disabled=state.busy||state.transitioning||(b.dataset.stage!=='library'&&!state.project);b.classList.toggle('active',b.dataset.stage===state.stage)});
  $('home').disabled=state.busy||state.transitioning;
  $('projectName').disabled=!state.project||state.busy||state.transitioning;
  $('openProjectFolder').disabled=!state.project||state.busy;
  $('runAI').disabled=!state.asset||state.busy||state.transitioning;
  updateAcquisitionControls();
}
function updateAcquisitionControls() {
  const locked=state.busy||state.transitioning;
  $('fileDropZone').setAttribute('aria-busy',String(locked));
  if(locked)clearFileDrag();
  document.querySelectorAll('[data-source]').forEach(tab=>tab.disabled=locked);
  $('cameraActivity').disabled=locked;
  $('goAnnotate').disabled=locked||!state.project?.assets.length;
  const visible=[...(state.project?.assets||[])].reverse(),selected=visible.filter(asset=>state.acquireSelection.has(asset.id));
  $('selectAllAssetsLabel').hidden=!visible.length;
  $('selectAllAssets').disabled=locked||!visible.length;
  $('selectAllAssets').checked=!!visible.length&&selected.length===visible.length;
  $('selectAllAssets').indeterminate=selected.length>0&&selected.length<visible.length;
  $('deleteSelectedAssets').hidden=!state.acquireSelection.size;
  $('deleteSelectedAssets').disabled=locked||!state.acquireSelection.size;
  $('deleteSelectedAssets').textContent=`刪除所選（${state.acquireSelection.size}）`;
  $('mergeSelected').disabled=locked||!$('mergeProjects').querySelector('input:checked');
  $('mergeProjects').querySelectorAll('input').forEach(input=>input.disabled=locked);
  $('mergeSelectionSummary').textContent=$('mergeProjects').querySelectorAll('input:checked').length
    ?`已選取 ${$('mergeProjects').querySelectorAll('input:checked').length} 份來源專案`:'尚未選擇來源專案';
}
function clearFileDrag() {
  $('fileDropZone').classList.remove('drop-active');
  $('fileDropTitle').textContent=state.system?.desktop===false?'匯入圖片與標註':'將圖片或資料夾拖曳至此';
}
window.workbenchNativeDrag=(kind,x,y,paths=[])=>{
  const zone=$('fileDropZone'),rect=zone.getBoundingClientRect();
  const ready=kind!=='leave'&&state.stage==='acquire'&&state.acquireSource==='files'
    &&state.project&&!state.busy&&!state.transitioning&&zone.getClientRects().length
    &&x>=rect.left&&x<=rect.right&&y>=rect.top&&y<=rect.bottom
    &&zone.contains(document.elementFromPoint(x,y));
  if(!ready){clearFileDrag();return false;}
  if(kind!=='drop'){
    zone.classList.add('drop-active');$('fileDropTitle').textContent='放開以檢查並匯入';return true;
  }
  clearFileDrag();
  if(!Array.isArray(paths)||!paths.length||paths.some(path=>typeof path!=='string'||!path.trim()))return false;
  $('importPaths').value=[...new Set(paths)].join('\n');
  // Reuse the same serialized, revision-aware import action as the file picker.
  $('importFiles').click();
  return true;
};
// Browser drops must never navigate away from the current project. Windows
// desktop drops are delivered separately by FileDropBridge with original paths.
document.addEventListener('dragover',event=>{event.preventDefault();event.dataTransfer.dropEffect='none';});
document.addEventListener('drop',event=>{
  event.preventDefault();clearFileDrag();
  if(state.stage==='acquire'&&state.acquireSource==='files')toast('請使用桌面版拖曳檔案，或在右側輸入完整路徑。');
});
function switchSource(source) {
  if(state.busy||state.transitioning||!['camera','files','video','screen','merge'].includes(source))return;
  if(source!=='camera')setCameraPreviewExpanded(false);
  clearFileDrag();
  setSourceInspector(false);
  state.acquireSource=source;
  const compactCapture=document.querySelector('.compact-camera-capture');if(compactCapture)compactCapture.hidden=source!=='camera';
  for(const tab of document.querySelectorAll('[data-source]')) {
    const active=tab.dataset.source===source;
    tab.classList.toggle('active',active);tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;
    $(`source-${tab.dataset.source}`).hidden=!active;
  }
  updateSourceInspectorLabel();
}
const sourceInspectorNames={camera:'相機設定',files:'匯入設定',video:'取樣設定',screen:'擷取設定',merge:'整併設定'};
function updateSourceInspectorLabel() {
  const name=sourceInspectorNames[state.acquireSource]||'來源設定',panel=$(`source-${state.acquireSource}`),active=state.acquireSource==='camera'?$('cameraSettingsInspector'):panel?.querySelector('.source-inspector');
  if(active&&!active.id)active.id=`${state.acquireSource}SourceInspector`;
  $('toggleSourceInspector').textContent=`${name} ›`;$('toggleSourceInspector').setAttribute('aria-label',`開啟${name}`);$('toggleSourceInspector').setAttribute('aria-controls',active?.id||'');
  $('toggleImageTools').hidden=state.acquireSource!=='camera'||document.querySelector('.source-workspace').classList.contains('inspector-open');
}
function setSourceInspector(open,kind='source') {
  const value=!!open,workspace=document.querySelector('.source-workspace'),panel=$(`source-${state.acquireSource}`);
  document.querySelectorAll('.source-inspector').forEach(node=>node.classList.remove('active-inspector'));
  if(value){const target=state.acquireSource==='camera'&&kind==='image'?$('imageToolsInspector'):state.acquireSource==='camera'?$('cameraSettingsInspector'):panel?.querySelector('.source-inspector');target?.classList.add('active-inspector')}
  workspace?.classList.toggle('inspector-open',value);
  $('toggleSourceInspector').setAttribute('aria-expanded',String(value&&kind==='source'));$('toggleSourceInspector').hidden=value;
  $('toggleImageTools').setAttribute('aria-expanded',String(value&&kind==='image'));$('toggleImageTools').hidden=value||state.acquireSource!=='camera';
  $('inspectorBackdrop').hidden=!value;$('closeSourceInspector').hidden=!value;
  if(!value)updateSourceInspectorLabel();
}
function setupCameraInspectors() {
  const camera=$('source-camera'),settings=camera.querySelector(':scope>.source-inspector:not(#imageToolsInspector)'),tools=$('imageToolsInspector');settings.id='cameraSettingsInspector';
  const target=settings.querySelector('.camera-target-section'),processing=target.nextElementSibling,footnote=settings.querySelector('.inspector-footnote');tools.append(target,processing,footnote);
  const capture=settings.querySelector('.camera-capture-section');capture.classList.add('compact-camera-capture');capture.hidden=state.acquireSource!=='camera';document.querySelector('.acquired-tray').prepend(capture);
}
function renderAcquisitionAssets() {
  const assets=state.project?.assets||[],list=$('acquiredAssets');
  const valid=new Set(assets.map(asset=>asset.id));for(const id of state.acquireSelection)if(!valid.has(id))state.acquireSelection.delete(id);
  const thumbnails=new Map([...list.querySelectorAll('[data-asset-id]')].map(row=>[row.dataset.assetId,row.querySelector('img')]));
  list.replaceChildren();$('acquireTotal').textContent=number(assets.length);
  $('acquireAssetHint').textContent=assets.length?'點選素材即可開啟標註':'加入影像後即可開始標註';
  document.querySelectorAll('[data-destination-name]').forEach(node=>node.textContent=state.project?.name||'目前專案');
  for(const asset of [...assets].reverse()) {
    const row=button('','acquired-item',()=>safe(async()=>{if(state.busy||state.transitioning)return;await selectAsset(asset.id);await switchStage('annotate')}));
    row.dataset.assetId=asset.id;row.title=`${asset.name} · ${reviewNames[asset.review_state]||'待審核'} · 開啟標註`;
    row.classList.toggle('selected',state.acquireSelection.has(asset.id));
    const img=thumbnails.get(asset.id)||document.createElement('img');if(!img.getAttribute('src'))img.src=thumbnailURL(asset);img.alt=asset.name;img.loading='lazy';
    const copy=element('div');copy.append(element('b',asset.name),element('small',`${asset.width} × ${asset.height}`),element('small',reviewNames[asset.review_state]||'待審核','acquired-state '+asset.review_state));
    const remove=button('×','acquired-delete',event=>{event.stopPropagation();safe(()=>confirmDeleteAssets([asset]))});remove.title=`刪除 ${asset.name}`;remove.setAttribute('aria-label',`刪除 ${asset.name}`);
    const check=document.createElement('input');check.type='checkbox';check.className='acquired-select';check.checked=state.acquireSelection.has(asset.id);check.title=`選取 ${asset.name}`;check.setAttribute('aria-label',`選取 ${asset.name}`);check.onclick=event=>event.stopPropagation();check.onchange=()=>{if(check.checked)state.acquireSelection.add(asset.id);else state.acquireSelection.delete(asset.id);row.classList.toggle('selected',check.checked);updateAcquisitionControls()};
    row.append(img,copy,remove,check);list.append(row);
  }
  if(!assets.length)list.append(element('div','尚無素材 · 拍攝或匯入後，影像會顯示在這裡。','acquired-empty'));
  updateAcquisitionControls();
}
async function confirmDeleteAssets(assets) {
  if(!assets.length)return;
  await flushAllEdits();
  const body=element('div'),shapeCount=assets.reduce((sum,asset)=>sum+Number(asset.shape_count||0),0),approved=assets.filter(asset=>asset.review_state==='approved').length;
  body.append(element('p',assets.length===1?`即將刪除「${assets[0].name}」。`:`即將刪除所選的 ${assets.length} 張圖片。`));
  const details=element('ul');details.append(element('li',`包含 ${shapeCount} 個標註物件`),element('li',`${approved} 張已核准圖片`),element('li','圖片、標註與修訂紀錄會從目前專案移除'));body.append(details);
  const result=await formDialog({title:assets.length===1?'刪除這張素材？':'批量刪除素材？',body,confirm:`刪除 ${assets.length} 張`,eyebrow:'DELETE ASSETS',onSubmit:()=>api(projectPath('/assets'),'DELETE',{asset_ids:assets.map(asset=>asset.id),revisions:Object.fromEntries(assets.map(asset=>[asset.id,asset.revision]))})});
  if(!result)return;
  const removed=new Set(result.asset_ids||assets.map(asset=>asset.id));state.acquireSelection.clear();state.project=result.project||await api(projectPath());
  if(state.asset&&removed.has(state.asset.id)){saver.load(null);editor.clear();state.asset=null;if(state.project.assets.length)await loadAsset(state.project.assets[0].id);}
  renderAssetList();renderAcquisitionAssets();await loadProjects();status(`已從目前專案刪除 ${number(result.deleted||removed.size)} 張圖片。`);
}
function updateProcessingFields() {
  const mode=$('processingMode').value;
  $('processingParams').hidden=!['binary','adaptive'].includes(mode);
  $('processingThreshold').hidden=mode!=='binary';$('thresholdLabel').hidden=mode!=='binary';
  $('calibrateBackground').hidden=mode!=='classical';$('calibrationHint').hidden=mode!=='classical';
}
function updateVideoSource() {
  const path=$('videoPath').value.trim();
  $('videoSourceName').textContent=path?path.split(/[\\/]/).pop():'選擇要取樣的影片';
  $('videoSourceName').title=path;
  $('videoSourceDescription').textContent=path?'設定右側取樣間隔，即可將畫格加入目前專案。':'本地影片與剛完成的相機錄影，都可以擷取成影像。';
  $('sampleIntervalLabel').textContent=`每 ${$('videoInterval').value||'—'} 秒`;
}
async function switchStage(stage) {
  if(state.busy||state.transitioning)return;
  if(stage!=='library'&&!state.project)return;
  if(stage!=='acquire')setCameraPreviewExpanded(false);
  state.transitioning=true;editor.locked=true;updateNavigation();
  try {
    await flushAllEdits();
    if(stage!=='annotate'&&state.editorMode==='cvat')await switchEditor('builtin');
    for(const id of ['library','acquire','annotate','review','split','train','models','export'])$(id).hidden=id!==stage;
    state.stage=stage;editor.active=stage==='annotate';
    if(stage==='library')await loadProjects();
    if(stage==='acquire'){renderMergeList();renderAcquisitionAssets();await cameraStatus();}
    if(stage==='annotate'){renderAssetList();requestAnimationFrame(()=>editor.fit(false));}
    if(stage==='review'){await refreshProject();state.reviewSelection.clear();state.reviewPage=0;renderReview();}
    if(stage==='split'){await refreshProject();await loadTraining();await renderSplitPage();}
    if(stage==='train'||stage==='models')await loadTraining();
    if(stage==='export'){await refreshProject();renderExport();}
  } finally {state.transitioning=false;editor.locked=false;updateNavigation();editor.render();}
}
async function openProject(id) {
  if(state.busy||state.transitioning)return;
  if(state.autoCapture?.active&&state.autoCapture.projectId!==id)stopAutoCapture('已切換專案，自動擷取已停止。');
  state.transitioning=true;editor.locked=true;updateNavigation();
  try {
    await flushAllEdits();
    const project=await api(`/api/projects/${id}`);
    clearTimeout(state.trainingTimer);state.trainingTimer=null;state.training=null;state.selectedRun=null;state.selectedModel=null;state.yoloCompatibility=null;state.yoloCompatibilitySignature='';trainingMonitor.reset();trainingParameters.reset();
    state.project=project;state.asset=null;saver.load(null);editor.clear();state.reviewSelection.clear();state.acquireSelection.clear();
    $('shapeLabel').value='';$('cameraTargetLabel').value='';
    $('projectName').textContent=project.name;$('projectName').title=project.name;updateClassList();
    closeReleaseDrawer();resetValidation('請執行驗證，檢查目前專案及目標格式。');
    $('importReport').hidden=true;$('videoReport').hidden=true;$('mergeReport').hidden=true;
    for(const section of ['library','acquire','annotate','review','train','models','export'])$(section).hidden=section!==(project.assets.length?'annotate':'acquire');
    state.stage=project.assets.length?'annotate':'acquire';editor.active=state.stage==='annotate';
    renderAssetList();renderMergeList();renderExport();
    if(project.assets.length)await loadAsset(project.assets[0].id);else await cameraStatus();
    status(`已開啟 ${project.name} · ${number(project.assets.length)} 張影像`);
  } finally {state.transitioning=false;editor.locked=false;updateNavigation();editor.render();}
}
async function refreshProject() {
  if(!state.project)return;
  const project=await api(projectPath());state.project=project;
  $('projectName').textContent=project.name;updateClassList();renderAssetList();
}
async function loadProjects() {
  const data=await api('/api/projects');state.projects=data.projects||[];renderProjects();
}
function renderProjects() {
  const query=$('projectSearch').value.trim().toLowerCase();
  const projects=state.projects.filter(p=>p.name.toLowerCase().includes(query));
  $('projectCount').textContent=number(state.projects.length);
  $('totalImages').textContent=number(state.projects.reduce((n,p)=>n+Number(stats(p).total||0),0));
  $('totalApproved').textContent=number(state.projects.reduce((n,p)=>n+Number(stats(p).approved||0),0));
  const grid=$('projectGrid');grid.replaceChildren();
  for(const p of projects) {
    const card=element('article',undefined,'project-card'),count=stats(p);
    const open=button('','project-card-open',()=>safe(()=>openProject(p.id)));
    card.onclick=event=>{if(event.target===card)safe(()=>openProject(p.id))};
    open.setAttribute('aria-label',`開啟專案「${p.name}」`);
    const top=element('div',undefined,'project-card-top');top.append(element('span','▦'),element('span','本地專案','badge'));
    const footer=element('div',undefined,'project-card-footer');footer.append(element('span',`${number(count.total)} 張影像 · ${number(count.approved)} 已核准`),element('strong','開啟 →'));
    open.append(top,element('h2',p.name),element('p',`更新於 ${date(p.updated_at||p.created_at)}`),footer);
    const remove=button('×','project-delete danger',()=>safe(()=>confirmDeleteProject(p)));
    remove.title=`刪除專案「${p.name}」`;remove.setAttribute('aria-label',remove.title);
    card.append(open,remove);grid.append(card);
  }
  $('libraryEmpty').hidden=!!projects.length;
  if(query&&!projects.length)$('libraryEmpty').querySelector('h2').textContent='找不到符合的專案';
  else $('libraryEmpty').querySelector('h2').textContent='建立第一個視覺資料專案';
}
function updateClassList() {
  const selectors=[$('shapeLabel'),$('cameraTargetLabel')],current=selectors.map(select=>select.value);
  for(const select of selectors)select.replaceChildren(new Option('請先建立並選擇類別',''));
  $('classQuickList').replaceChildren();
  for(const name of state.project?.classes||[]) {
    for(const select of selectors)select.append(new Option(name,name));
    const choice=button(name,'class-choice',()=>{
      if(editor.locked)return;
      $('shapeLabel').value=name;updateObjectClassUI();
    });
    choice.dataset.className=name;choice.title=`選擇類別「${name}」`;
    $('classQuickList').append(choice);
  }
  selectors.forEach((select,index)=>{if((state.project?.classes||[]).includes(current[index]))select.value=current[index]});
  if(!$('classQuickList').children.length)$('classQuickList').append(element('span','尚無類別，請按「新增／刪除」建立第一個類別。','class-empty'));
  updateObjectClassUI();
}
function updateObjectClassUI(selected=editor.selected()) {
  const label=$('shapeLabel').value.trim();
  $('classSelectionHint').textContent=label?`僅下一個新物件使用：${label}`:'尚未選擇下一個物件的類別';
  $('classActionHint').textContent='建立完成後自動清除；既有物件只在下方該列修改。';
  document.querySelectorAll('.class-choice').forEach(choice=>{
    const active=choice.dataset.className===label;choice.classList.toggle('active',active);
    choice.setAttribute('aria-pressed',String(active));
    choice.title=`將下一個新物件設為「${choice.dataset.className}」`;
  });
}
async function formDialog({title,body,confirm='確定',onSubmit,eyebrow='WORKSPACE'}) {
  return new Promise(resolve=>{
    const dialog=$('formDialog');$('dialogTitle').textContent=title;$('dialogEyebrow').textContent=eyebrow;
    $('dialogBody').replaceChildren(body);$('confirmDialog').textContent=confirm;$('confirmDialog').hidden=!onSubmit;
    $('cancelDialog').textContent=onSubmit?'取消':'關閉';
    let processing=false;
    const finish=value=>{if(processing)return;dialog.close();resolve(value)};
    $('closeDialog').onclick=()=>finish(null);$('cancelDialog').onclick=()=>finish(null);
    dialog.oncancel=event=>{if(processing)event.preventDefault();else resolve(null)};
    $('dialogForm').onsubmit=async event=>{
      event.preventDefault();if(processing||!onSubmit)return;
      processing=true;$('confirmDialog').disabled=true;$('cancelDialog').disabled=true;$('closeDialog').disabled=true;
      try{const result=await onSubmit();processing=false;dialog.close();resolve(result)}catch(error){toast(error.message,true)}
      finally{processing=false;$('confirmDialog').disabled=false;$('cancelDialog').disabled=false;$('closeDialog').disabled=false;}
    };
    dialog.showModal();setTimeout(()=>body.querySelector('input,textarea,select')?.focus(),30);
  });
}
async function createProject() {
  const body=element('div');body.append(element('p','為這份資料設定清楚的名稱，例如產品、場域或檢測任務。'));
  const input=document.createElement('input');input.placeholder='例如：零件表面瑕疵';input.maxLength=100;input.required=true;body.append(input);
  const result=await formDialog({title:'建立本地專案',body,confirm:'建立專案',onSubmit:async()=>{if(!input.value.trim())throw Error('請輸入專案名稱。');return api('/api/projects','POST',{name:input.value.trim()})}});
  if(result){await loadProjects();await openProject(result.id)}
}
async function confirmDeleteProject(project) {
  if(state.project?.id===project.id) {
    if(state.recording)throw Error('請先停止錄影，再刪除目前專案。');
    await flushAllEdits();
  }
  const count=stats(project),body=element('div');
  body.append(element('p',`確定要刪除「${project.name}」嗎？`));
  const warning=element('div',undefined,'delete-warning');
  warning.append(element('b',`${number(count.total)} 張影像`),element('span','專案內的原圖副本、標註、審核與修訂紀錄都會一併刪除。已匯出的資料集會保留。'));
  body.append(warning);
  const result=await formDialog({title:'刪除本地專案',body,confirm:'刪除專案',eyebrow:'DELETE PROJECT',
    onSubmit:()=>api(`/api/projects/${project.id}`,'DELETE',{revision:project.revision})});
  if(!result)return;
  if(state.project?.id===project.id) {
    if(state.autoCapture?.active)stopAutoCapture('目前專案已刪除，自動擷取已停止。');
    state.project=null;state.asset=null;state.reviewSelection.clear();state.acquireSelection.clear();
    state.cameraTarget=null;state.cameraTargetDraft=[];saver.load(null);editor.clear();editor.active=false;
    $('projectName').textContent='尚未開啟專案';$('projectName').title='';updateClassList();
  }
  await loadProjects();updateNavigation();
  status(`已刪除專案 ${project.name}`);toast(`已刪除「${project.name}」。`);
}
async function renameProject() {
  await flushAllEdits();const body=element('div'),input=document.createElement('input');input.value=state.project.name;input.maxLength=100;body.append(element('label','專案名稱'),input);
  const project=await formDialog({title:'編輯專案名稱',body,confirm:'儲存名稱',onSubmit:()=>api(projectPath(),'PATCH',{name:input.value.trim()})});
  if(project){state.project=project;$('projectName').textContent=project.name;await loadProjects()}
}
async function manageClasses() {
  await flushAllEdits();
  const usage=await api(projectPath('/classes')),body=element('div');
  body.append(element('p','每行一個類別。刪除使用中的類別時，必須明確選擇：把既有物件改成另一類，或連同這些標註物件一起刪除。'));
  const label=element('label','專案類別（每行一個）');
  const input=document.createElement('textarea');input.rows=7;input.value=usage.classes.map(item=>item.name).join('\n');
  const summary=element('div',undefined,'class-management-summary');
  body.append(label,input,element('label','目前使用情況'),summary);
  const decisions=new Map();
  const values=()=>[...new Set(input.value.split(/\r?\n/).map(value=>value.trim()).filter(Boolean))];
  const render=()=>{
    const desired=values();summary.replaceChildren();decisions.clear();
    for(const item of usage.classes) {
      const row=element('div',undefined,'class-management-row');
      const copy=element('div');copy.append(element('b',item.name),element('small',item.object_count?`${number(item.object_count)} 個物件 · ${number(item.image_count)} 張影像`:'尚未被標註使用'));
      row.append(copy);
      if(desired.includes(item.name))row.append(element('span','保留','badge approved'));
      else if(!item.object_count)row.append(element('span','儲存後刪除','badge rejected'));
      else {
        const select=document.createElement('select');select.setAttribute('aria-label',`刪除「${item.name}」前的標註處理方式`);
        select.append(new Option('選擇標註處理方式',''));
        for(const name of desired)select.append(new Option(`將全部 ${number(item.object_count)} 個物件改為 ${name}`,name));
        select.append(new Option(`刪除這 ${number(item.object_count)} 個標註物件`,'__delete_objects__'));
        select.value=decisions.get(item.name)||'';
        select.onchange=()=>{decisions.set(item.name,select.value);row.classList.toggle('delete-objects',select.value==='__delete_objects__')};
        row.classList.add('requires-replacement');row.append(select);
      }
      summary.append(row);
    }
    const added=desired.filter(name=>!usage.classes.some(item=>item.name===name));
    for(const name of added) {
      const row=element('div',undefined,'class-management-row');
      const copy=element('div');copy.append(element('b',name),element('small','新類別'));
      row.append(copy,element('span','新增','badge approved'));summary.append(row);
    }
    if(!usage.classes.length&&!added.length)summary.append(element('div','尚無類別。可在上方輸入第一個類別。','class-empty'));
  };
  input.addEventListener('input',render);render();
  const currentAssetId=state.asset?.id;
  const project=await formDialog({title:'管理物件類別',body,confirm:'儲存並套用',onSubmit:()=>{
    const desired=values(),mapping={},deleteObjects=[];
    for(const item of usage.classes)if(!desired.includes(item.name)&&item.object_count) {
      const decision=decisions.get(item.name);
      if(!decision)throw Error(`請選擇刪除「${item.name}」後如何處理既有標註。`);
      if(decision==='__delete_objects__')deleteObjects.push(item.name);else mapping[item.name]=decision;
    }
    return api(projectPath('/classes'),'POST',{classes:desired,replacements:mapping,delete_objects:deleteObjects,revision:usage.revision});
  }});
  if(project){
    state.project=project;updateClassList();editor.render();
    if(currentAssetId&&project.class_update?.asset_ids?.includes(currentAssetId))await loadAsset(currentAssetId);
    else renderAssetList();
    await loadProjects();
    const change=project.class_update||{};
    const operations=[];
    if(change.changed_objects)operations.push(`改派 ${number(change.changed_objects)} 個物件`);
    if(change.deleted_objects)operations.push(`刪除 ${number(change.deleted_objects)} 個標註物件`);
    const details=operations.length?`；已在 ${number(change.affected_images)} 張影像中${operations.join('、')}`:'。';
    toast(`類別清單已儲存${details}`);
  }
}

function filteredAssets() {
  const query=$('assetSearch').value.toLowerCase(),filter=$('assetFilter').value;
  return (state.project?.assets||[]).filter(a=>a.name.toLowerCase().includes(query)&&(filter==='all'||a.review_state===filter));
}
function renderAssetList() {
  const list=$('assetList'),items=filteredAssets();
  const thumbnails=new Map([...list.querySelectorAll('[data-asset-id]')].map(row=>[row.dataset.assetId,row.querySelector('img')]));
  list.replaceChildren();$('assetCount').textContent=number(state.project?.assets.length||0);
  for(const asset of items) {
    const row=button('','asset-row'+(asset.id===state.asset?.id?' active':''),()=>safe(()=>selectAsset(asset.id)));
    row.dataset.assetId=asset.id;
    const img=thumbnails.get(asset.id)||document.createElement('img');
    if(!img.getAttribute('src'))img.src=thumbnailURL(asset);img.alt='';img.loading='lazy';
    const copy=element('div');copy.append(element('b',asset.name),element('small',`${asset.width} × ${asset.height} · ${asset.shape_count||0} 個物件`));
    row.title=asset.name;row.append(img,copy,element('span',undefined,'asset-state-dot '+asset.review_state));list.append(row);
  }
  if(!items.length)list.append(element('p',state.project?.assets.length?'沒有符合條件的圖片。':'匯入圖片後會顯示在此。','empty-list'));
  updateAssetHeader();renderAcquisitionAssets();
}
function updateAssetHeader() {
  const asset=state.asset,items=filteredAssets(),index=items.findIndex(a=>a.id===asset?.id);
  $('assetTitle').textContent=asset?.name||'尚未選擇影像';$('assetTitle').title=asset?.name||'';
  $('assetPosition').textContent=`${index+1} / ${items.length}`;
  const review=asset?.review_state||'pending';$('assetReviewBadge').textContent=reviewNames[review];$('assetReviewBadge').className='badge '+review;
  $('previousAsset').disabled=index<=0;$('nextAsset').disabled=index<0||index>=items.length-1;
  $('previousAssetTop').disabled=index<=0;$('nextAssetTop').disabled=index<0||index>=items.length-1;
  updateSaveControl();
}
async function loadAsset(id) {
  const asset=await api(projectPath(`/assets/${id}`));asset.url=imageURL(asset);state.asset=asset;saver.load(asset);editor.load(asset);
  updateAssetHeader();renderAssetList();
}
async function selectAsset(id) {
  if(id===state.asset?.id||state.busy||state.transitioning)return;
  state.transitioning=true;editor.locked=true;updateNavigation();
  try{await flushAllEdits();await loadAsset(id)}finally{state.transitioning=false;editor.locked=false;editor.render();updateNavigation();}
}
async function navigateAsset(direction) {
  const items=filteredAssets(),index=items.findIndex(a=>a.id===state.asset?.id),next=items[index+direction];
  if(next)await selectAsset(next.id);
}

// Every state that cannot progress on its own tells the user what to do next.
const cvatGuides={
  reboot_required:{title:'需要你先重新啟動 Windows',steps:[
    '儲存所有未完成的工作，包含其他程式裡的。',
    '重新啟動 Windows（選「重新啟動」，不要關機再開機）。待安裝的更新會在這時裝完，可能需要幾分鐘。',
    '開機後回到這個畫面，按「繼續準備」，就會接著安裝 WSL 2。']},
  uac:{title:'Windows 正在等你確認',steps:[
    'Windows 的「使用者帳戶控制」視窗可能被其他視窗擋住，請找出來。',
    '按「是」允許準備環境。整個流程只有這一步需要管理員權限。']},
  waiting_action:{title:'上一步沒有完成，環境未變更',steps:[
    '剛才的管理員確認沒有通過，所以沒有安裝任何東西。',
    '按「繼續準備」重試，跳出確認視窗時選「是」。']},
  blocked:{title:'這台電腦目前無法安裝 CVAT',steps:[
    '需求：x64 的 Windows 10 22H2 或 Windows 11 23H2 以上、8 GB 以上記憶體，且 BIOS／UEFI 已啟用虛擬化。',
    '這不影響標註工作，可以直接按下方「返回內建編輯器」繼續使用。']},
  error:{title:'準備過程中斷了',steps:[
    '已下載完成的內容會保留，重試不會從頭開始。',
    '按「繼續準備」重試；若持續失敗，可先用內建編輯器標註。']},
};
const cvatWaitNotes={
  downloading_docker:'正在從 Docker 官方網站下載安裝檔（數百 MB）。',
  installing_docker:'正在安裝 Docker Desktop。',
  waiting_docker:'正在等待 Docker Desktop 啟動。首次啟動時它可能要求你同意授權條款，同意後這裡會自動繼續。',
  download:'正在下載 CVAT 容器映像檔（數 GB，這是整個流程最久的一段）。',
  services:'正在啟動 CVAT 服務。',
};
function cvatGuideFor(snapshot) {
  if(snapshot.ready)return null;
  if(snapshot.reboot_required)return cvatGuides.reboot_required;
  const note=cvatWaitNotes[snapshot.phase];
  if(note&&snapshot.busy)return {title:note,tone:'busy',steps:[
    '這一段沒有進度條，可能需要數分鐘到數十分鐘。',
    '只要上方的計時還在跳動，就代表仍在進行中，不是當機。',
    '這段期間可以讓視窗留在背景，不需要一直看著。']};
  return cvatGuides[snapshot.phase]||null;
}
function renderCvatGuidance(snapshot) {
  const guide=cvatGuideFor(snapshot),panel=$('cvatGuidance');
  panel.hidden=!guide;
  if(!guide)return;
  panel.className='cvat-guidance '+(guide.tone||'attention');
  $('cvatGuidanceTitle').textContent=guide.title;
  $('cvatGuidanceSteps').replaceChildren(...guide.steps.map(text=>element('li',text)));
}
function renderCvatStatus(snapshot) {
  state.cvatPollBaseText=snapshot.text||'CVAT 尚未準備。';
  state.cvatPollBusy=!!snapshot.busy;
  state.cvatPollStartedAt=Number(snapshot.started_at);
  if(!Number.isFinite(state.cvatPollStartedAt))state.cvatPollStartedAt=null;
  clearInterval(state.cvatElapsedTimer);
  updateCvatSetupText({busy:state.cvatPollBusy,startedAt:state.cvatPollStartedAt});

  $('cvatSteps').replaceChildren(...(snapshot.steps||[]).map(step=>element('div',step.label,`cvat-step ${step.state||''}`)));
  renderCvatGuidance(snapshot);
  if(!state.cvatRebootScheduled){
    $('rebootNow').hidden=!snapshot.can_reboot;
    $('rebootNow').textContent='立即重新啟動 Windows';
    $('rebootNow').title='儲存工作後，由這裡重新啟動並安裝待處理的更新。';
  }
  $('setupCvat').hidden=!!snapshot.ready;
  $('setupCvat').disabled=!!snapshot.busy||snapshot.can_setup===false;
  if(snapshot.reboot_required){
    $('setupCvat').textContent='重新啟動後才能繼續';
    $('setupCvat').title='必須先重新啟動 Windows，回到此頁後這顆按鈕就會解鎖。';
  } else if(snapshot.resume) {
    $('setupCvat').textContent='繼續準備';
    $('setupCvat').title='Windows 已重啟，點此繼續。';
  } else if(snapshot.busy){
    $('setupCvat').textContent='正在準備…';
    $('setupCvat').title='安裝中';
  } else {
    $('setupCvat').textContent='安裝並啟用';
    $('setupCvat').title='啟用 CVAT。';
  }

  if(snapshot.busy&&state.cvatPollStartedAt&&state.editorMode==='cvat'){
    state.cvatElapsedTimer=setInterval(()=>updateCvatSetupText({busy:state.cvatPollBusy,startedAt:state.cvatPollStartedAt}),1000);
  }
}
async function openCvat() {
  if(state.cvatLaunching)return;
  state.cvatLaunching=true;
  clearTimeout(state.cvatPoll);state.cvatPoll=null;
  clearInterval(state.cvatElapsedTimer);
  $('retryCvat').hidden=true;$('setupCvat').hidden=true;
  $('cvatGuidance').hidden=true;
  state.cvatPollStartedAt=Date.now()/1000;
  state.cvatPollBaseText='正在啟動 CVAT 並等待服務就緒…';state.cvatPollBusy=true;
  updateCvatSetupText();state.cvatElapsedTimer=setInterval(()=>updateCvatSetupText(),1000);
  try {
    if(!state.nativeBridge)throw Error('內建 CVAT 僅能在 Windows 桌面版開啟。');
    const projectId=state.project.id;
    const result=await pollJob(await api('/api/cvat/launch','POST',{project_id:projectId,asset_id:state.asset?.id}),null,current=>{
      state.cvatPollBaseText=current.message||'正在啟動 CVAT…';updateCvatSetupText();
    });
    if(state.editorMode!=='cvat'||state.project?.id!==projectId)return;
    const opened=await new Promise(resolve=>state.nativeBridge.openCvat(result.ticket,resolve));
    if(!opened)throw Error('CVAT 視窗切換失敗，請重試。');
    state.cvatWasOpened=true;
    state.cvatPollBaseText='CVAT 已開啟。';
    if(result.skipped_masks>0)toast(`有 ${result.skipped_masks} 個遮罩標註沒有送進 CVAT（CVAT 匯入僅支援方框與多邊形）。原始遮罩仍保留在內建編輯器，不會遺失。`,true);
  } catch(error) {
    state.cvatPollBaseText=`CVAT 開啟失敗：${error.message}`;
    $('retryCvat').hidden=false;
    throw error;
  } finally {
    state.cvatLaunching=false;state.cvatPollBusy=false;
    clearInterval(state.cvatElapsedTimer);state.cvatElapsedTimer=null;updateCvatSetupText();
  }
}
async function refreshCvat({openWhenReady=false}={}) {
  const snapshot=await api('/api/cvat/status');renderCvatStatus(snapshot);
  if(snapshot.ready&&openWhenReady&&state.editorMode==='cvat'){clearTimeout(state.cvatPoll);state.cvatPoll=null;await openCvat();return;}
  if(state.editorMode!=='cvat'||snapshot.ready){
    clearTimeout(state.cvatPoll);state.cvatPoll=null;
    return;
  }
  clearTimeout(state.cvatPoll);
  const interval = snapshot.busy ? 1200 : 5000;
  state.cvatPoll=setTimeout(()=>safe(()=>refreshCvat({openWhenReady:true})),interval);
}
async function switchEditor(mode) {
  $('retryCvat').hidden=true;
  if(mode==='builtin') {
    clearTimeout(state.cvatPoll);state.cvatPoll=null;
    clearInterval(state.cvatElapsedTimer);state.cvatElapsedTimer=null;
    state.editorMode='builtin';$('editorSelector').value='builtin';
    $('cvatSetup').hidden=true;document.querySelector('#annotate>.editor-layout').hidden=false;editor.active=state.stage==='annotate';
    requestAnimationFrame(()=>editor.fit(false));return;
  }
  if(mode==='labelme') {
    await flushAllEdits();
    if(!state.nativeBridge?.openLabelme){$('editorSelector').value=state.editorMode;throw Error('Labelme 整合需要更新後的 Windows 桌面版。');}
    const error=await new Promise(resolve=>state.nativeBridge.openLabelme(state.project.id,state.asset?.id||'',resolve));
    if(error){$('editorSelector').value=state.editorMode;throw Error(`Labelme 無法開啟：${error}`);}
    state.editorMode='labelme';$('editorSelector').value='labelme';editor.active=false;
    return;
  }
  await flushAllEdits();state.editorMode='cvat';$('editorSelector').value='cvat';editor.active=false;
  document.querySelector('#annotate>.editor-layout').hidden=true;$('cvatSetup').hidden=false;
  const snapshot=await api('/api/cvat/status');renderCvatStatus(snapshot);
  if(snapshot.ready)await openCvat();
  else await refreshCvat({openWhenReady:true});
}
window.workbenchExternalSync=async({source,asset_id})=>{
  try {
    if(source==='cvat') {
      const result=await pollJob(await api('/api/cvat/import','POST',{project_id:state.project.id}));
      toast(result.updated?`已同步 ${result.updated} 張圖片，修改內容已送往待審核。`:'CVAT 沒有新的標註變更。');
    }
    await refreshProject();
    const current=asset_id||state.asset?.id;
    if(current&&state.project.assets.some(asset=>asset.id===current))await loadAsset(current);
    await switchEditor('builtin');state.cvatWasOpened=false;
    state.nativeBridge.finishEditorSync('');
  } catch(error) {state.nativeBridge.finishEditorSync(String(error.message||error));}
};
window.workbenchExternalNavigate=target=>safe(async()=>{
  if(target==='review')await switchStage('review');
  else if(target!=='builtin')await switchEditor(target);
});

async function nativeChoose(kind) {
  try {const result=await api('/api/dialog','POST',{kind});return result.paths||[];}
  catch(error){toast(`${error.message}\n也可以直接在路徑欄位輸入完整路徑。`,true);return [];}
}
class OperationStopped extends Error {constructor(message='工作已停止。'){super(message);this.silent=true;}}
let acquisitionTask=null,acquisitionTaskHideTimer=null;
function taskSeconds(task) {
  const end=task.pausedAt||performance.now();return Math.max(0,(end-task.started-task.pausedTotal)/1000).toFixed(1)+' 秒';
}
function renderAcquisitionTask(task) {
  if(acquisitionTask!==task)return;
  const panel=$('acquisitionTask');panel.hidden=false;panel.classList.toggle('paused',task.paused);panel.classList.toggle('stopping',task.stopping);panel.classList.remove('completing');
  $('acquisitionTaskTitle').textContent=task.title;$('acquisitionTaskMessage').textContent=task.message;$('acquisitionTaskElapsed').textContent=taskSeconds(task);
  if(Number.isFinite(task.progress))$('acquisitionTaskProgress').value=Math.max(0,Math.min(100,task.progress));else $('acquisitionTaskProgress').removeAttribute('value');
  $('pauseAcquisitionTask').textContent=task.paused?'繼續':'暫停';$('pauseAcquisitionTask').disabled=task.stopping;
  $('stopAcquisitionTask').disabled=task.stopping;
}
function beginTimedTask(title) {
  clearTimeout(acquisitionTaskHideTimer);
  if(acquisitionTask)finishTimedTask(acquisitionTask,'stopped','已由下一項工作取代。',0);
  const task={title,message:'準備執行…',progress:null,started:performance.now(),pausedAt:0,pausedTotal:0,paused:false,stopping:false,stopped:false,jobId:null,timer:null};
  acquisitionTask=task;task.timer=setInterval(()=>renderAcquisitionTask(task),100);renderAcquisitionTask(task);return task;
}
function updateTimedTask(task,message,progress=null) {if(acquisitionTask!==task||task.stopped)return;task.message=message;if(progress!==undefined)task.progress=progress;renderAcquisitionTask(task);}
function finishTimedTask(task,state='complete',message='已完成',delay=1800) {
  if(acquisitionTask!==task)return;clearInterval(task.timer);task.stopped=true;task.paused=false;task.progress=state==='complete'?100:task.progress;task.message=message;
  renderAcquisitionTask(task);$('pauseAcquisitionTask').disabled=true;$('stopAcquisitionTask').disabled=true;
  acquisitionTaskHideTimer=setTimeout(()=>{if(acquisitionTask!==task)return;$('acquisitionTask').classList.add('completing');setTimeout(()=>{if(acquisitionTask===task){$('acquisitionTask').hidden=true;$('acquisitionTask').classList.remove('completing');acquisitionTask=null}},230)},delay);
}
async function taskCheckpoint(task) {
  if(task.stopping||task.stopped)throw new OperationStopped();
  while(task.paused&&!task.stopping)await new Promise(resolve=>setTimeout(resolve,120));
  if(task.stopping||task.stopped)throw new OperationStopped();
}
async function runTimedTask(title,work) {
  const task=beginTimedTask(title);
  try{const result=await work(task);await taskCheckpoint(task);finishTimedTask(task,'complete','完成 · '+taskSeconds(task));return result;}
  catch(error){if(error instanceof OperationStopped){finishTimedTask(task,'stopped','已停止 · '+taskSeconds(task),2400);throw error}finishTimedTask(task,'failed','失敗 · '+error.message,4500);throw error}
}
$('pauseAcquisitionTask').onclick=()=>safe(async()=>{
  const task=acquisitionTask;if(!task||task.stopping||task.stopped)return;
  if(task.jobId)await api(`/api/jobs/${task.jobId}/${task.paused?'resume':'pause'}`,'POST',{});
  if(task.paused){task.pausedTotal+=performance.now()-task.pausedAt;task.pausedAt=0;task.paused=false;task.message='繼續執行…'}else{task.paused=true;task.pausedAt=performance.now();task.message='已暫停';}
  renderAcquisitionTask(task);
});
$('stopAcquisitionTask').onclick=()=>safe(async()=>{
  const task=acquisitionTask;if(!task||task.stopping||task.stopped)return;task.stopping=true;task.paused=false;task.message='正在安全停止…';renderAcquisitionTask(task);
  if(task.jobId)await api(`/api/jobs/${task.jobId}/cancel`,'POST',{});else finishTimedTask(task,'stopped','已停止等待 · '+taskSeconds(task),2400);
});
async function pollJob(job,onDone,onProgress) {
  let current=job;const id=job.id||job.job_id||job.job;
  if(!id) return onDone ? onDone(job.result||job) : job.result||job;
  const tracker=state.stage==='acquire'?acquisitionTask:null;if(tracker)tracker.jobId=id;
  $('jobStatus').hidden=false;
  try {
    while(true) {
      if(onProgress)onProgress(current);
      $('jobMessage').textContent=current.message||'正在處理工作…';
      if(Number.isFinite(current.progress))$('jobProgress').value=current.progress;else $('jobProgress').removeAttribute('value');
      if(tracker){tracker.paused=current.state==='paused';tracker.stopping=current.state==='stopping';updateTimedTask(tracker,current.message||'正在處理工作…',current.progress)}
      if(current.state==='succeeded'){if(onDone)await onDone(current.result);return current.result;}
      if(current.state==='failed')throw Error(typeof current.error==='string'?current.error:current.error?.message||current.message||'工作未完成。');
      if(current.state==='cancelled')throw new OperationStopped(current.message||'工作已停止。');
      if(tracker)await taskCheckpoint(tracker);
      await new Promise(resolve=>setTimeout(resolve,600));current=await api(`/api/jobs/${id}`);
    }
  } finally {$('jobStatus').hidden=true;}
}
function readable(value) {
  if(typeof value==='string')return value;
  if(value?.message)return `${value.source?value.source+'：':''}${value.message}`;
  return JSON.stringify(value);
}
function showImportReport(id,result) {
  const report=$(id);report.hidden=false;report.replaceChildren();
  const added=result?.imported??result?.added??result?.count??result?.assets?.length??result?.records?.length;
  report.append(element('strong',`匯入已完成${added!==undefined?' · '+added+' 張影像':''}`));
  const issues=result?.issues||result?.report?.issues||[];
  for(const issue of issues)report.append(element('div',readable(issue)));
  const conflicts=result?.conflicts||[];
  if(result?.duplicates)report.append(element('div',`${result.duplicates} 張重複原圖已略過。`));
  if(Array.isArray(conflicts))for(const conflict of conflicts)report.append(element('div',`來源差異：${readable(conflict)}`));
  else if(conflicts)report.append(element('div',`來源差異：${readable(conflicts)}`));
  if(!issues.length&&!conflicts.length&&!result?.duplicates)report.append(element('div','未回報匯入問題。'));
}
async function afterAcquisition(result,reportId='importReport') {
  await refreshProject();showImportReport(reportId,result);
  if(!state.asset&&state.project.assets.length)await loadAsset(state.project.assets[0].id);
  await loadProjects();
  const added=result?.imported??result?.added??result?.count;
  status(`${added!==undefined?'已加入 '+number(added)+' 張影像 · ':''}專案共 ${number(state.project.assets.length)} 張`);
}
async function importPaths(paths) {
  if(!paths.length)throw Error('請選擇檔案或輸入至少一個完整路徑。');
  await flushAllEdits();const job=await api(projectPath('/import'),'POST',{paths});const result=await pollJob(job);await afterAcquisition(result);
}
function renderMergeList() {
  const list=$('mergeProjects');list.replaceChildren();
  for(const project of state.projects.filter(p=>p.id!==state.project?.id)) {
    const label=element('label'),check=document.createElement('input');check.type='checkbox';check.value=project.id;label.append(check,element('span',`${project.name} · ${number(stats(project).total)} 張`));list.append(label);
  }
  if(!list.children.length)list.append(element('p','目前沒有其他專案可合併。'));
  list.onchange=updateAcquisitionControls;updateAcquisitionControls();
}

function cameraTargetDimensions() {
  const details=state.cameraDetails||{};return {width:Number(details.width||0),height:Number(details.height||0)};
}
function cameraTargetShape() {
  const target=state.cameraTarget,{width,height}=cameraTargetDimensions();
  return target&&target.width===width&&target.height===height?structuredClone(target.shape):null;
}
function cameraTargetPayload() {
  return $('cameraTargetAttach').checked?cameraTargetShape():null;
}
function targetPoint(event) {
  const svg=event.currentTarget,matrix=svg.getScreenCTM();if(!matrix)return null;
  const point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;
  const local=point.matrixTransform(matrix.inverse()),{width,height}=cameraTargetDimensions();
  if(local.x<0||local.y<0||local.x>width||local.y>height)return null;
  return {x:Math.max(0,Math.min(width,local.x)),y:Math.max(0,Math.min(height,local.y))};
}
function targetSvgShape(svg,shape,className='') {
  if(!shape||shape.type==='mask')return;
  const node=document.createElementNS('http://www.w3.org/2000/svg',shape.type==='rectangle'?'rect':'polygon');
  if(shape.type==='rectangle')for(const key of ['x','y','width','height'])node.setAttribute(key,shape[key]);
  else node.setAttribute('points',shape.points.map(point=>point.join(',')).join(' '));
  if(className)node.setAttribute('class',className);svg.append(node);
}
function renderTargetMask(data,width,height) {
  let scratch=null;
  if(data&&width&&height){scratch=document.createElement('canvas');scratch.width=width;scratch.height=height;const context=scratch.getContext('2d'),image=context.createImageData(width,height);for(let index=0;index<data.length;index++)if(data[index]){const offset=index*4;image.data[offset]=76;image.data[offset+1]=224;image.data[offset+2]=199;image.data[offset+3]=210}context.putImageData(image,0,0)}
  for(const id of ['cameraRawTargetMask','cameraOutputTargetMask']){
    const canvas=$(id);canvas.width=width||1;canvas.height=height||1;const context=canvas.getContext('2d');context.clearRect(0,0,canvas.width,canvas.height);
    if(scratch)context.drawImage(scratch,0,0);
  }
}
function renderCameraTarget(previewShape=null,maskData=null) {
  const {width,height}=cameraTargetDimensions(),shape=previewShape||cameraTargetShape();
  for(const id of ['cameraRawTargetOverlay','cameraOutputTargetOverlay']){const svg=$(id);svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${Math.max(1,width)} ${Math.max(1,height)}`);targetSvgShape(svg,shape,previewShape?'target-draft':'');if(state.cameraTargetDraft.length)targetSvgShape(svg,{type:'polygon',points:state.cameraTargetDraft},'target-draft')}
  if(maskData)renderTargetMask(maskData,width,height);else if(shape?.type==='mask')renderTargetMask(decodeMask(shape.counts,width,height),width,height);else renderTargetMask(null,width,height);
  const has=!!cameraTargetShape();$('clearCameraTarget').disabled=!has||state.busy;
  document.querySelector('.camera-target-section').classList.toggle('target-active',has);
  if(has)$('cameraTargetStatus').textContent=`${shapeNames[shape.type]||shape.type} · ${shape.label} · 同時用於處理範圍與擷取標註`;
}
function updateCameraTargetTool() {
  const tool=$('cameraTargetTool').value,enabled=state.camera&&tool!=='none';
  $('cameraTargetBrushFields').hidden=!['mask','mask_erase'].includes(tool);
  document.querySelectorAll('[data-target-hint]').forEach(node=>node.classList.toggle('active',node.dataset.targetHint===tool));
  if(enabled&&$('previewLayout').value!=='compare'&&!$('showRawFrame').checked){$('showRawFrame').checked=true;updatePreviewLayout()}
  updateCameraTargetEditingSurface();
  if(!state.camera)$('cameraTargetStatus').textContent='啟動相機後即可繪製目標區域。';
  else if(tool==='rectangle')$('cameraTargetStatus').textContent='Q · 在原圖拖曳 Bounding Box。';
  else if(tool==='polygon')$('cameraTargetStatus').textContent='W · 逐點建立 Polygon，雙擊或 Enter 完成。';
  else if(tool==='mask')$('cameraTargetStatus').textContent='E · 在原圖拖曳繪製 Mask。';
  else if(tool==='mask_erase')$('cameraTargetStatus').textContent='Shift+E · 拖曳擦除既有 Mask。';
  else if(!cameraTargetShape())$('cameraTargetStatus').textContent='尚未設定目標區域。';
  renderCameraTarget();
}
function commitCameraTarget(shape) {
  const {width,height}=cameraTargetDimensions(),label=$('cameraTargetLabel').value.trim();if(!label){toast('請先在專案類別中新增並選擇一個類別。',true);return}
  state.cameraTarget={width,height,shape:{...shape,label}};state.cameraTargetDraft=[];state.cameraTargetGesture=null;renderCameraTarget();
  if($('cameraTargetUseProcessing').checked&&state.camera&&!state.busy&&!$('applyProcessing').disabled)$('applyProcessing').click();
}
function clearCameraTarget(message='尚未設定目標區域。') {
  state.cameraTarget=null;state.cameraTargetDraft=[];state.cameraTargetGesture=null;renderCameraTarget();$('cameraTargetStatus').textContent=message;
  if(state.camera&&$('cameraTargetUseProcessing').checked&&!state.busy&&!$('applyProcessing').disabled)$('applyProcessing').click();
}
function updateCameraTargetEditingSurface() {
  const enabled=state.camera&&$('cameraTargetTool').value!=='none'&&!state.busy,compare=$('previewLayout').value==='compare';
  $('cameraRawTargetOverlay').classList.toggle('editing',enabled&&compare);
  $('cameraOutputTargetOverlay').classList.toggle('editing',enabled&&!compare&&$('showRawFrame').checked);
  $('cameraRawTargetOverlay').classList.toggle('target-overlay-view',!enabled||!compare);
  $('cameraOutputTargetOverlay').classList.toggle('target-overlay-view',!enabled||compare||!$('showRawFrame').checked);
}
function installCameraTargetEvents() {
  for(const overlay of [$('cameraRawTargetOverlay'),$('cameraOutputTargetOverlay')]){
  overlay.onpointerdown=event=>{
    if(!state.camera||event.button!==0)return;const tool=$('cameraTargetTool').value,point=targetPoint(event);if(!point||tool==='none')return;event.preventDefault();
    if(tool!=='mask_erase'&&!$('cameraTargetLabel').value){toast('請先新增並選擇標註類別。',true);return}
    if(tool==='polygon'){if(event.detail>1)return;state.cameraTargetDraft.push([point.x,point.y]);renderCameraTarget();return}
    if(tool==='rectangle')state.cameraTargetGesture={type:tool,start:point};
    else if(['mask','mask_erase'].includes(tool)){
      const {width,height}=cameraTargetDimensions();if(width*height>16777216){toast('Mask 編輯上限為 1,677 萬像素。',true);return}
      const existing=cameraTargetShape();if(tool==='mask_erase'&&existing?.type!=='mask'){toast('請先建立一個 Mask，再使用橡皮擦。',true);return}
      const data=existing?.type==='mask'?decodeMask(existing.counts,width,height):new Uint8Array(width*height);
      state.cameraTargetGesture={type:tool,last:point,data};brush(data,width,height,point,point,Number($('cameraTargetBrushRadius').value)||24,tool==='mask_erase');renderCameraTarget(null,data);
    }
    try{overlay.setPointerCapture(event.pointerId)}catch{}
  };
  overlay.onpointermove=event=>{
    const gesture=state.cameraTargetGesture,point=targetPoint(event);if(!gesture||!point)return;
    const {width,height}=cameraTargetDimensions();
    if(gesture.type==='rectangle'){const x=Math.min(point.x,gesture.start.x),y=Math.min(point.y,gesture.start.y);renderCameraTarget({type:'rectangle',x,y,width:Math.abs(point.x-gesture.start.x),height:Math.abs(point.y-gesture.start.y),label:$('cameraTargetLabel').value})}
    else{brush(gesture.data,width,height,gesture.last,point,Number($('cameraTargetBrushRadius').value)||24,gesture.type==='mask_erase');gesture.last=point;if(!gesture.renderQueued){gesture.renderQueued=true;requestAnimationFrame(()=>{gesture.renderQueued=false;if(state.cameraTargetGesture===gesture)renderTargetMask(gesture.data,width,height)})}}
  };
  overlay.onpointerup=event=>{
    const gesture=state.cameraTargetGesture,point=targetPoint(event);if(!gesture||!point)return;state.cameraTargetGesture=null;
    if(gesture.type==='rectangle'){const x=Math.min(point.x,gesture.start.x),y=Math.min(point.y,gesture.start.y),width=Math.abs(point.x-gesture.start.x),height=Math.abs(point.y-gesture.start.y);if(width>=2&&height>=2)commitCameraTarget({type:'rectangle',x,y,width,height});else renderCameraTarget()}
    else{const {width,height}=cameraTargetDimensions();if(gesture.data.some(Boolean))commitCameraTarget({type:'mask',x:0,y:0,width,height,counts:encodeMask(gesture.data,width,height)});else clearCameraTarget('Mask 已完全擦除，目標區域已清除。')}
    try{overlay.releasePointerCapture(event.pointerId)}catch{}
  };
  overlay.ondblclick=event=>{event.preventDefault();if($('cameraTargetTool').value==='polygon'&&state.cameraTargetDraft.length>=3)commitCameraTarget({type:'polygon',points:state.cameraTargetDraft.map(point=>[...point])})};
  }
  document.addEventListener('keydown',event=>{
    if(state.stage!=='acquire'||state.acquireSource!=='camera'||event.ctrlKey||event.altKey||event.metaKey)return;
    if(event.key==='Escape'&&document.querySelector('.camera-workspace').classList.contains('preview-expanded')){event.preventDefault();setCameraPreviewExpanded(false);return}
    const focus=event.target;if(focus instanceof HTMLElement&&(focus.matches('input,textarea,select')||focus.isContentEditable))return;
    const key=event.key.toLowerCase(),shortcut=key==='q'?'rectangle':key==='w'?'polygon':key==='e'?(event.shiftKey?'mask_erase':'mask'):null;
    if(shortcut){event.preventDefault();$('cameraTargetTool').value=shortcut;updateCameraTargetTool();return}
    if(key==='s'&&!event.shiftKey){
      if(event.repeat||$('takeSnapshot').disabled)return;
      event.preventDefault();$('takeSnapshot').click();return;
    }
    if(event.key==='Enter'&&$('cameraTargetTool').value==='polygon'&&state.cameraTargetDraft.length>=3){event.preventDefault();commitCameraTarget({type:'polygon',points:state.cameraTargetDraft.map(point=>[...point])})}
    else if(event.key==='Backspace'&&$('cameraTargetTool').value==='polygon'&&state.cameraTargetDraft.length){event.preventDefault();state.cameraTargetDraft.pop();renderCameraTarget()}
    else if(event.key==='Escape'){
      event.preventDefault();
      if(state.cameraTargetDraft.length||state.cameraTargetGesture){state.cameraTargetDraft=[];state.cameraTargetGesture=null;renderCameraTarget()}
      else{$('cameraTargetTool').value='none';updateCameraTargetTool()}
    }
  });
}

function cameraFlags(value) {return {running:!!(value?.running||value?.active),recording:!!value?.recording};}
let cameraModes=[],cameraModesLoading=false;
async function loadCameraModes() {
  if(cameraModesLoading)return;
  cameraModesLoading=true;updateCameraControls();
  $('cameraModesInfo').textContent='正在讀取此鏡頭支援的解析度與 FPS…';
  const index=Number($('cameraDevice').value);
  try {
    const result=await api(`/api/camera/capabilities?index=${index}`);
    cameraModes=result.modes||[];
    const previous=$('cameraResolution').value,resolutions=[...new Set(cameraModes.map(m=>`${m.width}x${m.height}`))];
    $('cameraResolution').replaceChildren(...resolutions.map(value=>new Option(value.replace('x',' × '),value)),new Option('自訂解析度','custom'));
    $('cameraResolution').value=resolutions.includes(previous)?previous:resolutions.includes('1280x720')?'1280x720':resolutions[0]||'custom';
    updateCameraMode();
    if(result.error)$('cameraModesInfo').textContent=result.error+'；自訂值需以實際輸出確認。';
  }catch(error){
    cameraModes=[];$('cameraResolution').replaceChildren(new Option('自訂解析度','custom'));updateCameraMode();
    $('cameraModesInfo').textContent=`無法讀取鏡頭模式：${error.message}。可重新偵測或自訂設定。`;
  }finally{cameraModesLoading=false;updateCameraControls();}
}
function selectedCameraModes() {
  return cameraModes.filter(m=>`${m.width}x${m.height}`===$('cameraResolution').value);
}
function updateCameraMode() {
  const custom=$('cameraResolution').value==='custom',modes=selectedCameraModes();
  $('cameraCustomSize').hidden=!custom;
  const rates=[...new Set(modes.flatMap(m=>m.fps_options))].sort((a,b)=>a-b);
  $('cameraRates').replaceChildren(...rates.map(value=>new Option(String(value),String(value))));
  if(modes.length){
    const fps=Number($('cameraFPS').value);
    if(!modes.some(m=>fps>=m.min_fps&&fps<=m.max_fps))$('cameraFPS').value=rates.reduce((a,b)=>Math.abs(a-fps)<=Math.abs(b-fps)?a:b);
    $('cameraFPS').min=Math.min(...modes.map(m=>m.min_fps));$('cameraFPS').max=Math.max(...modes.map(m=>m.max_fps));
    $('cameraModesInfo').textContent=modes.map(m=>`${m.pixel_format}：${m.min_fps===m.max_fps?m.max_fps:`${m.min_fps}–${m.max_fps}`} FPS`).join('；')+'。停止相機後可修改。';
  }else{$('cameraFPS').min=1;$('cameraFPS').max=240;$('cameraModesInfo').textContent='自訂設定未經模式清單驗證，啟動後請確認實際解析度與 FPS。';}
}
function cameraConfiguration() {
  const fps=Number($('cameraFPS').value),custom=$('cameraResolution').value==='custom';
  if(!$('cameraFPS').value.trim()||!Number.isFinite(fps)||fps<1||fps>240)throw Error('請輸入有效 FPS（1–240）。');
  let width,height,pixel_format='MJPG';
  if(custom){
    width=Number($('cameraWidth').value);height=Number($('cameraHeight').value);
    if(![width,height].every(v=>Number.isInteger(v)&&v>=32&&v<=8192))throw Error('影像寬高必須為 32–8192 的整數。');
  }else{
    const mode=selectedCameraModes().filter(m=>fps>=m.min_fps&&fps<=m.max_fps).sort((a,b)=>(a.pixel_format!=='MJPG')-(b.pixel_format!=='MJPG'))[0];
    if(!mode)throw Error('此解析度不支援所填 FPS，請依下方鏡頭模式範圍設定。');
    ({width,height,pixel_format}=mode);
  }
  return {index:Number($('cameraDevice').value),width,height,fps,pixel_format};
}
let cameraStatusTimer=null,cameraStatusRequest=null,cameraMutation=0;
function receiveCameraStatus(value) {
    state.cameraDetails=value;Object.assign(state,{camera:cameraFlags(value).running,recording:cameraFlags(value).recording});
    if(state.cameraTarget&&state.camera&&(state.cameraTarget.width!==Number(value.width)||state.cameraTarget.height!==Number(value.height)))clearCameraTarget('相機解析度已改變，請重新繪製目標區域。');
    if(state.autoCapture?.active&&!state.camera&&value?.state!=='starting')stopAutoCapture(value?.error||'相機連線已停止。');
    if(state.camera&&value.state!=='stopping')startPreview();else stopPreview();
    updateCameraControls();renderCameraTarget();
    if(value.processing_error)$('processingStatus').textContent=`${value.processing_error} · 預覽暫時顯示原圖。`;
    else if(value.processing_mode&&value.processing_mode!=='original')$('processingStatus').textContent=`${value.processing_state==='processing'?'正在處理影格':value.processing_state==='ready'?'預覽已就緒':value.processing_state||'處理中'}${value.processing_ms?' · '+Math.round(value.processing_ms)+' ms':''} · 拍攝保存原圖`;
    else $('processingStatus').textContent='拍攝與錄影保存原始影像。';
}
function cameraStatus() {
  if(cameraStatusRequest)return cameraStatusRequest;
  clearTimeout(cameraStatusTimer);
  const revision=cameraMutation;
  cameraStatusRequest=(async()=>{
    try{const value=await api('/api/camera/status');if(revision===cameraMutation)receiveCameraStatus(value);}
    catch(error){$('cameraStatus').textContent=error.message;}
    finally{
      cameraStatusRequest=null;
      // Startup has no frame yet. Keep polling independently of img.onload.
      if(state.stage==='acquire'||state.camera||['starting','stopping'].includes(state.cameraDetails?.state))
        cameraStatusTimer=setTimeout(cameraStatus,700);
    }
  })();
  return cameraStatusRequest;
}
async function cameraCommand(path,payload) {
  ++cameraMutation;
  try{const value=await api(path,'POST',payload);++cameraMutation;receiveCameraStatus(value);}
  finally{clearTimeout(cameraStatusTimer);cameraStatusTimer=setTimeout(cameraStatus,0);}
}
function stopPreview() {
  clearTimeout(state.previewTimer);state.previewRunning=false;state.previewReady=false;state.previewError='';
  for(const id of ['cameraFrame','cameraRawFrame']){const frame=$(id);frame.onload=null;frame.onerror=null;if(frame.hasAttribute('src'))frame.removeAttribute('src');}
}
function updatePreviewLayout(restart=true) {
  const compare=$('previewLayout').value==='compare';$('cameraPreview').classList.toggle('compare',compare);
  $('cameraPreview').classList.toggle('show-crosshair',$('showCenterCrosshair').checked);
  $('showRawFrame').disabled=compare;$('showRawFrame').closest('label').title=compare?'分割比較固定同時顯示原圖與處理輸出':'';
  $('outputPreviewLabel').textContent=compare?'處理輸出':$('showRawFrame').checked?'原圖':'處理輸出';
  updateCameraTargetEditingSurface();
  if(restart&&state.camera){stopPreview();startPreview();}else updateCameraControls();
}
function setCameraPreviewExpanded(expanded) {
  const workspace=document.querySelector('.camera-workspace'),page=$('acquire'),button=$('togglePreviewFullscreen'),active=!!expanded;
  if(!workspace||!button)return;
  workspace.classList.toggle('preview-expanded',active);page.classList.toggle('preview-expanded-layout',active);
  button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));button.setAttribute('aria-label',active?'還原即時預覽':'放大即時預覽');button.title=active?'還原即時預覽 · Esc':'放大即時預覽';
  button.querySelector('.expand-icon').hidden=active;button.querySelector('.restore-icon').hidden=!active;
  if(active)setSourceInspector(false);
  page.scrollTop=0;
  requestAnimationFrame(updateCameraTargetEditingSurface);
}
function updateCameraControls() {
  const details=state.cameraDetails||{},recording=details.last_recording;
  const starting=details.state==='starting',stopping=details.state==='stopping',engaged=state.camera||starting||stopping;
  $('startCamera').disabled=engaged||state.busy||cameraModesLoading;$('stopCamera').disabled=!engaged||stopping||state.busy;
  $('stopCamera').textContent=starting?'取消啟動':stopping?'停止中…':'停止相機';
  const auto=!!state.autoCapture?.active;
  $('takeSnapshot').disabled=!state.camera||stopping||state.busy||!state.project||auto;$('startRecording').disabled=!state.camera||stopping||state.recording||state.busy||!state.project||auto;
  $('startAutoCapture').disabled=!state.camera||stopping||state.recording||state.busy||!state.project||auto;
  $('stopAutoCapture').disabled=!auto;$('startAutoCapture').hidden=auto;$('stopAutoCapture').hidden=!auto;
  for(const id of ['autoCaptureInterval','autoCaptureLimit'])$(id).disabled=auto||state.busy;
  document.querySelector('.camera-capture-section').classList.toggle('auto-active',auto);
  $('stopRecording').disabled=!state.recording||state.busy;$('cameraDevice').disabled=engaged||state.busy||cameraModesLoading;
  $('findCameras').disabled=engaged||state.busy||cameraModesLoading;
  for(const id of ['cameraResolution','cameraFPS','cameraWidth','cameraHeight'])$(id).disabled=engaged||state.busy||cameraModesLoading;
  $('extractRecording').disabled=state.busy||!state.project||!recording?.path;
  $('applyProcessing').disabled=!state.camera||state.busy;$('calibrateBackground').disabled=!state.camera||state.busy;
  for(const id of ['cameraTargetTool','cameraTargetLabel','cameraTargetBrushRadius','cameraTargetUseProcessing','cameraTargetAttach'])$(id).disabled=!state.camera||state.busy;
  $('clearCameraTarget').disabled=!cameraTargetShape()||state.busy;
  updateCameraTargetEditingSurface();
  const compare=$('previewLayout').value==='compare';
  $('cameraFrame').hidden=!state.camera||!state.previewReady;$('cameraRawFrame').hidden=!state.camera||!state.previewReady||!compare;
  $('cameraOutputPane').hidden=$('cameraFrame').hidden;$('cameraRawPane').hidden=$('cameraRawFrame').hidden;$('cameraEmpty').hidden=state.camera&&!!state.previewReady;
  $('cameraEmpty').querySelector('h3').textContent=details.error?'相機啟動或連線失敗':starting?'相機啟動中…':stopping?'正在停止相機…':state.camera?'正在載入影像…':'相機尚未連接';
  $('cameraEmpty').querySelector('p').textContent=details.error||state.previewError||(starting?'正在等待鏡頭回傳第一張影像，可按「取消啟動」。':state.camera?'正在接收相機影格。':'在右側選擇裝置，啟動即時預覽');
  $('cameraStatus').textContent=details.error||state.previewError||(starting?'正在啟動相機，等待第一張影像…':stopping?'正在等待相機停止…':state.recording?'正在錄影 · 停止後可取樣畫格':auto?`自動擷取中 · 已加入 ${state.autoCapture.count} 張`:state.camera?'相機已啟動 · 即時預覽':'相機尚未啟動');
  $('startCamera').hidden=engaged;$('stopCamera').hidden=!engaged;
  $('startRecording').hidden=state.recording;$('stopRecording').hidden=!state.recording;
  $('cameraActivity').hidden=!engaged;$('cameraActivity').classList.toggle('recording',state.recording);
  $('cameraActivity').querySelector('span').textContent=starting?'相機啟動中 · 返回預覽':stopping?'相機停止中 · 返回預覽':state.recording?'錄影中 · 返回預覽':'相機運作中 · 返回預覽';
  $('recordedFrameTools').hidden=!recording?.path;
  if(recording?.path)$('recordingSummary').textContent=`${recording.name||'錄影已保存'} · ${Number(recording.duration_seconds||0).toFixed(1)} 秒 · ${recording.frames||0} 幀`;
  const actual=Number(details.measured_fps||0),negotiated=Number(details.fps||0);
  $('cameraReadout').textContent=starting?'正在啟動裝置':state.camera?`${details.width||'—'} × ${details.height||'—'} · 協商 ${Number(negotiated.toFixed(2))} FPS · 鏡頭實測 ${Number(actual.toFixed(1))} FPS${details.pixel_format&&details.pixel_format!=='unknown'?' · '+details.pixel_format:''}`:'等待連接裝置';
  $('cameraReadout').title=details.requested?`要求 ${details.requested.width} × ${details.requested.height}，${details.requested.fps} FPS；預覽更新頻率與鏡頭擷取頻率分開。`:'';
  const requested=details.requested;
  const changed=state.camera&&requested&&(details.width!==requested.width||details.height!==requested.height||Math.abs(negotiated-requested.fps)>.1||(details.pixel_format!=='unknown'&&details.pixel_format!==requested.pixel_format));
  $('cameraNegotiation').hidden=!changed;
  if(changed)$('cameraNegotiation').textContent=`鏡頭未完全採用要求設定；目前輸出 ${details.width} × ${details.height}、${Number(negotiated.toFixed(2))} FPS、${details.pixel_format}。`;
  updateAcquisitionControls();
}
function startPreview() {
  if(state.previewRunning)return;state.previewRunning=true;
  const frame=$('cameraFrame'),raw=$('cameraRawFrame');
  let pending=false,rawPending=false,requestedAt=0,rawRequestedAt=0;
  const next=()=>{
    if(!state.camera||!state.previewRunning)return;
    if(state.stage==='acquire'&&state.acquireSource==='camera'&&(!pending||Date.now()-requestedAt>5000)){
      pending=true;requestedAt=Date.now();
      const compare=$('previewLayout').value==='compare';
      frame.src=`/api/camera/frame?processed=${compare||!$('showRawFrame').checked?'1':'0'}&t=${Date.now()}`;
      if(compare&&(!rawPending||Date.now()-rawRequestedAt>5000)){rawPending=true;rawRequestedAt=Date.now();raw.src=`/api/camera/frame?processed=0&t=${Date.now()}`;}
    }
    state.previewTimer=setTimeout(next,150);
  };
  frame.onload=()=>{pending=false;state.previewReady=true;state.previewError='';updateCameraControls();};
  frame.onerror=()=>{pending=false;state.previewReady=false;state.previewError='暫時讀不到影像，正在重試。';updateCameraControls();};
  raw.onload=()=>{rawPending=false;};raw.onerror=()=>{rawPending=false;};
  next();
}

function stopAutoCapture(message='已停止自動擷取。') {
  const session=state.autoCapture;if(!session)return;
  session.active=false;clearTimeout(session.timer);state.autoCapture=null;
  $('autoCaptureStatus').textContent=`${message}${session.attempts?` 本次執行 ${session.attempts} 次，加入 ${session.count} 張。`:''}`;updateCameraControls();
}
async function runAutoCapture(session) {
  if(state.autoCapture!==session||!session.active)return;
  if(!state.camera||state.recording||state.project?.id!==session.projectId){stopAutoCapture('採集條件已變更，已停止自動擷取。');return;}
  try {
    const result=await api(`/api/projects/${session.projectId}/capture`,'POST',{batch_id:session.batchId,target_shape:cameraTargetPayload()});
    session.attempts++;
    if(result?.id)session.count++;
    state.project=await api(`/api/projects/${session.projectId}`);renderAssetList();renderAcquisitionAssets();
    if(state.autoCapture!==session||!session.active)return;
    $('autoCaptureStatus').textContent=`自動擷取中 · 已執行 ${session.attempts} 次 · 加入 ${session.count} 張${session.limit?` · 上限 ${session.limit} 次`:''}`;
    if(session.limit&&session.attempts>=session.limit){stopAutoCapture('已達本次擷取次數。');return;}
  }catch(error){stopAutoCapture(`自動擷取失敗：${error.message}`);toast(error.message,true);return;}
  if(state.autoCapture===session&&session.active)session.timer=setTimeout(()=>runAutoCapture(session),session.interval*1000);
}
async function startAutoCapture() {
  const interval=Number($('autoCaptureInterval').value),limit=Number($('autoCaptureLimit').value);
  if(!Number.isFinite(interval)||interval<.5||interval>3600)throw Error('自動擷取間隔必須介於 0.5～3600 秒。');
  if(!Number.isInteger(limit)||limit<0||limit>100000)throw Error('本次張數必須是 0～100,000 的整數。');
  if(!state.camera||state.recording||!state.project)throw Error('請先啟動相機，並停止錄影。');
  await flushAllEdits();
  const session={active:true,projectId:state.project.id,batchId:`auto_${Date.now().toString(36)}`,interval,limit,count:0,attempts:0,timer:null};state.autoCapture=session;
  $('autoCaptureStatus').textContent='正在擷取第一張影像…';updateCameraControls();await runAutoCapture(session);
}

function filteredReview() {
  const query=$('reviewSearch').value.toLowerCase(),filter=$('reviewFilter').value;
  return (state.project?.assets||[]).filter(a=>a.name.toLowerCase().includes(query)&&(filter==='all'||a.review_state===filter));
}
function reviewPageItems() {return filteredReview().slice(state.reviewPage*48,(state.reviewPage+1)*48);}
function updateReviewSelection() {
  $('reviewSelectedCount').textContent=`已選 ${state.reviewSelection.size} 張`;
  const page=reviewPageItems();$('reviewSelectAll').checked=!!page.length&&page.every(a=>state.reviewSelection.has(a.id));
  $('reviewSelectAll').indeterminate=page.some(a=>state.reviewSelection.has(a.id))&&!$('reviewSelectAll').checked;
  for(const id of ['reviewApprove','reviewReject','reviewPending','assignSelected'])$(id).disabled=!state.reviewSelection.size||state.busy;
}
function renderReview() {
  const counts=stats(state.project);const summary=$('reviewStats');summary.replaceChildren();
  for(const [key,name]of Object.entries({total:'全部影像',pending:'待審核',approved:'已核准',rejected:'已退回'})){const chip=element('div',undefined,'stat-chip');chip.append(element('span',name),element('b',number(counts[key])));summary.append(chip);}
  const pages=Math.max(1,Math.ceil(filteredReview().length/48));state.reviewPage=Math.max(0,Math.min(state.reviewPage,pages-1));
  $('reviewPageLabel').textContent=`第 ${state.reviewPage+1} / ${pages} 頁 · ${number(filteredReview().length)} 張`;
  $('reviewPrevious').disabled=state.reviewPage<=0;$('reviewNext').disabled=state.reviewPage>=pages-1;
  const grid=$('reviewGrid');grid.replaceChildren();const generation=++state.reviewGeneration;
  for(const asset of reviewPageItems()) {
    const card=element('article',undefined,'review-card'+(state.reviewSelection.has(asset.id)?' selected':''));
    const media=element('div',undefined,'review-card-media'),img=document.createElement('img');img.src=thumbnailURL(asset);img.alt=asset.name;img.loading='lazy';
    const check=document.createElement('input');check.type='checkbox';check.checked=state.reviewSelection.has(asset.id);check.setAttribute('aria-label',`選取 ${asset.name}`);
    const label=element('label');label.append(check);label.onclick=e=>e.stopPropagation();
    check.onchange=()=>{if(check.checked)state.reviewSelection.add(asset.id);else state.reviewSelection.delete(asset.id);card.classList.toggle('selected',check.checked);updateReviewSelection()};
    const overlay=document.createElementNS('http://www.w3.org/2000/svg','svg');overlay.setAttribute('viewBox',`0 0 ${asset.width} ${asset.height}`);overlay.setAttribute('preserveAspectRatio','xMidYMid meet');
    media.append(img,overlay,label,element('span',reviewNames[asset.review_state],'badge '+asset.review_state));
    media.onclick=()=>safe(async()=>{await selectAsset(asset.id);await switchStage('annotate')});
    const body=element('div',undefined,'review-card-body');body.append(element('b',asset.name),element('small',`${asset.width} × ${asset.height} · ${asset.shape_count||0} 個物件`));
    const footer=element('div',undefined,'review-card-footer');footer.append(element('small',asset.split?`${asset.split} · ${asset.batch_id||''}`:asset.batch_id||'尚未指定批次'),button('檢視標註 →','text-button',()=>media.click()));body.append(footer);card.append(media,body);grid.append(card);
    queueReviewPreview({asset,overlay,generation});
  }
  if(!grid.children.length)grid.append(element('p','沒有符合篩選條件的圖片。','muted'));
  updateReviewSelection();
}
const previewQueue=[];let previewWorkers=0;
function queueReviewPreview(task) {
  previewQueue.push(task);
  while(previewWorkers<3&&previewQueue.length)runReviewPreview();
}
async function runReviewPreview() {
  previewWorkers++;
  try {
    while(previewQueue.length){const task=previewQueue.shift();if(task.generation!==state.reviewGeneration)continue;
      try{const full=await api(projectPath(`/assets/${task.asset.id}`));if(task.generation===state.reviewGeneration)drawReviewOverlay(task.overlay,full)}catch(error){task.overlay.setAttribute('aria-label','標註預覽讀取失敗');}
    }
  }finally{previewWorkers--;}
}
function drawReviewOverlay(svg,asset) {
  const ratio=Math.max(asset.width/240,asset.height/150),ns='http://www.w3.org/2000/svg';
  for(const shape of asset.shapes||[]) {
    let node,t=kind(shape);const color=colorFor(shape.label);
    if(t==='mask') {
      const canvas=document.createElement('canvas');canvas.width=asset.width;canvas.height=asset.height;
      const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(asset.width,asset.height),data=decodeMask(shape.counts,asset.width,asset.height);
      for(let i=0;i<data.length;i++)if(data[i]){pixels.data[i*4]=69;pixels.data[i*4+1]=198;pixels.data[i*4+2]=177;pixels.data[i*4+3]=110;}ctx.putImageData(pixels,0,0);
      node=document.createElementNS(ns,'image');node.setAttribute('href',canvas.toDataURL());node.setAttribute('width',asset.width);node.setAttribute('height',asset.height);
    } else {
      node=document.createElementNS(ns,t==='rectangle'?'rect':t==='point'?'circle':t==='linestrip'?'polyline':'polygon');
      if(t==='rectangle'){for(const k of ['x','y','width','height'])node.setAttribute(k,shape[k])}
      else if(t==='point'){node.setAttribute('cx',shape.points[0][0]);node.setAttribute('cy',shape.points[0][1]);node.setAttribute('r',3*ratio)}
      else node.setAttribute('points',shape.points.map(p=>p.join(',')).join(' '));
      node.setAttribute('stroke',color);node.setAttribute('stroke-width',1.5*ratio);node.setAttribute('fill',t==='point'?color:'none');
    }
    svg.append(node);
  }
}
async function reviewSelection(reviewState) {
  await flushAllEdits();const ids=[...state.reviewSelection];if(!ids.length)throw Error('請先選取需要更新的圖片。');
  const revisions=Object.fromEntries(state.project.assets.filter(a=>state.reviewSelection.has(a.id)).map(a=>[a.id,a.revision]));
  state.project=await api(projectPath('/review'),'POST',{asset_ids:ids,state:reviewState,revisions});
  const selectedMeta=state.project.assets.find(a=>a.id===state.asset?.id);if(selectedMeta)Object.assign(state.asset,{review_state:selectedMeta.review_state,revision:selectedMeta.revision});
  state.reviewSelection.clear();renderReview();updateAssetHeader();toast(`${ids.length} 張圖片已設為${reviewNames[reviewState]}。`);
}
async function assignSelected() {
  const ids=[...state.reviewSelection];if(!ids.length)return;
  const body=element('div');body.append(element('p',`將更新已選取的 ${ids.length} 張圖片。相同拍攝批次應放在同一訓練分割。`));
  const batch=document.createElement('input');batch.placeholder='例如：camera-A-2026-09-09';batch.maxLength=120;
  const split=document.createElement('select');for(const [value,text]of [['','未指定'],['train','train · 訓練'],['val','val · 驗證'],['test','test · 測試']])split.append(new Option(text,value));
  body.append(element('label','來源批次（留空保留原批次）'),batch,element('label','資料分割'),split);
  const result=await formDialog({title:'指定批次與資料分割',body,confirm:'儲存設定',onSubmit:()=>api(projectPath('/assign'),'POST',{asset_ids:ids,split:split.value,...(batch.value.trim()?{batch_id:batch.value.trim()}:{})})});
  if(result){state.project=result;renderReview();toast('批次與分割設定已更新。')}
}

function renderExport() {
  $('formatDescription').textContent=formatDescriptions[$('exportFormat').value];
  renderBatchTable();const list=$('exportHistory');list.replaceChildren();
  for(const record of [...(state.project?.exports||[])].reverse()) {
    const row=element('article',undefined,'export-record'),body=element('div');
    body.append(element('b',`${record.version||record.name||'匯出版本'} · ${formatNames[record.format]||record.format||''}`),element('p',record.path||record.output_path||''),element('small',date(record.created_at||record.timestamp)));
    const open=button('開啟資料夾 ↗','secondary',()=>safe(()=>api('/api/open-folder','POST',{project_id:state.project.id,...(record.id?{export_id:record.id}:{})})));
    row.append(body,open);list.append(row);
  }
  if(!list.children.length)list.append(element('p','尚無匯出版本。完成審核後，執行驗證並建立第一個版本。','muted'));
}
const BATCH_PAGE_SIZE=3,VALIDATION_PAGE_SIZE=3;
function openReleaseDrawer(title,details){
  const shell=$('releaseDrawer'),body=$('releaseDrawerBody');state.releaseDrawerFocus=document.activeElement;
  $('releaseDrawerTitle').textContent=title;body.replaceChildren();
  for(const [label,value]of details){const row=element('div',undefined,'release-detail-row');row.append(element('span',label),element('div',String(value)));body.append(row)}
  shell.hidden=false;requestAnimationFrame(()=>shell.classList.add('open'));$('releaseDrawerClose').focus();
}
function closeReleaseDrawer(){
  const shell=$('releaseDrawer');if(shell.hidden)return;shell.classList.remove('open');shell.hidden=true;
  if(state.releaseDrawerFocus instanceof HTMLElement)state.releaseDrawerFocus.focus();state.releaseDrawerFocus=null;
}
function renderBatchTable() {
  const table=$('batchTable'),summary=$('splitSummary'),assets=state.project?.assets||[];table.replaceChildren();summary?.replaceChildren();
  const meta={train:['train · 訓練集','模型學習用'],val:['val · 驗證集','調整與比較模型用'],test:['test · 測試集','最終成效評估用'],'':['未指定','匯出前需要指定']};
  const splitMaps=new Map();
  for(const asset of assets){const split=asset.split||'',name=asset.batch_id||'';if(!splitMaps.has(split))splitMaps.set(split,new Map());const batches=splitMaps.get(split);if(!batches.has(name))batches.set(name,[]);batches.get(name).push(asset)}
  const bySplit=new Map([...splitMaps].map(([split,batches])=>[split,[...batches].map(([name,batchAssets])=>({name,assets:batchAssets}))]));
  for(const key of ['train','val','test']){
    const batches=bySplit.get(key)||[],splitAssets=batches.flatMap(batch=>batch.assets),approved=splitAssets.filter(asset=>asset.review_state==='approved').length,item=element('div',undefined,'split-summary-item');
    item.append(element('span',meta[key][0],'split-summary-label'),element('b',`${number(approved)} 張`),element('small',`${number(batches.length)} 個來源批次 · ${meta[key][1]}`));summary?.append(item);
  }
  const balance=$('classBalanceTable');balance?.replaceChildren();
  if(balance){
    const matrix=document.createElement('table'),head=document.createElement('thead'),headRow=document.createElement('tr');
    for(const name of ['類別','train','val','test','合計'])headRow.append(element('th',name));head.append(headRow);matrix.append(head);
    const body=document.createElement('tbody');
    for(const className of state.project?.classes||[]){
      const row=document.createElement('tr'),counts={train:0,val:0,test:0};
      for(const asset of assets)if(asset.review_state==='approved'&&counts[asset.split]!==undefined)counts[asset.split]+=Number(asset.class_counts?.[className]||0);
      row.append(element('th',className));for(const split of ['train','val','test'])row.append(element('td',number(counts[split])));row.append(element('td',number(counts.train+counts.val+counts.test)));body.append(row);
    }
    matrix.append(body);balance.append(matrix);
  }
  const keys=['train','val','test'];if((bySplit.get('')||[]).length)keys.push('');
  if(!keys.includes(state.splitTab))state.splitTab=keys[0];
  const tabs=element('div',undefined,'split-tabs');tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','選擇資料集合');
  for(const key of keys){
    const batches=bySplit.get(key)||[],count=batches.reduce((total,batch)=>total+batch.assets.length,0),tab=button(`${meta[key][0]} · ${number(count)}`,'split-tab',()=>{state.splitTab=key;state.splitPage=0;renderBatchTable()});
    tab.setAttribute('role','tab');tab.setAttribute('aria-selected',String(state.splitTab===key));tabs.append(tab);
  }
  const batches=bySplit.get(state.splitTab)||[],pages=Math.max(1,Math.ceil(batches.length/BATCH_PAGE_SIZE));state.splitPage=Math.min(state.splitPage,pages-1);
  const viewport=element('div',undefined,'batch-viewport');viewport.setAttribute('role','tabpanel');
  for(const {name,assets:batchAssets}of batches.slice(state.splitPage*BATCH_PAGE_SIZE,(state.splitPage+1)*BATCH_PAGE_SIZE)){
    const approved=batchAssets.filter(asset=>asset.review_state==='approved').length,row=element('div',undefined,'batch-row-fixed');
    const info=button(name||'尚未指定批次','batch-detail-link',()=>openReleaseDrawer('來源批次詳情',[
      ['來源批次',name||'尚未指定批次'],['目前集合',meta[state.splitTab][0]],['此集合影像',`${number(batchAssets.length)} 張`],['已核准',`${number(approved)} 張`]
    ]));info.title='查看來源批次詳情';info.append(element('small',`${number(batchAssets.length)} 張 · ${number(approved)} 已核准`));
    const select=document.createElement('select');for(const [v,t]of [['','未指定分割'],['train','train · 訓練集'],['val','val · 驗證集'],['test','test · 測試集']])select.append(new Option(t,v));select.value=state.splitTab;
    select.setAttribute('aria-label',`${name||'未指定批次'}的資料分割`);
    const apply=button('儲存','secondary',()=>safe(async()=>{
      if(state.busy)return;apply.disabled=true;select.disabled=true;
      try{await flushAllEdits();state.project=await api(projectPath('/assign'),'POST',{asset_ids:batchAssets.map(asset=>asset.id),split:select.value});state.splitPage=0;renderBatchTable();toast('批次分割已儲存，請重新執行驗證。')}
      finally{apply.disabled=false;select.disabled=false;}
    }));row.append(info,select,apply);viewport.append(row);
  }
  if(!batches.length)viewport.append(element('p',assets.length?'此集合目前沒有來源批次。':'加入圖片後，可在此設定來源批次與資料分割。','batch-empty'));
  const pager=element('div',undefined,'fixed-pager'),previous=button('上一頁','secondary',()=>{state.splitPage--;renderBatchTable()}),next=button('下一頁','secondary',()=>{state.splitPage++;renderBatchTable()});
  previous.disabled=state.splitPage===0;next.disabled=state.splitPage>=pages-1;pager.append(previous,element('span',`第 ${state.splitPage+1} / ${pages} 頁 · ${number(batches.length)} 個來源批次`),next);
  table.append(tabs,viewport,pager);
}
async function autoSplitProject(){
  const values=$('splitRatio').value.split('/').map(Number),ratios={train:values[0],val:values[1],test:values[2]};
  const result=await api(projectPath('/auto-split'),'POST',{ratios});state.project=result.project;renderBatchTable();
  const report=result.report||{},message=$('splitBalanceReport');message.hidden=false;message.replaceChildren(element('strong',`已依 cls 數量完成 ${ratios.train} / ${ratios.val} / ${ratios.test} 分割`),element('div',`更新 ${number(report.changed||0)} 張影像；分配結果：train ${number(report.image_counts?.train||0)}、val ${number(report.image_counts?.val||0)}、test ${number(report.image_counts?.test||0)}。`));
  for(const warning of report.warnings||[])message.append(element('div',warning,'split-warning'));
  resetValidation('資料分割已更新，請重新執行驗證。');toast('已依類別數量完成自動分割。');
}
function resetValidation(message){
  state.validationResult=null;state.validationTab=null;state.validationPage=0;
  const root=$('validationReport');root.className='report-empty';root.replaceChildren(element('p',message,'muted'));
}
function renderValidation(result) {
  const base=result?.report||result||{},report=base.validation?{...base.validation,losses:base.losses||base.validation.losses}:base;
  state.validationResult={result,report};state.validationPage=0;
  state.validationTab=(report.errors||[]).length?'errors':(report.warnings||[]).length?'warnings':(report.losses||[]).length?'losses':'errors';
  renderValidationView();
}
function renderValidationView(){
  const root=$('validationReport'),stored=state.validationResult;if(!stored)return;
  const {result,report}=stored,valid=report.valid??!(report.errors?.length);root.className='validation-report-shell';root.replaceChildren();
  const header=element('div',undefined,'report-header compact'+(valid?'':' invalid')),text=element('div');
  text.append(element('h3',result?.path?'匯出版本已建立':valid?'檢查通過':'仍有問題需要處理'),element('p',result?.path?'圖片與封裝檢查均已完成。':valid?'請確認轉換影響後建立匯出版本。':'請依問題清單修正後重新驗證。'));header.append(element('span',valid?'✓':'!','report-symbol'),text);
  if(result?.path)header.append(button('查看匯出位置','secondary',()=>openReleaseDrawer('匯出版本位置',[['輸出路徑',result.path],...(result.zip_path?[['壓縮檔',result.zip_path]]:[])])));
  root.append(header);
  const counts=report.stats||{},statRow=element('div',undefined,'report-stats fixed');
  const keys={total:'全部影像',assets:'影像',approved:'已核准',pending:'待審核',rejected:'已退回',images:'輸出影像',annotations:'標註',shapes:'標註物件',classes:'類別',batches:'來源批次',exported:'輸出影像',eligible:'可匯出'};
  for(const [key,value]of Object.entries(counts))if(typeof value==='number')statRow.append(element('span',`${keys[key]||key} ${number(value)}`));root.append(statRow);
  const categories=[['errors','必須處理'],['warnings','檢查提醒'],['losses','格式轉換影響']],tabs=element('div',undefined,'validation-tabs');tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','驗證結果分類');
  for(const [key,title]of categories){const count=(report[key]||[]).length,tab=button(`${title} · ${number(count)}`,'validation-tab',()=>{state.validationTab=key;state.validationPage=0;renderValidationView()});tab.setAttribute('role','tab');tab.setAttribute('aria-selected',String(state.validationTab===key));tabs.append(tab)}root.append(tabs);
  const entries=report[state.validationTab]||[],pages=Math.max(1,Math.ceil(entries.length/VALIDATION_PAGE_SIZE));state.validationPage=Math.min(state.validationPage,pages-1);
  const viewport=element('div',undefined,'validation-viewport '+state.validationTab);viewport.setAttribute('role','tabpanel');
  for(const [offset,entry]of entries.slice(state.validationPage*VALIDATION_PAGE_SIZE,(state.validationPage+1)*VALIDATION_PAGE_SIZE).entries()){
    const item=element('div',undefined,'validation-item'),message=readable(entry),index=state.validationPage*VALIDATION_PAGE_SIZE+offset+1,body=element('div');
    body.append(element('b',message),element('small',`${categories.find(category=>category[0]===state.validationTab)[1]} · 第 ${number(index)} 項`));
    item.append(body,button('詳情','secondary',()=>openReleaseDrawer('驗證項目詳情',[
      ['分類',categories.find(category=>category[0]===state.validationTab)[1]],['項目',`${index} / ${entries.length}`],['說明',message]
    ])));viewport.append(item);
  }
  if(!entries.length)viewport.append(element('p',state.validationTab==='errors'?'此分類沒有必須處理的問題。':'此分類目前沒有項目。','validation-empty'));root.append(viewport);
  const pager=element('div',undefined,'fixed-pager validation-pager'),previous=button('上一頁','secondary',()=>{state.validationPage--;renderValidationView()}),next=button('下一頁','secondary',()=>{state.validationPage++;renderValidationView()});
  previous.disabled=state.validationPage===0;next.disabled=state.validationPage>=pages-1;pager.append(previous,element('span',`第 ${state.validationPage+1} / ${pages} 頁 · ${number(entries.length)} 項`),next);root.append(pager);
}
async function validateProject() {
  await flushAllEdits();const job=await api(projectPath('/validate'),'POST',{format:$('exportFormat').value,tolerance:0});
  const result=await pollJob(job);renderValidation(result);toast('驗證已完成。');
}
async function exportProject() {
  await flushAllEdits();const format=$('exportFormat').value;
  const validation=await pollJob(await api(projectPath('/validate'),'POST',{format,tolerance:0}));renderValidation(validation);
  const report=validation?.report||validation;
  if(report?.valid===false||report?.errors?.length)throw Error('驗證尚未通過，請先修正報告中的問題。');
  if(report?.losses?.length&&!$('acknowledgeLoss').checked)throw Error('請閱讀轉換損失報告，勾選接受後再建立匯出版本。');
  const version=$('exportVersion').value.trim();if(!version)throw Error('請輸入版本名稱。');
  const result=await pollJob(await api(projectPath('/export'),'POST',{format,version,tolerance:0,acknowledge_loss:$('acknowledgeLoss').checked,...($('outputPath').value.trim()?{output_dir:$('outputPath').value.trim()}:{})}));
  renderValidation(result);await refreshProject();renderExport();toast('匯出版本已建立，可從下方紀錄開啟資料夾。');
}
async function runAI() {
  document.querySelector('.ai-section').open=true;
  if(!state.asset)throw Error('請先選取一張圖片。');
  if(!editor.label())throw Error('請先新增並選擇 AI 候選要使用的物件類別。');
  await saver.flush();if(editor.draft.length||editor.gesture)throw Error('請先完成目前的繪圖操作。');
  if(editor.candidate)throw Error('請先接受或捨棄現有候選遮罩。');
  const assetId=state.asset.id,revision=state.asset.revision;
  $('aiMessage').textContent='正在本機執行分割…';
  try {
    const result=await pollJob(await api(projectPath('/ai'),'POST',{asset_id:assetId,revision,engine:$('aiEngine').value,label:editor.label(),...editor.aiInput()}));
    if(state.asset?.id!==assetId||state.asset.revision!==revision)throw Error('圖片修訂已變更，未套用 AI 候選。');
    const shape=result?.shape||result?.candidate;if(!shape)throw Error('AI 未回傳可用的候選形狀。');
    editor.setCandidate(shape);$('aiMessage').textContent='候選已產生。黃色區域為預覽，接受後才會加入標註。';
    if(result.diagnostics){const extra=Array.isArray(result.diagnostics)?result.diagnostics.map(readable).join('；'):typeof result.diagnostics==='string'?result.diagnostics:result.diagnostics.message;$('aiMessage').textContent+=extra?' '+extra:'';}
  }catch(error){$('aiMessage').textContent=error.message;throw error;}
}

const splitManager=new SplitManager({api,onApplied:async(result,pid,createVersion)=>{
  if(state.project?.id!==pid)return;
  state.project=result.project;renderBatchTable();renderReview();resetValidation('資料分割已更新，請重新驗證。');
  if(createVersion)await createDatasetVersion();else await loadTraining();
  if(state.stage==='split')await renderSplitPage();
  toast(createVersion?'已建立新的固定資料版本。':'已套用專案分割；建立新資料版本後，後續訓練才會使用新分配。');
}});
async function openSplitManager(){await flushAllEdits();await splitManager.open(state.project.id)}
async function renderSplitPage(){
  if(!state.project||!state.training)return;
  const projectId=state.project.id,info=await api(projectPath('/split-info'),'POST',{});
  if(state.project?.id!==projectId||state.stage!=='split')return;
  const readiness=state.training.readiness||{},stats=readiness.stats||{},splits=stats.splits||{};
  const root=$('splitFlowStats');root.replaceChildren();
  for(const [label,value,note] of [
    ['已核准圖片',number(stats.approved||0),`排除 ${number(stats.excluded||0)} 張未核准圖片`],
    ['獨立來源群組',number(info.groups?.length||0),'智慧分割不拆開同來源群組'],
    ['分割方案',state.project.split_plan?.current?'已套用':'待確認',state.project.split_plan?.current?`種子 ${number(state.project.split_plan.seed)}`:'請預覽並套用智慧分割']]){
    const card=element('div');card.append(element('span',label),element('b',value),element('small',note));root.append(card);
  }
  const distribution=$('splitFlowDistribution');distribution.replaceChildren();
  const approved=Math.max(1,Number(stats.approved)||0);
  for(const [key,label] of [['train','Train'],['val','Validation'],['test','Test']]){
    const count=Number(splits[key])||0,row=element('div',undefined,'split-flow-row'),copy=element('div');
    copy.append(element('b',label),element('span',`${number(count)} 張 · ${(count/approved*100).toFixed(1)}%`));
    const bar=element('div',undefined,'split-flow-bar'),fill=element('i');fill.style.width=`${Math.min(100,count/approved*100)}%`;bar.append(fill);row.append(copy,bar);distribution.append(row);
  }
  const statusRoot=$('splitFlowStatus');statusRoot.replaceChildren();
  if(readiness.ready)statusRoot.append(element('div','資料分割符合建立固定資料版本的條件。','readiness-item'));
  for(const item of readiness.blockers||[]){const row=element('div',undefined,'readiness-item error');row.append(element('b',item.message),element('span',item.action));statusRoot.append(row)}
  for(const item of readiness.warnings||[]){const row=element('div',undefined,'readiness-item warning');row.append(element('b',item.message),element('span',item.action));statusRoot.append(row)}
  $('openSplitFlowManager').disabled=!(stats.approved>0);
  $('continueToTraining').disabled=!readiness.ready;
  $('splitFlowNextHint').textContent=readiness.ready?'分割已具備訓練條件；下一步建立固定資料版本。':'請先處理上方阻擋項目。';
}
const activeRunStates=new Set(['queued','preparing','running','stopping']);
function invalidateYoloCompatibility(){state.yoloCompatibility=null;state.yoloCompatibilitySignature='';if(state.stage==='train'&&state.training)renderTraining();else renderYoloCompatibility()}
const trainingParameters=new TrainingParameters({preferredDevice:()=>settingValue('device'),onChange:invalidateYoloCompatibility});
const trainingMonitor=new TrainingMonitor({loadReport:(projectId,runId)=>api(`/api/projects/${projectId}/training-runs/${runId}/metrics`),onSelect:id=>{state.selectedRun=id},onModel:id=>{state.selectedModel=id;safe(()=>switchStage('models'))}});
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
  state.trainingTimer=setTimeout(()=>{if(trainingProjectIsCurrent(projectId))loadTraining({quiet:true})},900);
}
function updateBackgroundTraining(error=''){
  const run=activeTrainingRun(),button=$('backgroundTraining');
  button.hidden=!run&&!error;
  if(error){$('backgroundTrainingText').textContent=`訓練狀態暫時無法更新：${error}`;$('backgroundTrainingProgress').removeAttribute('value');return}
  if(!run)return;
  const epoch=run.epoch?` · ${number(run.epoch)} / ${number(run.config?.epochs)}`:'';
  $('backgroundTrainingText').textContent=`${run.run_id} ${trainingStatusName(run.status)}${epoch}`;
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
  const yolo=$('trainingEngine')?.value?.startsWith('yolo26');panel.hidden=!yolo;if(!yolo)return;
  const report=state.yoloCompatibility,badge=$('yoloCompatibilityBadge'),root=$('yoloCompatibilityReport');root.replaceChildren();
  badge.className='badge';
  if(!report){badge.textContent='尚未檢查';return}
  const summary=report.summary||{};badge.textContent=report.compatible?'檢查通過':'需要處理';badge.classList.add(report.compatible?'approved':'error');
  const message=report.compatible
    ?`已掃描 ${number(summary.assets_scanned)} 張、${number(summary.shapes_scanned)} 個標註；${summary.pixels_repaired?`Run 副本將修補 ${number(summary.affected_assets)} 張圖片中的 ${number(summary.holes_repaired)} 個孔洞，共 ${number(summary.pixels_repaired)} px。`:'不需要相容修補。'} 原始標註不會變更。`
    :`已掃描 ${number(summary.assets_scanned)} 張；${number(summary.blocked_assets)} 張超過安全門檻，尚不能開始 YOLO Seg 訓練。`;
  root.append(element('div',message,`yolo-compatibility-summary${report.compatible?'':' error'}`));
  const issues=[...(report.blockers||[]),...(report.repairs||[])];if(!issues.length)return;
  const list=element('div',undefined,'yolo-issue-list');
  for(const issue of issues){const row=element('div',undefined,'yolo-issue'),copy=element('div');copy.append(element('b',`${issue.name} · ${issue.label||'未命名標註'}`),element('p',issue.message));const locate=button('放大位置','text-button',()=>safe(()=>showYoloLocation(issue)));row.append(copy,locate);const coords=(issue.holes||[]).map(hole=>{const [x,y,w,h]=hole.bbox;return `X=${x}${w>1?`～${x+w-1}`:''}、Y=${y}${h>1?`～${y+h-1}`:''} · ${hole.pixels} px`}).join('；');if(coords)row.append(element('p',coords));list.append(row)}
  root.append(list);
}
async function showYoloLocation(issue){
  const body=element('div'),lead=element('p',`${issue.name}／${issue.label||'未命名標註'}。紅框與十字標出孔洞；畫面已放大，因此 1 px 也清楚可見。`),canvas=document.createElement('canvas');canvas.width=760;canvas.height=420;canvas.className='yolo-location-canvas';body.append(lead,canvas);
  const coords=element('div',undefined,'yolo-coordinate-list');for(const [index,hole] of (issue.holes||[]).entries()){const [x,y,w,h]=hole.bbox;coords.append(element('div',`孔洞 ${index+1}：X=${x}${w>1?`～${x+w-1}`:''}，Y=${y}${h>1?`～${y+h-1}`:''}，${hole.pixels} px`))}body.append(coords);
  const jump=button('前往內建編輯器中的圖片','secondary',()=>safe(async()=>{$('cancelDialog').click();await selectAsset(issue.asset_id);await switchStage('annotate')}));body.append(jump);
  const showing=formDialog({title:'YOLO Seg 孔洞位置',body,eyebrow:'MASK DIAGNOSTIC'}),image=new Image();
  image.onload=async()=>{const holes=issue.holes||[],precise=holes.length>0,boxes=precise?holes.map(item=>item.bbox):[[0,0,image.naturalWidth,image.naturalHeight]],minX=Math.min(...boxes.map(b=>b[0])),minY=Math.min(...boxes.map(b=>b[1])),maxX=Math.max(...boxes.map(b=>b[0]+b[2])),maxY=Math.max(...boxes.map(b=>b[1]+b[3]));let sw=precise?Math.max(96,maxX-minX+64):image.naturalWidth,sh=precise?Math.max(64,maxY-minY+64):image.naturalHeight,sx=precise?(minX+maxX-sw)/2:0,sy=precise?(minY+maxY-sh)/2:0;sx=Math.max(0,Math.min(image.naturalWidth-sw,sx));sy=Math.max(0,Math.min(image.naturalHeight-sh,sy));sw=Math.min(sw,image.naturalWidth);sh=Math.min(sh,image.naturalHeight);const context=canvas.getContext('2d');context.imageSmoothingEnabled=false;context.drawImage(image,sx,sy,sw,sh,0,0,canvas.width,canvas.height);try{const asset=await api(`/api/projects/${state.project.id}/assets/${issue.asset_id}`),shape=asset.shapes?.find(item=>item.id===issue.shape_id);if(shape?.type==='mask'){const mask=decodeMask(shape.counts,asset.width,asset.height),scaleX=canvas.width/sw,scaleY=canvas.height/sh;context.fillStyle='#37dbc066';for(let y=Math.max(0,Math.floor(sy));y<Math.min(asset.height,Math.ceil(sy+sh));y++)for(let x=Math.max(0,Math.floor(sx));x<Math.min(asset.width,Math.ceil(sx+sw));x++)if(mask[y*asset.width+x])context.fillRect((x-sx)*scaleX,(y-sy)*scaleY,Math.max(1,scaleX),Math.max(1,scaleY))}}catch{}if(precise)for(const [index,[x,y,w,h]] of boxes.entries()){const dx=(x-sx)/sw*canvas.width,dy=(y-sy)/sh*canvas.height,dw=Math.max(10,w/sw*canvas.width),dh=Math.max(10,h/sh*canvas.height),cx=dx+(w/sw*canvas.width)/2,cy=dy+(h/sh*canvas.height)/2;context.fillStyle='#ff2f42aa';context.strokeStyle='#fff';context.lineWidth=2;context.fillRect(cx-dw/2,cy-dh/2,dw,dh);context.strokeRect(cx-dw/2,cy-dh/2,dw,dh);context.beginPath();context.moveTo(cx-18,cy);context.lineTo(cx+18,cy);context.moveTo(cx,cy-18);context.lineTo(cx,cy+18);context.stroke();context.fillStyle='#fff';context.font='bold 16px sans-serif';context.fillText(String(index+1),cx+10,cy-10)}canvas.dataset.ready='true'};
  image.onerror=()=>{const context=canvas.getContext('2d');context.fillStyle='#d9e8ed';context.font='16px sans-serif';context.fillText('無法載入圖片預覽；請依下方座標前往編輯器。',24,50);canvas.dataset.ready='true'};image.src=`/api/projects/${state.project.id}/assets/${issue.asset_id}/image`;await showing;
}
async function checkYoloCompatibility(){
  const dataset=selectedDataset();if(!dataset)throw Error('請先建立或選擇訓練資料版本。');
  const config=trainingParameters.collect(),signature=yoloCompatibilitySignature(dataset,config);$('checkYoloCompatibility').disabled=true;$('yoloCompatibilityBadge').textContent='掃描中…';
  try{const report=await api(projectPath('/training-compatibility'),'POST',{dataset_version_id:dataset.id,config:{engine:$('trainingEngine').value,...config}});state.yoloCompatibility=report;state.yoloCompatibilitySignature=signature;renderYoloCompatibility();return report}
  catch(error){$('yoloCompatibilityBadge').textContent='檢查失敗';$('yoloCompatibilityBadge').className='badge error';throw error}
  finally{$('checkYoloCompatibility').disabled=false}
}
function renderTraining(){
  if(!state.training)return;
  const {readiness,datasets=[],runs=[],capabilities}=state.training,active=activeTrainingRun();
  renderReadiness(readiness);
  const datasetSelect=$('trainingDataset'),remember=datasetSelect.value;datasetSelect.replaceChildren();
  if(!datasets.length)datasetSelect.append(new Option('尚無資料版本',''));
  for(const item of datasets)datasetSelect.append(new Option(`${item.id} · ${number(item.asset_count)} 張 · ${date(item.created_at)}`,item.id));
  datasetSelect.value=datasets.some(item=>item.id===remember)?remember:datasets[0]?.id||'';
  const dataset=selectedDataset();$('trainingDatasetCurrent').textContent=dataset?.id||'尚未建立';
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
  renderYoloCompatibility();
  $('trainingTaskLabel').textContent=engine?`${engine.task_name||capabilities.tasks?.[engine.task]||engine.task} · ${engine.annotation||'依模型需求'}`:'尚未選擇模型';
  const notice=$('trainingModelNotice');notice.hidden=!!engine?.train;if(!notice.hidden){$('trainingModelNoticeTitle').textContent=engine?.integration==='ready'?'此模型尚未準備完成':'此模型已納入開發待辦';$('trainingModelNoticeText').textContent=engine?.unavailable_reason||'完成必要元件與 Workbench adapter 後即可使用。'}
  const summary=$('trainingSummary');summary.replaceChildren();
  const deviceName=engine?.component==='builtin'?'CPU':({auto:'自動選擇',cuda:'NVIDIA CUDA',cpu:'CPU'}[$('trainingDevice').value]||'自動選擇');
  for(const [label,value]of [['資料版本',dataset?.id||'—'],['圖片',dataset?`${number(dataset.asset_count)} 張`:'—'],['任務',engine?.task_name||capabilities.tasks?.[engine?.task]||'—'],['引擎',engine?.name||'—'],['裝置',`${deviceName} · 獨立程序`]]){summary.append(element('dt',label),element('dd',value))}
  $('startTraining').disabled=!dataset||!engine?.train||!engine?.parameters?.length||!!active;
  if(engine?.key?.startsWith('yolo26')&&state.yoloCompatibility&&!state.yoloCompatibility.compatible)$('startTraining').disabled=true;
  $('stopTraining').hidden=!active;$('startTraining').hidden=!!active;
  $('trainingActionHint').textContent=active?`${active.run_id} ${trainingStatusName(active.status)}；切換頁面後仍在背景執行。`:!engine?.train?(engine?.unavailable_reason||'請先到設定中心準備模型。'):dataset?'開始時會固定目前顯示的資料、引擎與參數。':'先建立或選擇固定資料版本。';
  trainingMonitor.render();
}
async function createDatasetVersion(){await flushAllEdits();const created=await api(projectPath('/dataset-versions'),'POST',{});await loadTraining();$('trainingDataset').value=created.id;renderTraining();toast(`已建立固定訓練資料 ${created.id}。`)}
async function startTrainingRun(){
  const dataset=selectedDataset();if(!dataset)throw Error('請先建立或選擇訓練資料版本。');
  const config=trainingParameters.collect();
  if($('trainingEngine').value.startsWith('yolo26')){const signature=yoloCompatibilitySignature(dataset,config),report=state.yoloCompatibilitySignature===signature?state.yoloCompatibility:await checkYoloCompatibility();if(!report?.compatible)throw Error('YOLO Seg 相容檢查未通過；請查看標出的圖片與孔洞位置。')}
  const run=await api(projectPath('/training-runs'),'POST',{dataset_version_id:dataset.id,config:{engine:$('trainingEngine').value,...config}});
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
  const test=model.test||model.validation||{},values=test.accuracy!==undefined?[['Test Accuracy',Number(test.accuracy).toFixed(3)],['Test Macro F1',Number(test.macro_f1).toFixed(3)],['Test Macro Recall',Number(test.macro_recall).toFixed(3)]]:test.mask_map50_95!==undefined?[['Test Mask mAP50–95',Number(test.mask_map50_95).toFixed(3)],['Test Mask mAP50',Number(test.mask_map50).toFixed(3)]]:test.box_map50_95!==undefined?[['Test Box mAP50–95',Number(test.box_map50_95).toFixed(3)],['Test Box mAP50',Number(test.box_map50).toFixed(3)]]:test.box_mean_iou!==undefined?[['Test Box IoU',Number(test.box_mean_iou).toFixed(3)],['Recall@0.5',Number(test.recall_50).toFixed(3)]]:test.mean_dice!==undefined?[['Test mIoU',Number(test.mean_iou).toFixed(3)],['Test Dice',Number(test.mean_dice).toFixed(3)]]:[['Test Mean IoU',Number(test.mean_iou||0).toFixed(3)]];values.push(['測試圖片',number(test.images)],['類別數',number(model.classes?.length)]);const score=element('div',undefined,'run-metrics');for(const [label,value]of values){const item=element('div',undefined,'run-metric');item.append(element('span',label),element('b',value));score.append(item)}root.append(score);
  const abilities=['訓練','評估',...(definition?.predict?['預標註']:[]),'匯出'].join('／'),facts=element('dl',undefined,'model-facts');for(const [label,value]of [['來源 Run',model.run_id],['訓練資料',model.dataset_version_id],['類別',(model.classes||[]).join('、')],['模型能力',abilities],['建立時間',date(model.created_at)]])facts.append(element('dt',label),element('dd',value));root.append(facts,element('p',model.task==='image_classification'?'分類結果保留為圖片層級評估，不會建立覆蓋整張圖片的 Bounding Box。':'模型與資料版本、類別映射和評估結果一起保存；專案後續修改不會回寫此模型。','readiness-item'));
  if(model.yolo_compatibility){const report=model.yolo_compatibility,summary=report.summary||{},audit=element('details',undefined,'training-monitor-details'),heading=document.createElement('summary');heading.textContent=`YOLO Seg 相容稽核 · 修補 ${number(summary.pixels_repaired)} px`;const copy=element('p',`已掃描 ${number(summary.assets_scanned)} 張；${number(summary.affected_assets)} 張的 Run 副本修補 ${number(summary.holes_repaired)} 個微小孔洞。原始標註與固定資料版本未變更，模型內部評估使用相容副本。`,'readiness-item');audit.append(heading,copy);for(const issue of report.repairs||[]){const row=element('div',undefined,'yolo-issue');row.append(element('b',`${issue.name} · ${issue.label||'未命名標註'}`),button('放大位置','text-button',()=>safe(()=>showYoloLocation(issue))),element('p',issue.message));audit.append(row)}root.append(audit)}
  const actions=element('div',undefined,'model-export-actions'),exportButton=button('匯出模型封裝','primary',()=>safe(()=>exportSelectedModel(model.model_version_id)));actions.append(exportButton);const exports=(state.training?.model_exports||[]).filter(item=>item.model_version_id===model.model_version_id);if(exports.length){const latest=exports[0],open=button('開啟最近匯出資料夾','secondary',()=>safe(()=>api('/api/open-folder','POST',{project_id:state.project.id,model_export_id:latest.export_id})));actions.append(open);root.append(actions,element('p',`最近匯出：${latest.export_id} · ${date(latest.created_at)} · ${number(latest.bytes)} bytes`,'field-note'))}else root.append(actions,element('p','封裝包含模型、checkpoint、評估、Epoch 指標、來源 Run 與 SHA-256 manifest，不包含訓練圖片。','field-note'))}
async function exportSelectedModel(modelId){const job=await api(projectPath('/model-exports'),'POST',{model_version_id:modelId}),result=await pollJob(job);await loadTraining();toast(`${result.model_version_id} 已匯出為 ${result.export_id}。`);await api('/api/open-folder','POST',{project_id:state.project.id,model_export_id:result.export_id})}
async function generatePredictions(){const model=state.training?.models?.find(item=>item.model_version_id===state.selectedModel)||state.training?.models?.[0];if(!model)throw Error('請先完成一個模型版本。');let assetIds;if($('predictionTarget').value==='current'){if(!state.asset)throw Error('請先在標註頁選擇圖片。');assetIds=[state.asset.id]}const job=await api(projectPath('/predictions'),'POST',{model_version_id:model.model_version_id,...(assetIds?{asset_ids:assetIds}:{})});await pollJob(job);await loadTraining();toast('預標註候選已產生，接受後會進入待審核。')}
async function acceptPredictions(candidateId){const result=await api(`/api/predictions/${candidateId}/accept`,'POST',{});state.project=result.project;if(state.asset&&result.accepted.includes(state.asset.id))await loadAsset(state.asset.id);await loadTraining();renderAssetList();toast(`${number(result.accepted.length)} 張候選已寫入待審標註。`)}

document.querySelectorAll('[data-stage]').forEach(b=>b.onclick=()=>safe(()=>switchStage(b.dataset.stage)));
document.querySelectorAll('[data-source]').forEach(tab=>{
  tab.onclick=()=>switchSource(tab.dataset.source);
  tab.onkeydown=event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)||state.busy||state.transitioning)return;
    event.preventDefault();const tabs=[...document.querySelectorAll('[data-source]')],index=tabs.indexOf(tab);
    const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
    switchSource(tabs[next].dataset.source);tabs[next].focus();
  };
});
$('cameraActivity').onclick=()=>switchSource('camera');
$('toggleSourceInspector').onclick=()=>setSourceInspector(true,'source');$('toggleImageTools').onclick=()=>setSourceInspector(true,'image');$('closeSourceInspector').onclick=$('inspectorBackdrop').onclick=()=>setSourceInspector(false);
$('showManualImport').onclick=()=>{$('manualImport').open=true;$('importPaths').focus();};
$('processingMode').onchange=()=>{updateProcessingFields();updatePreviewLayout(false);$('processingStatus').textContent=state.camera?'輸出方式已變更，按「套用預覽」開始執行。':'拍攝與錄影保存原始影像。'};
$('previewLayout').onchange=()=>updatePreviewLayout();$('showRawFrame').onchange=()=>updatePreviewLayout();$('showCenterCrosshair').onchange=()=>updatePreviewLayout(false);$('togglePreviewFullscreen').onclick=()=>setCameraPreviewExpanded(!document.querySelector('.camera-workspace').classList.contains('preview-expanded'));
$('cameraTargetTool').onchange=updateCameraTargetTool;$('cameraTargetLabel').onchange=()=>{const shape=cameraTargetShape();if(shape)commitCameraTarget({...shape,label:$('cameraTargetLabel').value.trim()})};
$('cameraTargetUseProcessing').onchange=()=>{if(state.camera&&!state.busy&&!$('applyProcessing').disabled)$('applyProcessing').click()};
$('clearCameraTarget').onclick=()=>clearCameraTarget();installCameraTargetEvents();
$('videoPath').oninput=updateVideoSource;$('videoInterval').oninput=updateVideoSource;
bind('home',()=>switchStage('library'));bind('newProject',createProject);bind('newProjectEmpty',createProject);
bind('refreshProjects',loadProjects);$('projectSearch').oninput=renderProjects;
bind('projectName',renameProject);bind('manageClasses',manageClasses);
bind('openProjectFolder',()=>api('/api/open-folder','POST',{project_id:state.project.id}));
bind('goAnnotate',()=>switchStage('annotate'));bind('addMore',()=>switchStage('acquire'));
$('selectAllAssets').onchange=()=>{for(const asset of (state.project?.assets||[]))if($('selectAllAssets').checked)state.acquireSelection.add(asset.id);else state.acquireSelection.delete(asset.id);renderAcquisitionAssets()};
bind('deleteSelectedAssets',()=>confirmDeleteAssets((state.project?.assets||[]).filter(asset=>state.acquireSelection.has(asset.id))),{busy:true});
bind('saveNow',()=>saver.flush());bind('previousAsset',()=>navigateAsset(-1));bind('nextAsset',()=>navigateAsset(1));
$('assetSearch').oninput=renderAssetList;$('assetFilter').onchange=renderAssetList;
$('toggleAssets').onclick=()=>{const hidden=!$('assetSidebar').hidden;$('assetSidebar').hidden=hidden;document.querySelector('.editor-layout').classList.toggle('hide-assets',hidden);editor.fit(false)};
$('toggleProperties').onclick=()=>{const hidden=!$('propertiesSidebar').hidden;$('propertiesSidebar').hidden=hidden;document.querySelector('.editor-layout').classList.toggle('hide-properties',hidden);editor.fit(false)};
bind('chooseFolder',async()=>{const paths=await nativeChoose('folder');if(paths.length){$('importPaths').value=paths.join('\n');await importPaths(paths)}},{busy:true,task:'匯入資料夾'});
bind('chooseImages',async()=>{const paths=await nativeChoose('images');if(paths.length){$('importPaths').value=paths.join('\n');await importPaths(paths)}},{busy:true,task:'匯入圖片與標註'});
bind('importFiles',()=>importPaths($('importPaths').value.split(/\r?\n/).map(p=>p.trim().replace(/^"|"$/g,'')).filter(Boolean)),{busy:true,task:'檢查並匯入資料'});
bind('mergeSelected',async()=>{await flushAllEdits();const ids=[...$('mergeProjects').querySelectorAll('input:checked')].map(i=>i.value);if(!ids.length)throw Error('請選取至少一份來源專案。');const result=await pollJob(await api(projectPath('/merge'),'POST',{project_ids:ids}));await afterAcquisition(result,'mergeReport')},{busy:true,task:'整併專案資料'});
bind('screenCapture',async()=>{await flushAllEdits();await api(projectPath('/screen'),'POST',{});await afterAcquisition({added:1})},{busy:true,task:'擷取並加入桌面影像'});
bind('chooseVideo',async()=>{const paths=await nativeChoose('video');if(paths.length){$('videoPath').value=paths[0];updateVideoSource()}});
bind('extractVideo',async()=>{await flushAllEdits();const path=$('videoPath').value.trim(),interval=Number($('videoInterval').value);if(!path)throw Error('請先選擇影片或輸入完整路徑。');if(!Number.isFinite(interval)||interval<.1)throw Error('取樣間隔至少為 0.1 秒。');const result=await pollJob(await api(projectPath('/video'),'POST',{path,interval_seconds:interval}));await afterAcquisition(result,'videoReport')},{busy:true,task:'影片畫格取樣'});
bind('findCameras',async()=>{const previous=$('cameraDevice').value,result=await api('/api/camera/devices');$('cameraDevice').replaceChildren();for(const device of result.devices||[])$('cameraDevice').append(new Option(device.name||`相機 ${device.index}`,String(device.index)));if(!result.devices?.length){$('cameraDevice').append(new Option('未偵測到相機','0'));toast('未偵測到相機，請確認連線與其他程式的占用情形。',true)}else{if([...$('cameraDevice').options].some(o=>o.value===previous))$('cameraDevice').value=previous;await loadCameraModes()}},{busy:true,task:'偵測相機與支援模式'});
$('cameraDevice').onchange=()=>safe(loadCameraModes);
$('cameraResolution').onchange=updateCameraMode;
bind('startCamera',()=>cameraCommand('/api/camera/start',cameraConfiguration()),{busy:true,task:'啟動相機預覽'});
bind('stopCamera',async()=>{if(state.recording)throw Error('請先停止錄影，再停止相機。');if(state.autoCapture?.active)stopAutoCapture('相機停止前已結束自動擷取。');await cameraCommand('/api/camera/stop',{})},{busy:true});
bind('takeSnapshot',async()=>{await flushAllEdits();await api(projectPath('/capture'),'POST',{target_shape:cameraTargetPayload()});await afterAcquisition({added:1})},{busy:true});
bind('startAutoCapture',startAutoCapture);bind('stopAutoCapture',()=>stopAutoCapture());
bind('startRecording',async()=>{await api('/api/camera/record/start','POST',{project_id:state.project.id});state.recording=true;updateCameraControls()},{busy:true});
bind('stopRecording',async()=>{const value=await api('/api/camera/record/stop','POST',{});state.cameraDetails=value;state.recording=false;updateCameraControls();const path=value.path||value.video_path||value.recording_path||value.last_recording?.path;if(path){$('videoPath').value=path;updateVideoSource()}toast('錄影已保存，可直接從相機頁取幀。')},{busy:true});
bind('extractRecording',async()=>{await flushAllEdits();const recording=state.cameraDetails?.last_recording,path=recording?.path,interval=Number($('cameraVideoInterval').value);if(!path)throw Error('尚無可取幀的相機錄影。');if(!Number.isFinite(interval)||interval<.1)throw Error('取幀間隔至少為 0.1 秒。');const result=await pollJob(await api(projectPath('/video'),'POST',{path,interval_seconds:interval,target_shape:cameraTargetPayload()}));await afterAcquisition(result,'cameraVideoReport');toast('錄影畫格已加入目前專案。')},{busy:true,task:'從錄影擷取畫格'});
async function applyProcessing(calibrate=false) {
  const mode=$('processingMode').value,settings={};
  if(mode==='binary') {
    const raw=$('processingThreshold').value.trim(),threshold=Number(raw);
    if(!raw||!Number.isInteger(threshold)||threshold<0||threshold>255)throw Error('二值閾值必須為 0 到 255 的整數。');
    settings.threshold=threshold;
  }
  if(['binary','adaptive'].includes(mode))settings.invert=$('processingInvert').checked;
  const target=cameraTargetShape();if(target&&$('cameraTargetUseProcessing').checked)settings.roi_shape=target;
  if(calibrate)settings.calibrate=true;
  const tracker=acquisitionTask;updateTimedTask(tracker,'送出影像輸出設定…',20);await taskCheckpoint(tracker);
  const value=await api('/api/camera/processing','POST',{mode,settings});updateTimedTask(tracker,'等待第一張處理影像…',55);
  for(let attempt=0;attempt<20&&state.camera;attempt++){
    await taskCheckpoint(tracker);const current=await api('/api/camera/status');receiveCameraStatus(current);
    updateTimedTask(tracker,current.processing_state==='ready'?'處理輸出已就緒':'正在建立處理輸出…',Math.min(95,60+attempt*2));
    if(current.processing_error){
      if(current.processing_state==='needs_background'){$('processingStatus').textContent=`${current.processing_error} · 預覽暫時顯示原圖。`;updateTimedTask(tracker,'等待背景校正',100);return current}
      throw Error(current.processing_error);
    }
    if(mode==='original'||current.processing_state==='ready')break;
    await new Promise(resolve=>setTimeout(resolve,180));
  }
  $('processingStatus').textContent=value.processing_error|| (calibrate?'背景校正已送出，請保持場景清空直到預覽穩定。':'預覽設定已更新；拍攝仍保存原圖。');
}
bind('applyProcessing',()=>applyProcessing(false),{busy:true,task:()=>`切換影像輸出 · ${$('processingMode').selectedOptions[0]?.textContent||'預覽'}`});bind('calibrateBackground',()=>{if($('processingMode').value!=='classical')throw Error('請先選擇「古典背景分割」模式，再進行背景校正。');return applyProcessing(true)},{busy:true,task:'背景校正'});
bind('runAI',runAI,{busy:true});
bind('reviewApprove',()=>reviewSelection('approved'),{busy:true});bind('reviewReject',()=>reviewSelection('rejected'),{busy:true});bind('reviewPending',()=>reviewSelection('pending'),{busy:true});bind('assignSelected',assignSelected);
$('reviewSearch').oninput=()=>{state.reviewPage=0;renderReview()};$('reviewFilter').onchange=()=>{state.reviewSelection.clear();state.reviewPage=0;renderReview()};
$('reviewSelectAll').onchange=()=>{for(const asset of reviewPageItems())if($('reviewSelectAll').checked)state.reviewSelection.add(asset.id);else state.reviewSelection.delete(asset.id);renderReview()};
$('reviewPrevious').onclick=()=>{state.reviewPage--;renderReview()};$('reviewNext').onclick=()=>{state.reviewPage++;renderReview()};
bind('prepareTraining',()=>switchStage('split'));
bind('refreshTraining',()=>loadTraining());
bind('prepareAutoSplit',()=>switchStage('split'));
bind('openSplitFlowManager',openSplitManager);
bind('refreshSplitPage',async()=>{await refreshProject();await loadTraining();await renderSplitPage()});
bind('backToReview',()=>switchStage('review'));
bind('continueToTraining',()=>switchStage('train'));
bind('createDatasetVersion',createDatasetVersion,{busy:true,task:'建立固定訓練資料'});
$('trainingDataset').onchange=()=>{invalidateYoloCompatibility();renderTraining()};$('trainingEngine').onchange=()=>{invalidateYoloCompatibility();renderTraining()};$('trainingDevice').onchange=renderTraining;
$('checkYoloCompatibility').onclick=()=>safe(checkYoloCompatibility);
$('openSettingsModels').onclick=()=>safe(()=>openSettings('models',$('trainingEngine').value));
bind('startTraining',startTrainingRun,{busy:true,task:'啟動獨立訓練程序'});
bind('stopTraining',stopTrainingRun,{busy:true});
bind('refreshModels',()=>loadTraining());
bind('generatePredictions',generatePredictions,{busy:true,task:'產生預標註候選'});
bind('openExchange',()=>switchStage('export'));bind('backToModels',()=>switchStage('models'));
$('backgroundTraining').onclick=()=>safe(()=>switchStage('train'));
bind('chooseOutput',async()=>{const paths=await nativeChoose('output');if(paths.length)$('outputPath').value=paths[0]});
$('exportFormat').onchange=()=>{$('formatDescription').textContent=formatDescriptions[$('exportFormat').value];$('acknowledgeLoss').checked=false;resetValidation('目標格式已變更，請重新執行驗證。')};
bind('autoSplit',autoSplitProject,{busy:true,task:'依類別數量自動分割'});bind('validateProject',validateProject,{busy:true});bind('exportProject',exportProject,{busy:true});bind('refreshExports',async()=>{await refreshProject();renderExport()});
$('releaseDrawerClose').onclick=closeReleaseDrawer;$('releaseDrawer').onclick=event=>{if(event.target===$('releaseDrawer'))closeReleaseDrawer()};
bind('help',async()=>{
  const body=element('div');body.append(element('p','建議流程：建立專案 → 匯入／採集 → 標註 → 人工審核 → 智慧資料分割 → 建立固定訓練資料 → 訓練與評估 → 產生預標註候選 → 接受後再次審核。'));
  const list=element('ul');for(const text of ['F11：切換整套 Vision Workbench 的全螢幕與一般視窗。','即時預覽的展開圖示：只放大採集工作區；再次點擊或按 Esc 還原。','相機採集頁：S 擷取目前原始畫格。輸入文字或調整數值時不會觸發。','採集原圖：Q 選擇 Box、W 選擇 Polygon、E 選擇 Mask、Shift+E 擦除、Esc 關閉或取消。','Ctrl S：立即儲存；修改停止後也會自動儲存。','A／D 或左右方向鍵：切換上一張／下一張圖片。','V 選取、R 矩形、P 多邊形、L 折線、K 關鍵點、O 旋轉框。','B 遮罩筆刷、E 橡皮擦、H 平移；滑鼠滾輪縮放。','Enter 完成頂點；Esc 取消尚未完成的繪圖。','Shift 點選多個物件；Ctrl A 全選物件；Delete 刪除選取。','Ctrl Z / Ctrl Y：復原與重做。','選取模式雙擊邊線可插入頂點，Alt 點控制點可移除頂點。','AI 候選須先接受或捨棄，才能離開編輯工作區。','所有修改皆會重新進入待審核，只有已核准資料可以匯出。'])list.append(element('li',text));body.append(list);
  if(saver.dirty){const rescue=button('下載目前未儲存的標註副本','secondary',()=>{const blob=new Blob([JSON.stringify({format:'vision-workbench-recovery',project_id:state.project.id,asset:state.asset},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`recovery-${state.asset.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),10000)});body.append(rescue);}
  await formDialog({title:'操作指南與快捷鍵',body,eyebrow:'WORKBENCH GUIDE'});
});
$('settings').onclick=()=>safe(()=>openSettings());$('closeSettings').onclick=closeSettings;$('settingsShell').onclick=event=>{if(event.target===$('settingsShell'))closeSettings()};
document.querySelectorAll('[data-settings-page]').forEach(item=>item.onclick=()=>showSettingsPage(item.dataset.settingsPage));
$('settingScale').onchange=event=>saveSetting('scale',event.target.value);$('settingDevice').onchange=event=>{saveSetting('device',event.target.value);$('trainingDevice').value=event.target.value;renderTraining()};
$('modelCatalogSearch').oninput=renderModelCatalog;$('modelCatalogFilter').onchange=renderModelCatalog;$('refreshModelCatalog').onclick=()=>safe(()=>loadModelCatalog(true));
$('goModelUpdates').onclick=()=>showSettingsPage('models');$('openDesktopUpdater').onclick=()=>safe(runDesktopUpdate);
$('copyDiagnostics').onclick=()=>safe(async()=>{await navigator.clipboard.writeText($('diagnosticSummary').textContent);toast('診斷摘要已複製。')});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!$('settingsShell').hidden){event.preventDefault();closeSettings();return}if(event.key==='Escape'&&!$('releaseDrawer').hidden){event.preventDefault();closeReleaseDrawer();return}if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();if(state.asset&&!state.busy)safe(()=>saver.flush())}});

$('editorSelector').onchange=()=>safe(()=>switchEditor($('editorSelector').value));
$('previousAssetTop').onclick=()=>safe(()=>navigateAsset(-1));$('nextAssetTop').onclick=()=>safe(()=>navigateAsset(1));
$('backToBuiltin').onclick=()=>switchEditor('builtin');
$('rebootNow').onclick=()=>safe(async()=>{
  if(state.cvatRebootScheduled){
    await api('/api/cvat/reboot/cancel','POST',{});
    state.cvatRebootScheduled=false;$('rebootNow').textContent='立即重新啟動 Windows';
    toast('已取消重新啟動。');return;
  }
  const body=element('div');
  body.append(element('p','Windows 會開始倒數並重新啟動，關機與開機時會裝完待處理的更新，可能需要幾分鐘。'));
  const points=element('ul');
  points.append(element('li','請先儲存其他程式裡未完成的工作。'),
                element('li','這個專案的標註已存在本機，不會遺失。'),
                element('li','倒數期間可以回到這裡按「取消重新啟動」中止。'));
  body.append(points);
  const result=await formDialog({title:'現在重新啟動 Windows？',body,confirm:'重新啟動',eyebrow:'RESTART WINDOWS',
    onSubmit:()=>api('/api/cvat/reboot','POST',{confirm:true})});
  if(!result)return;
  state.cvatRebootScheduled=true;$('rebootNow').textContent='取消重新啟動';
  toast(`Windows 將在 ${result.seconds} 秒後重新啟動。`);
});
$('setupCvat').onclick=()=>safe(async()=>{const snapshot=await api('/api/cvat/setup','POST',{});renderCvatStatus(snapshot);await refreshCvat({openWhenReady:true})});
$('retryCvat').onclick=()=>safe(async()=>{
  $('retryCvat').hidden=true;
  const snapshot=await api('/api/cvat/status');renderCvatStatus(snapshot);
  if(snapshot.ready)await openCvat();
  else if(snapshot.busy)await refreshCvat({openWhenReady:true});
});

async function init() {
  try {
    setupCameraInspectors();
    const [system]=await Promise.all([api('/api/system'),loadProjects()]);state.system=system;
    applySettings();$('trainingDevice').value=settingValue('device');
    if(!system.desktop){clearFileDrag();$('fileDropHint').textContent='拖曳匯入請使用桌面版；瀏覽器版可在右側輸入完整路徑。';}
    $('versionLabel').textContent=`VISION WORKBENCH · ${system.version||'1.0'}`;
    if(system.default_export_path)$('outputPath').placeholder=system.default_export_path;
    if(window.qt?.webChannelTransport&&window.QWebChannel)await new Promise(resolve=>new QWebChannel(qt.webChannelTransport,channel=>{state.nativeBridge=channel.objects.workbenchNative;resolve()}));
    updateSourceInspectorLabel();setSourceInspector(false);updateNavigation();updateCameraControls();updateCameraTargetTool();renderExport();status('本機服務已連線 · 選擇專案開始工作');
  }catch(error){toast(error.message,true);status(error.message,true);}
}
init();
