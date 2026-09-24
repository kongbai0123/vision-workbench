import {WorkflowDraft} from './workflow-draft.mjs';
import {AnnotationEditor, colorFor, shapeNames} from './editor.mjs';
import {kind, decodeMask, encodeMask, brush} from './shapes.mjs';
import {SaveQueue} from './save-queue.mjs';
import {TaskTiming, taskStage} from './task-timing.mjs';
import {TrainingMonitor} from './training-monitor.mjs';
import {TrainingParameters} from './training-parameters.mjs';
import {SplitManager} from './split-manager.mjs';
import {createTrainingPage} from './pages/training.mjs';
import {createSettingsPage} from './pages/settings.mjs';
import {createReviewPage} from './pages/review.mjs';
import {createExportPage} from './pages/export.mjs';
import {createPreparationPage} from './pages/preparation.mjs';
import {createCameraPage} from './pages/camera.mjs';
import {state, mergeProjectDelta} from './state.mjs';
import {nativeCallbacks, installNativeCallbacks, connectNative} from './native-bridge.mjs';
import {nearestVisibleScrollTop} from './scroll-position.mjs';
installNativeCallbacks();

const $ = id => document.getElementById(id);
const reviewNames = {pending:'待審核',approved:'已核准',rejected:'已排除訓練'};
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
  clearTimeout(toastTimer);const element=$('toast');
  element.textContent=String(message);element.classList.toggle('error',error);element.hidden=false;
  // A modal dialog renders in the top layer, above any z-index. The toast joins that
  // layer as a popover, and re-enters it last so it stays above a dialog opened first.
  if(element.popover){if(element.matches(':popover-open'))element.hidePopover();element.showPopover()}
  toastTimer=setTimeout(hideToast,error?9500:4500);
}
function hideToast() {
  const element=$('toast');
  if(element.popover&&element.matches(':popover-open'))element.hidePopover();
  element.hidden=true;
}
function status(message,error=false) {$('statusText').textContent=message;$('connectionDot').classList.toggle('error',error);}
function setUpdateIndicators(count){
  const visible=Number.isInteger(count)&&count>0;
  for(const id of ['updateDot','settingsUpdateNavDot','settingsUpdateCardDot'])if($(id))$(id).hidden=!visible;
  $('desktopUpdateCard')?.classList.toggle('has-update',visible);
  $('settings').title=count===null?'開啟設定中心；暫時無法檢查更新':count>0?`開啟設定中心；有新更新：${count} 個程式檔案已修改`:'開啟設定中心';
}
nativeCallbacks.updateStatus=count=>{
  setUpdateIndicators(count);
  if(state.desktopUpdate?.state==='updating'||state.desktopUpdate?.state==='restarting')return;
  if(count===null)renderDesktopUpdate({state:'unavailable',count:null,changes:[],blockers:[],message:'暫時無法讀取程式檔案，請稍後再試。'});
  else if(count>0&&state.desktopUpdate?.count!==count)renderDesktopUpdate({state:'available',count,changes:[],blockers:[],message:`偵測到 ${count} 個程式檔案有新修改。`});
  else if(count===0&&state.desktopUpdate?.state==='available')renderDesktopUpdate({state:'current',count:0,changes:[],blockers:[],message:'目前執行中的程式已是最新狀態。'});
};
async function api(path,method='GET',body) {
  if(method==='POST'&&state.project&&/\/(review|assign)$/.test(path))body={...body,delta_base:state.project.revision};
  const binary=typeof Blob!=='undefined'&&body instanceof Blob;
  let response;
  try {response=await fetch(path,{method,cache:'no-store',headers:body===undefined?{}:binary?{'Content-Type':'application/octet-stream','X-Workbench':'1','X-Workbench-Filename':encodeURIComponent(body.name||'model.pt')}:{'Content-Type':'application/json','X-Workbench':'1'},...(body===undefined?{}:{body:binary?body:JSON.stringify(body)})});}
  catch {throw Error('無法連線到本機服務。編輯內容仍保留在畫面，請恢復服務後儲存。');}
  let value;try{value=await response.json()}catch{throw Error(`本機服務回應格式無效（${response.status}）。`)}
  if(!response.ok){const e=Error(value.message||value.error||`操作失敗（${response.status}）`);e.status=response.status;e.details=value;throw e;}
  if(value?.delta)return mergeProjectDelta(state.project,value)||await api(`/api/projects/${value.id}`);
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

const {settingValue, saveSetting, applySettings, showSettingsPage, openSettings, closeSettings, renderSettingsSummary, parseNativeResult, invokeNativeUpdate, renderDesktopUpdate, refreshDesktopUpdate, runDesktopUpdate, loadModelCatalog, catalogState, renderModelCatalog, renderCatalogDetail, installModelComponent}=createSettingsPage({$,state,api,element,button,safe,switchStage,toast,setUpdateIndicators,nativeCallbacks,
  renderTraining:(...args)=>renderTraining(...args),loadTraining:(...args)=>loadTraining(...args)});

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
    if(state.asset?.id===asset.id&&saved.mask_cleanup?.pixels_filled&&Array.isArray(saved.shapes)){
      state.asset.shapes=structuredClone(saved.shapes);editor.render();
      toast(`已自動填補 ${number(saved.mask_cleanup.holes_filled)} 個微小孔洞（共 ${number(saved.mask_cleanup.pixels_filled)} px）。`);
    }
    updateAssetHeader();renderAssetList();
    status(`已儲存 ${asset.name} · 修訂 ${String(saved.revision).slice(0,8)}`);
  },
});
const editor = new AnnotationEditor({
  onChange:()=>{saver.changed();updateAssetHeader();},onNotice:toast,onSelection:selected=>updateObjectClassUI(selected),
  onCreated:()=>updateObjectClassUI(),
  onAssetNavigation:direction=>safe(()=>navigateAsset(direction)),
});

async function flushAllEdits() {
  if(editor.gesture)throw Error('請先完成畫布上的拖曳操作。');
  if(editor.draft.length)throw Error('目前仍有尚未完成的頂點。請按 Enter 完成，或按 Esc 取消。');
  if(editor.candidate)throw Error('目前仍有 AI 候選遮罩。請先接受或捨棄，再切換工作或關閉。');
  await saver.flush();
  await workflowDraft.flush();
}
nativeCallbacks.flush=async()=>{if(state.autoCapture?.active)stopAutoCapture('軟體正在結束目前工作，已停止自動擷取。');await flushAllEdits();return true;};
nativeCallbacks.state=()=>({projectId:state.project?.id||null,assetId:state.asset?.id||null,dirty:saver.dirty||workflowDraft.dirty||editor.hasUncommittedWork,stage:state.stage,busy:state.busy||!!state.autoCapture?.active,transitioning:state.transitioning,loading:state.transitioning,saving:!!saver.inFlight,locked:editor.locked});
window.addEventListener('beforeunload',event=>{if(saver.dirty||workflowDraft.dirty||editor.hasUncommittedWork||state.busy||state.autoCapture?.active){event.preventDefault();event.returnValue='';}});

async function safe(work) {try{return await work()}catch(error){toast(error.message,true);status(error.message,true);return null;}}
function bind(id,work,{busy=false,task=null}={}) {
  $(id).onclick=async()=>{
    if($(id).disabled||state.busy||state.transitioning)return;
    $(id).disabled=true;
    if(busy){state.busy=true;editor.locked=true;updateNavigation();}
    try{if(task)await runTimedTask(typeof task==='function'?task():task,work);else await work()}catch(error){if(!error.silent){toast(error.message,true);status(error.message,true)}}
    finally{if(busy){state.busy=false;editor.locked=false;updateNavigation();editor.render();editor.updateCandidate();if(state.stage==='review')updateReviewSelection()}$(id).disabled=false;updateCameraControls();}
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
// Desktop drops never reach the DOM: FileDropBridge consumes the Qt event so the
// original paths survive, so every zone that accepts dropped files registers here.
// The newest registration wins, letting a dialog take the drop from the page below.
const nativeDropZones=[];
function registerNativeDrop(zone) {
  nativeDropZones.push(zone);
  return ()=>{const index=nativeDropZones.indexOf(zone);if(index>=0)nativeDropZones.splice(index,1)};
}
function nativeDropZoneAt(x,y) {
  for(let index=nativeDropZones.length-1;index>=0;index--) {
    const zone=nativeDropZones[index],element=zone.element();
    if(!element||!element.getClientRects().length||!zone.ready())continue;
    const rect=element.getBoundingClientRect();
    if(x<rect.left||x>rect.right||y<rect.top||y>rect.bottom)continue;
    if(element.contains(document.elementFromPoint(x,y)))return zone;
  }
  return null;
}
nativeCallbacks.drag=(kind,x,y,paths=[])=>{
  const zone=kind==='leave'?null:nativeDropZoneAt(x,y);
  for(const other of nativeDropZones)if(other!==zone)other.hover(false);
  if(!zone)return false;
  if(kind!=='drop'){zone.hover(true);return true;}
  zone.hover(false);
  if(!Array.isArray(paths)||!paths.length||paths.some(path=>typeof path!=='string'||!path.trim()))return false;
  return zone.drop([...new Set(paths)])!==false;
};
registerNativeDrop({
  element:()=>$('fileDropZone'),
  ready:()=>state.stage==='acquire'&&state.acquireSource==='files'&&!!state.project&&!state.busy&&!state.transitioning,
  hover:active=>{
    if(!active){clearFileDrag();return;}
    $('fileDropZone').classList.add('drop-active');$('fileDropTitle').textContent='放開以檢查並匯入';
  },
  drop:paths=>{
    $('importPaths').value=paths.join('\n');
    // Reuse the same serialized, revision-aware import action as the file picker.
    $('importFiles').click();
    return true;
  },
});
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
async function confirmDeleteAssets(assets,{review=false}={}) {
  if(!assets.length)return null;
  await flushAllEdits();
  const body=element('div'),shapeCount=assets.reduce((sum,asset)=>sum+Number(asset.shape_count||0),0),approved=assets.filter(asset=>asset.review_state==='approved').length;
  body.append(element('p',review?(assets.length===1?`即將永久刪除「${assets[0].name}」。`:`即將永久刪除選取的 ${assets.length} 張圖片。`):(assets.length===1?`即將刪除「${assets[0].name}」。`:`即將刪除所選的 ${assets.length} 張圖片。`)));
  const details=element('ul');details.append(element('li',`包含 ${shapeCount} 個標註物件`),element('li',`${approved} 張已核准圖片`),element('li','圖片、標註與修訂紀錄會從目前專案移除'));if(review)details.append(element('li','此操作無法從資料審核的垃圾桶還原'));body.append(details);
  const result=await formDialog({title:review?(assets.length===1?'永久刪除這張圖片？':'永久刪除選取圖片？'):(assets.length===1?'刪除這張素材？':'批量刪除素材？'),body,confirm:`${review?'永久':''}刪除 ${assets.length} 張`,eyebrow:'DELETE ASSETS',onSubmit:()=>api(projectPath('/assets'),'DELETE',{asset_ids:assets.map(asset=>asset.id),revisions:Object.fromEntries(assets.map(asset=>[asset.id,asset.revision]))})});
  if(!result)return null;
  const removed=new Set(result.asset_ids||assets.map(asset=>asset.id));state.acquireSelection.clear();state.project=result.project||await api(projectPath());
  if(state.asset&&removed.has(state.asset.id)){saver.load(null);editor.clear();state.asset=null;if(state.project.assets.length)await loadAsset(state.project.assets[0].id);}
  renderAssetList();renderAcquisitionAssets();await loadProjects();status(`已從目前專案刪除 ${number(result.deleted||removed.size)} 張圖片。`);
  return result;
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
  if(state.stage==='review')state.reviewScroll=$('review').scrollTop;
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
    if(stage==='review'){await refreshProject();const ids=new Set(state.project.assets.map(a=>a.id));state.reviewSelection=new Set([...state.reviewSelection].filter(id=>ids.has(id)));renderReview();$('review').scrollTop=state.reviewScroll||0;void checkReviewYoloCompatibility().catch(error=>toast(error.message,true));}
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
    clearTimeout(state.trainingTimer);state.trainingTimer=null;state.training=null;state.selectedRun=null;state.selectedModel=null;state.yoloCompatibility=null;state.yoloCompatibilitySignature='';state.reviewYoloCompatibility=null;state.reviewYoloCompatibilityLoading=false;trainingMonitor.reset();trainingParameters.reset();
    await workflowDraft.load(id);
    state.project=project;state.asset=null;saver.load(null);editor.clear();state.reviewSelection.clear();state.acquireSelection.clear();
    $('shapeLabel').value='';$('cameraTargetLabel').value='';
    state.reviewPage=0;state.reviewScroll=0;$('reviewSearch').value='';$('reviewFilter').value='pending';$('reviewAnnotationFilter').value='all';$('reviewReasonFilter').value='';
    $('projectName').textContent=project.name;$('projectName').title=project.name;updateClassList();
    closeReleaseDrawer();resetValidation('請執行驗證，檢查目前專案及目標格式。');
    $('importReport').hidden=true;$('videoReport').hidden=true;$('mergeReport').hidden=true;
    for(const section of ['library','acquire','annotate','review','split','train','models','export'])$(section).hidden=section!==(project.assets.length?'annotate':'acquire');
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
  $('classSelectionHint').textContent=label?`目前標註類別：${label}`:'請選擇標註類別';
  $('classActionHint').textContent='新增物件與切換圖片會沿用目前類別；既有物件可在下方該列修改。';
  document.querySelectorAll('.class-choice').forEach(choice=>{
    const active=choice.dataset.className===label;choice.classList.toggle('active',active);
    choice.setAttribute('aria-pressed',String(active));
    choice.title=`後續新物件使用「${choice.dataset.className}」`;
  });
}
async function formDialog({title,body,confirm='確定',onSubmit,eyebrow='WORKSPACE',wide=false}) {
  return new Promise(resolve=>{
    const dialog=$('formDialog');$('dialogTitle').textContent=title;$('dialogEyebrow').textContent=eyebrow;
    dialog.classList.toggle('diagnostic-dialog',wide);
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
  const previousScrollTop=list.scrollTop;
  const restoreFocus=list.contains(document.activeElement);
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
  list.scrollTop=previousScrollTop;
  const active=list.querySelector('.asset-row.active');
  if(active&&restoreFocus)active.focus({preventScroll:true});
  if(active)requestAnimationFrame(()=>{
    if(!active.isConnected)return;
    const bounds=list.getBoundingClientRect(),rect=active.getBoundingClientRect();
    list.scrollTop=nearestVisibleScrollTop({
      scrollTop:list.scrollTop,viewportHeight:list.clientHeight,contentHeight:list.scrollHeight,
      itemTop:list.scrollTop+rect.top-bounds.top,itemHeight:rect.height,
    });
  });
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
  updateAssetHeader();renderAssetList();if(typeof renderAnnotationModels==='function')renderAnnotationModels();
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
nativeCallbacks.externalSync=async({source,asset_id})=>{
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
nativeCallbacks.navigate=target=>safe(async()=>{
  if(target==='review')await switchStage('review');
  else if(target!=='builtin')await switchEditor(target);
});

async function nativeChoose(kind) {
  try {const result=await api('/api/dialog','POST',{kind});return result.paths||[];}
  catch(error){toast(`${error.message}\n也可以直接在路徑欄位輸入完整路徑。`,true);return [];}
}
class OperationStopped extends Error {constructor(message='工作已停止。'){super(message);this.silent=true;}}
let acquisitionTask=null,acquisitionTaskHideTimer=null;
function renderAcquisitionTask(task) {
  if(acquisitionTask!==task)return;
  const panel=$('acquisitionTask');panel.hidden=false;panel.classList.toggle('paused',task.paused);panel.classList.toggle('stopping',task.stopping);panel.classList.remove('completing');
  $('acquisitionTaskTitle').textContent=task.title;$('acquisitionTaskMessage').textContent=taskStage(task.message);$('acquisitionTaskElapsed').textContent=task.timing.text();
  $('acquisitionTaskProgress').hidden=true;
  $('pauseAcquisitionTask').textContent=task.paused?'繼續':'暫停';$('pauseAcquisitionTask').disabled=task.stopping;
  $('stopAcquisitionTask').disabled=task.stopping;
}
function beginTimedTask(title) {
  clearTimeout(acquisitionTaskHideTimer);
  if(acquisitionTask)finishTimedTask(acquisitionTask,'stopped','已由下一項工作取代。',0);
  const task={title,message:'準備執行…',progress:null,started:performance.now(),pausedAt:0,pausedTotal:0,paused:false,stopping:false,stopped:false,jobId:null,timer:null};
  task.timing=new TaskTiming();task.timing.update(null);acquisitionTask=task;task.timer=setInterval(()=>renderAcquisitionTask(task),1000);renderAcquisitionTask(task);return task;
}
function updateTimedTask(task,message,progress=null) {if(acquisitionTask!==task||task.stopped)return;task.message=message;if(progress!==undefined)task.progress=progress;task.timing.update(null,task.paused?'paused':task.stopping?'stopping':'running');renderAcquisitionTask(task);}
function finishTimedTask(task,state='complete',message='已完成',delay=1800) {
  if(acquisitionTask!==task)return;clearInterval(task.timer);task.stopped=true;task.paused=false;task.progress=state==='complete'?100:task.progress;task.message=message;
  task.timing.update(null,state);renderAcquisitionTask(task);$('pauseAcquisitionTask').disabled=true;$('stopAcquisitionTask').disabled=true;
  acquisitionTaskHideTimer=setTimeout(()=>{if(acquisitionTask!==task)return;$('acquisitionTask').classList.add('completing');setTimeout(()=>{if(acquisitionTask===task){$('acquisitionTask').hidden=true;$('acquisitionTask').classList.remove('completing');acquisitionTask=null}},230)},delay);
}
async function taskCheckpoint(task) {
  if(task.stopping||task.stopped)throw new OperationStopped();
  while(task.paused&&!task.stopping)await new Promise(resolve=>setTimeout(resolve,120));
  if(task.stopping||task.stopped)throw new OperationStopped();
}
async function runTimedTask(title,work) {
  const task=beginTimedTask(title);
  try{const result=await work(task);await taskCheckpoint(task);finishTimedTask(task,'complete','完成');return result;}
  catch(error){if(error instanceof OperationStopped){finishTimedTask(task,'stopped','已停止',2400);throw error}finishTimedTask(task,'failed','失敗 · '+error.message,4500);throw error}
}
$('pauseAcquisitionTask').onclick=()=>safe(async()=>{
  const task=acquisitionTask;if(!task||task.stopping||task.stopped)return;
  if(task.jobId)await api(`/api/jobs/${task.jobId}/${task.paused?'resume':'pause'}`,'POST',{});
  if(task.paused){task.pausedTotal+=performance.now()-task.pausedAt;task.pausedAt=0;task.paused=false;task.message='繼續執行…'}else{task.paused=true;task.pausedAt=performance.now();task.message='已暫停';}
  task.timing.update(null,task.paused?'paused':'running');renderAcquisitionTask(task);
});
$('stopAcquisitionTask').onclick=()=>safe(async()=>{
  const task=acquisitionTask;if(!task||task.stopping||task.stopped)return;task.stopping=true;task.paused=false;task.message='正在安全停止…';task.timing.update(null,'stopping');renderAcquisitionTask(task);
  if(task.jobId)await api(`/api/jobs/${task.jobId}/cancel`,'POST',{});else finishTimedTask(task,'stopped','已停止等待',2400);
});
async function pollJob(job,onDone,onProgress) {
  let current=job;const id=job.id||job.job_id||job.job;const timing=new TaskTiming();
  if(!id) return onDone ? onDone(job.result||job) : job.result||job;
  const tracker=state.stage==='acquire'?acquisitionTask:null;if(tracker)tracker.jobId=id;
  $('jobStatus').hidden=false;
  try {
    while(true) {
      if(onProgress)onProgress(current);
      timing.update(current.progress,current.state,current.progress_phase);
      $('jobMessage').textContent=`${timing.text()} · ${taskStage(current.message||'正在處理工作…')}`;
      $('jobMessage').title=$('jobMessage').textContent;
      $('jobProgress').hidden=true;
      if(tracker){tracker.paused=current.state==='paused';tracker.stopping=current.state==='stopping';tracker.message=current.message||'正在處理工作…';tracker.progress=current.progress;tracker.timing=timing;renderAcquisitionTask(tracker)}
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
  await flushAllEdits();
  const preview=await pollJob(await api(projectPath('/import-preview'),'POST',{paths}));
  const body=element('div'),selected=new Set(preview.items.map(item=>item.id));
  const count=element('p'),grid=element('div',undefined,'review-grid');
  const update=()=>{count.textContent=`已選 ${selected.size} / ${preview.items.length} 張。疑似模糊僅供參考，可自行保留。`};
  const checks=[];
  body.append(count,button('全選／取消全選','secondary',()=>{const all=selected.size!==preview.items.length;selected.clear();for(const {item,check} of checks){check.checked=all;if(all)selected.add(item.id)}update()}));
  for(const issue of preview.issues||[])body.append(element('p',issue.message));
  for(const item of preview.items){
    const card=element('label',undefined,'review-card'),img=document.createElement('img'),check=document.createElement('input');
    img.src=item.thumbnail;img.alt=item.name;img.style.maxWidth='100%';check.type='checkbox';check.checked=true;
    check.onchange=()=>{if(check.checked)selected.add(item.id);else selected.delete(item.id);update()};checks.push({item,check});
    card.append(check,img,element('b',item.name),element('p',`${item.shape_count} 個標註${item.suspected_blur?' · 疑似模糊':''}`));grid.append(card);
  }
  body.append(grid);update();
  const result=await formDialog({title:'選擇要匯入的圖片',body,wide:true,confirm:'匯入勾選圖片',onSubmit:async()=>{
    if(!selected.size)throw Error('請至少勾選一張圖片。');
    return pollJob(await api(projectPath('/import-confirm'),'POST',{token:preview.token,selected:[...selected]}));
  }});
  if(result)await afterAcquisition({...result,issues:preview.issues});
}
function renderMergeList() {
  const list=$('mergeProjects');list.replaceChildren();
  for(const project of state.projects.filter(p=>p.id!==state.project?.id)) {
    const label=element('label'),check=document.createElement('input');check.type='checkbox';check.value=project.id;label.append(check,element('span',`${project.name} · ${number(stats(project).total)} 張`));list.append(label);
  }
  if(!list.children.length)list.append(element('p','目前沒有其他專案可合併。'));
  list.onchange=updateAcquisitionControls;updateAcquisitionControls();
}

const {cameraTargetDimensions, cameraTargetShape, cameraTargetPayload, targetPoint, targetSvgShape, renderTargetMask, renderCameraTarget, updateCameraTargetTool, commitCameraTarget, clearCameraTarget, updateCameraTargetEditingSurface, installCameraTargetEvents, cameraFlags, loadCameraModes, selectedCameraModes, updateCameraMode, cameraConfiguration, receiveCameraStatus, cameraStatus, cameraCommand, stopPreview, updatePreviewLayout, setCameraPreviewExpanded, updateCameraControls, startPreview, stopAutoCapture, runAutoCapture, startAutoCapture}=createCameraPage({$,state,decodeMask,shapeNames,toast,brush,encodeMask,api,setSourceInspector,updateAcquisitionControls,renderAssetList,renderAcquisitionAssets,flushAllEdits});

const {filteredReview, reviewPageItems, updateReviewSelection, renderReview, queueReviewPreview, runReviewPreview, drawReviewOverlay, reviewSelection, assignSelected, previewReviewAsset, excludeReview, trashReview, deleteReview, restoreReview}=createReviewPage({$,state,stats,element,number,date,thumbnailURL,reviewNames,button,safe,selectAsset,switchStage,api,projectPath,kind,colorFor,decodeMask,flushAllEdits,updateAssetHeader,toast,formDialog,imageURL,saver,editor,renderAssetList,confirmDeleteAssets,
  renderYoloCompatibility:(...args)=>renderYoloCompatibility(...args),checkReviewYoloCompatibility:(...args)=>checkReviewYoloCompatibility(...args)});

const {renderExport, openReleaseDrawer, closeReleaseDrawer, renderBatchTable, autoSplitProject, resetValidation, renderValidation, renderValidationView, validateProject, exportProject}=createExportPage({$,state,formatDescriptions,element,formatNames,date,button,safe,api,number,flushAllEdits,projectPath,toast,readable,pollJob,refreshProject,openSplitManager:(...args)=>openSplitManager(...args)});
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

const {openSplitManager, renderSplitPage, augmentationProfile, renderAugmentationPreparation,renderAugmentationError,createPreparedDatasetVersion}=createPreparationPage({$,state,SplitManager,api,renderBatchTable,renderReview,resetValidation,toast,flushAllEdits,projectPath,number,element,thumbnailURL,drawReviewOverlay,
  createDatasetVersion:(...args)=>createDatasetVersion(...args),loadTraining:(...args)=>loadTraining(...args)});

const {validateTrainingSetup,importExternalModel,invalidateYoloCompatibility, trainingParameters, trainingMonitor, applyTrainingConfigTab, loadTraining, yoloCompatibilitySignature, renderYoloCompatibility, checkReviewYoloCompatibility, checkYoloCompatibility, renderTraining, renderAnnotationModels, createDatasetVersion, startTrainingRun, stopTrainingRun, generatePredictions,startModelTrial,runModelComparison,setTrialFrame,toggleTrialPlayback,drawTrial} = createTrainingPage({$, state, toast, status, api, projectPath, number, stats, date, button, element, settingValue, editor, flushAllEdits, safe, switchStage, formDialog, renderAssetList, loadAsset, selectAsset, pollJob, nativeChoose, registerNativeDrop, augmentationProfile, TrainingParameters, TrainingMonitor});
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
const reasonFilter=$('reviewReasonFilter');
const annotationFilter=$('reviewAnnotationFilter');
bind('reviewCorrection',()=>reviewSelection('pending','待修正'),{busy:true});bind('reviewTrash',trashReview,{busy:true});bind('reviewDelete',deleteReview,{busy:true});bind('reviewRestore',restoreReview,{busy:true});
bind('reviewQuality',async()=>{state.project=await pollJob(await api(projectPath('/review-quality'),'POST',{}));renderReview();toast('模糊提示已更新，請人工確認是否保留。')},{busy:true});
bind('reviewApprove',()=>reviewSelection('approved'),{busy:true});bind('reviewReject',excludeReview,{busy:true});bind('reviewPending',()=>reviewSelection('pending'),{busy:true});bind('assignSelected',assignSelected);
const resetReviewFilter=()=>{state.reviewSelection.clear();state.reviewPage=0;state.reviewScroll=0;renderReview()};
$('reviewSearch').oninput=resetReviewFilter;$('reviewFilter').onchange=resetReviewFilter;annotationFilter.onchange=resetReviewFilter;reasonFilter.onchange=resetReviewFilter;
$('reviewSelectAll').onchange=()=>{for(const asset of filteredReview())if($('reviewSelectAll').checked)state.reviewSelection.add(asset.id);else state.reviewSelection.delete(asset.id);renderReview()};
$('reviewPrevious').onclick=()=>{state.reviewPage--;renderReview()};$('reviewNext').onclick=()=>{state.reviewPage++;renderReview()};
const workflowDraft=new WorkflowDraft({api,collect:()=>{
  trainingParameters.remember();
  return {augmentation:{preset:$('augmentationPreset').value,expansion_count:$('augmentationExpansion').value,
    brightness:$('augmentationBrightness').value,contrast:$('augmentationContrast').value,
    fliplr:$('augmentationFlipLR').value,flipud:$('augmentationFlipUD').value},
    engine:$('trainingEngine').value,dataset:$('trainingDataset').value,
    parameters:Object.fromEntries(trainingParameters.drafts),tab:state.trainingConfigTab};
},restore:payload=>{
  const a=payload.augmentation||{};
  state.augmentationPreset=a.preset||'light';state.augmentationExpansion=a.expansion_count??0;
  $('augmentationPreset').value=state.augmentationPreset;$('augmentationExpansion').value=state.augmentationExpansion;
  for(const [id,key,fallback] of [['augmentationBrightness','brightness',.1],['augmentationContrast','contrast',.1],['augmentationFlipLR','fliplr',.5],['augmentationFlipUD','flipud',0]])$(id).value=a[key]??fallback;
  trainingParameters.reset();trainingParameters.drafts=new Map(Object.entries(payload.parameters||{}));
  $('trainingEngine').replaceChildren(new Option(payload.engine||'',payload.engine||''));
  $('trainingDataset').replaceChildren(new Option(payload.dataset||'',payload.dataset||''));
  state.trainingConfigTab=payload.tab||'basic';state.trainingPreflight=null;
},onError:error=>toast(`設定尚未保存：${error.message}`,true)});
window.addEventListener('training-configuration-changed',()=>workflowDraft.changed());
bind('prepareTraining',()=>switchStage('split'));
bind('refreshTraining',()=>loadTraining());
bind('prepareAutoSplit',()=>switchStage('split'));
bind('openSplitFlowManager',openSplitManager);
bind('refreshSplitPage',async()=>{await refreshProject();await loadTraining();await renderSplitPage()});
bind('backToReview',()=>switchStage('review'));
bind('continueToTraining',()=>switchStage('train'));
bind('createDatasetVersion',createDatasetVersion,{busy:true,task:'建立固定訓練資料'});
bind('createPreparationDatasetVersion',createPreparedDatasetVersion,{busy:true,task:'固定資料分割與增強配方'});
function updatePreparationDraft(){
  state.augmentationPreset=$('augmentationPreset').value;
  state.augmentationExpansion=$('augmentationExpansion').value;
  workflowDraft.changed();state.trainingPreflight=null;
  try{renderAugmentationPreparation()}catch(error){renderAugmentationError(error)}
}
$('augmentationPreset').onchange=updatePreparationDraft;
$('augmentationExpansion').oninput=updatePreparationDraft;
for(const id of ['augmentationBrightness','augmentationContrast','augmentationFlipLR','augmentationFlipUD'])$(id).oninput=updatePreparationDraft;
document.querySelectorAll('[data-training-tab]').forEach(button=>button.onclick=()=>{state.trainingConfigTab=button.dataset.trainingTab;applyTrainingConfigTab();workflowDraft.changed()});
$('trainingDataset').onchange=()=>{invalidateYoloCompatibility();renderTraining();workflowDraft.changed()};$('trainingEngine').onchange=()=>{invalidateYoloCompatibility();renderTraining();workflowDraft.changed()};$('trainingDevice').onchange=renderTraining;
bind('validateTrainingSetup',validateTrainingSetup,{busy:true,task:'驗證設定（不訓練）'});
$('checkYoloCompatibility').onclick=()=>safe(checkReviewYoloCompatibility);
$('openSettingsModels').onclick=()=>safe(()=>openSettings('models',$('trainingEngine').value));
bind('startTraining',startTrainingRun,{busy:true,task:'啟動獨立訓練程序'});
bind('stopTraining',stopTrainingRun,{busy:true});
bind('importModel',importExternalModel);
bind('refreshModels',()=>loadTraining());
bind('generateCurrentPrediction',generatePredictions,{busy:true,task:'產生目前圖片候選'});
bind('trialImages',()=>startModelTrial('images'),{busy:true,task:'外部圖片模型試跑'});
bind('trialVideo',()=>startModelTrial('video'),{busy:true,task:'完整影片模型試跑'});
bind('runModelComparison',runModelComparison,{busy:true,task:'標註比對評估'});
$('trialTimeline').oninput=event=>setTrialFrame(event.target.value);$('trialPlay').onclick=toggleTrialPlayback;$('trialConfidence').oninput=()=>{$('trialConfidenceValue').value=Number($('trialConfidence').value).toFixed(2);drawTrial()};
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
    state.nativeBridge=await connectNative();
    updateSourceInspectorLabel();setSourceInspector(false);updateNavigation();updateCameraControls();updateCameraTargetTool();renderExport();status('本機服務已連線 · 選擇專案開始工作');
  }catch(error){toast(error.message,true);status(error.message,true);}
}
init();
