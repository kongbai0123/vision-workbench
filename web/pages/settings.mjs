import {TaskTiming, taskStage} from '../task-timing.mjs';

const mergedSettingsSections = new Map([
  ['compute', ['general', 'settingsComputeSection']],
  ['diagnostics', ['updates', 'settingsDiagnosticsSection']],
]);

// Settings and model-catalog page; dependencies are injected for browser tests.
export function createSettingsPage(context) {
const {$, state, api, element, button, safe, switchStage, toast, setUpdateIndicators, renderTraining, loadTraining, nativeCallbacks, nativeChoose, formDialog}=context;
const settingDefaults={scale:'1',device:'auto'};
function settingValue(name){try{return localStorage.getItem(`vision-workbench.${name}`)||settingDefaults[name]}catch{return settingDefaults[name]}}
function saveSetting(name,value){try{localStorage.setItem(`vision-workbench.${name}`,value)}catch{}applySettings()}
function applySettings(){document.body.style.zoom=settingValue('scale');if($('settingScale'))$('settingScale').value=settingValue('scale');if($('settingDevice'))$('settingDevice').value=settingValue('device')}

function showSettingsPage(requestedPage='general'){
  const [page, section] = mergedSettingsSections.get(requestedPage) || [requestedPage];
  state.settingsPage=page;
  for(const item of document.querySelectorAll('[data-settings-page]')){
    const active=item.dataset.settingsPage===page;
    item.classList.toggle('active',active);
    if(active)item.setAttribute('aria-current','page');
    else item.removeAttribute('aria-current');
  }
  document.querySelectorAll('[data-settings-panel]').forEach(item=>item.hidden=item.dataset.settingsPanel!==page);
  $('settingsContent').scrollTop=0;
  if(section)$(section).scrollIntoView({block:'start'});
  if(requestedPage==='updates'){
    void refreshDesktopUpdate().catch(error=>renderDesktopUpdate({state:'unavailable',count:null,changes:[],message:error.message}));
    void refreshRemoteVersion().catch(error=>renderRemoteVersion({state:'unavailable',message:error.message}));
  }
}
async function openSettings(page='general',modelKey=null){
  state.settingsFocus=document.activeElement;$('settingsShell').hidden=false;showSettingsPage(page);applySettings();
  if(modelKey)state.selectedCatalogModel=modelKey;
  if(!state.system)state.system=await api('/api/system');
  await loadModelCatalog();renderSettingsSummary();
  if(modelKey&&state.settingsPage==='models')$('settingsTrainingModelsTitle').scrollIntoView({block:'start'});
  $('closeSettings').focus();
}
function closeSettings(){if($('settingsShell').hidden)return;$('settingsShell').hidden=true;state.settingsFocus?.focus?.();state.settingsFocus=null}
$('goSettingsDiagnostics').onclick=()=>{
  showSettingsPage('diagnostics');
  $('settingsDiagnosticsTitle').focus({preventScroll:true});
};
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
function renderRemoteVersion(snapshot){
  if(!snapshot)return;
  state.remoteVersion=snapshot;
  if(['waiting','saving','updating','restarting','updated'].includes(state.remoteUpdate?.state))return;
  const names={checking:'檢查中',available:'發現新版',current:'版本已同步',unavailable:'無法檢查'};
  const badge=$('githubVersionBadge');badge.textContent=names[snapshot.state]||'尚未檢查';
  badge.className='settings-state '+(snapshot.state==='available'?'warning':snapshot.state==='unavailable'?'error':'ready');
  $('githubVersionStatus').textContent=snapshot.message||'尚未檢查 GitHub 版本。';
  $('githubVersionCurrent').textContent=snapshot.current?`目前 v${snapshot.current}`:'目前版本讀取中';
  $('githubVersionLatest').textContent=snapshot.latest?`GitHub v${snapshot.latest}`:'GitHub 版本待確認';
  setUpdateIndicators(state.desktopUpdate?.count??0);
}
async function refreshRemoteVersion(force=false){
  if(!state.nativeBridge?.remoteVersionStatus){renderRemoteVersion({state:'unavailable',message:'GitHub 版本檢查只在 Git 桌面安裝版提供。'});return state.remoteVersion}
  const snapshot=await invokeNativeUpdate(force&&state.nativeBridge.checkRemoteVersion?'checkRemoteVersion':'remoteVersionStatus');renderRemoteVersion(snapshot);return snapshot;
}
function renderRemoteUpdate(snapshot){
  if(!snapshot)return;
  state.remoteUpdate=snapshot;
  const button=$('githubUpdateButton');
  const active=['waiting','saving','updating','restarting','updated'].includes(snapshot.state);
  $('githubVersionStatus').textContent=snapshot.message||'GitHub 更新未完成。';
  const names={waiting:'等待工作完成',saving:'正在儲存',updating:'正在更新',restarting:'正在重新啟動',updated:'正在重新啟動',current:'版本已同步',blocked:'等待處理',unavailable:'無法更新',error:'更新失敗'};
  $('githubVersionBadge').textContent=names[snapshot.state]||'更新狀態';
  $('githubVersionBadge').className='settings-state '+(['blocked','waiting'].includes(snapshot.state)?'warning':['error','unavailable'].includes(snapshot.state)?'error':'ready');
  if(active){
    button.disabled=true;
    button.textContent=['restarting','updated'].includes(snapshot.state)?'正在重新啟動…':snapshot.state==='waiting'?'等待工作完成…':snapshot.state==='saving'?'正在儲存…':'更新中…';
  }else{
    button.disabled=false;button.textContent='從 GitHub 更新';
    if(snapshot.state==='blocked'||snapshot.state==='error')toast(snapshot.message||'GitHub 更新未完成。',true);
  }
}
async function runRemoteUpdate(){
  if(!state.nativeBridge?.applyRemoteUpdate)throw Error('GitHub 更新只在 Windows 桌面版提供。');
  if(['waiting','saving','updating','restarting','updated'].includes(state.remoteUpdate?.state))return;
  renderRemoteUpdate({state:'waiting',message:'正在確認目前工作，儲存後自動更新並重新啟動…'});
  try{renderRemoteUpdate(await invokeNativeUpdate('applyRemoteUpdate'))}
  catch(error){renderRemoteUpdate({state:'error',message:error.message||'GitHub 更新未完成。'})}
}
function renderDesktopUpdate(snapshot){
  if(!snapshot)return;
  state.desktopUpdate={...(state.desktopUpdate||{}),...snapshot};const update=state.desktopUpdate;
  setUpdateIndicators(update.count);
  const statusNames={available:'本機有變更',blocked:'等待處理',updating:'正在更新',restarting:'正在重啟',current:'本機已同步',unavailable:'無法檢查',error:'更新失敗'};
  const badge=$('desktopUpdateBadge');badge.textContent=statusNames[update.state]||'尚未檢查';badge.className='settings-state '+(update.state==='current'?'ready':update.state==='available'||update.state==='blocked'?'warning':update.state==='unavailable'||update.state==='error'?'error':'ready');
  $('desktopUpdateStatus').textContent=update.message||'尚未檢查更新。';
  const changes=Array.isArray(update.changes)?update.changes:[],details=$('desktopUpdateDetails');
  details.hidden=!(Number.isInteger(update.count)&&update.count>0);$('desktopUpdateCount').textContent=update.count>0?`${update.count} 個待套用程式檔案`:'';
  $('desktopUpdateFiles').replaceChildren(...changes.map(path=>element('li',path)));
  if(update.count>0&&!changes.length)$('desktopUpdateFiles').append(element('li','按下更新後會重新確認並列出檔案。'));
  const blockers=Array.isArray(update.blockers)?update.blockers:[],blocker=$('desktopUpdateBlockers');blocker.hidden=!blockers.length;blocker.textContent=blockers.join('\n');
  const action=$('openDesktopUpdater'),running=['updating','restarting'].includes(update.state);action.disabled=running;
  action.textContent=update.state==='available'?'套用本機變更並重新啟動':update.state==='blocked'?'重新檢查':running?'更新中…':update.state==='current'?'再次檢查':'檢查本機變更';
}
async function refreshDesktopUpdate(){
  if(!state.nativeBridge?.updateStatus){renderDesktopUpdate({state:'unavailable',count:null,changes:[],message:'本機程式更新只在 Windows 桌面版提供。'});return state.desktopUpdate}
  const snapshot=await invokeNativeUpdate('updateStatus');renderDesktopUpdate(snapshot);return snapshot;
}
async function runDesktopUpdate(){
  if(!state.nativeBridge?.applyUpdate)throw Error('本機程式更新只在 Windows 桌面版提供。');
  const snapshot=await refreshDesktopUpdate();
  if(snapshot.state!=='available'){
    if(snapshot.state==='current')toast('執行中的程式與本機檔案一致。');
    else if(snapshot.state==='blocked')toast(snapshot.blockers?.[0]||snapshot.message,true);
    else if(snapshot.state!=='updating')toast(snapshot.message||'目前無法開始更新。',true);
    return;
  }
  renderDesktopUpdate(await invokeNativeUpdate('applyUpdate'));
}
nativeCallbacks.updateProgress=snapshot=>{renderDesktopUpdate(snapshot);if(snapshot?.state==='error')toast(snapshot.message||'更新未完成。',true)};
nativeCallbacks.remoteUpdateStatus=renderRemoteUpdate;
async function loadModelCatalog(refresh=false){
  state.modelCatalog=await api(`/api/model-catalog${refresh?'?refresh=1':''}`);
  renderAnnotationModelCatalog(state.modelCatalog.annotation_models);
  renderModelCatalog();renderSettingsSummary();return state.modelCatalog;
}
function renderAnnotationModelCatalog(catalog){
  $('settingsAnnotationModels').hidden=!catalog;
  if(!catalog)return;
  const location=$('settingsAnnotationModelLocation');
  location.replaceChildren('批量找框模型的預設位置為程式資料夾下的 ',element('code',catalog.install_path));
  if(catalog.custom_location)location.append('。目前使用自訂位置，請在該位置準備模型。');
  else if(!catalog.installable)location.append('。目前環境無法從設定安裝。');
  $('startAnnotationModelInstall').disabled=!catalog.installable||Boolean(state.settingsJob);
  const states={available:['已找到檔案','ready'],missing:['未找到','warning'],incomplete:['檔案不完整','error']};
  $('settingsAnnotationModelList').replaceChildren(...catalog.components.map(component=>{
    const [label,tone]=states[component.state]||['尚未檢查',''];
    const row=element('div',undefined,'settings-annotation-model-row'),details=element('div');
    details.append(element('b',component.name));
    const path=element('small',undefined,'muted');
    path.append(element('code',component.install_path));
    if(component.message)path.append(` · ${component.message}`);
    details.append(path);row.append(details,element('span',label,'settings-state '+tone));
    return row;
  }));
}
async function chooseAnnotationModelSource(){
  const body=element('div'),input=element('input'),row=element('div',undefined,'inline annotation-model-source-row');
  body.append(element('p','選擇已備妥模型的 Vision Workbench 資料夾，或其 models/annotation-assist 資料夾。'));
  const label=element('label','來源資料夾的完整路徑');
  input.id='annotationModelSource';input.required=true;input.spellcheck=false;
  label.htmlFor=input.id;row.append(input);
  if(state.system?.desktop)row.append(button('瀏覽資料夾','secondary',async()=>{
    const [source]=await nativeChoose('annotation-models');
    if(source)input.value=source;
  }));
  body.append(label,row);
  return formDialog({title:'安裝批量找框模型',body,confirm:'安裝',onSubmit:()=>{
    const source=input.value.trim();
    if(!source)throw Error('請輸入模型來源資料夾的完整路徑。');
    return source;
  }});
}
$('startAnnotationModelInstall').onclick=()=>safe(async()=>{
  const source=await chooseAnnotationModelSource();
  if(source)await installModelComponent('annotation-assist',{source});
});
function catalogState(model){if(model.train||model.predict)return [model.inference_only?'可匯入／推論':'可使用','ready'];if(model.integration!=='ready')return ['整合開發中','warning'];if(model.runtime_state==='broken')return ['需要修復','error'];return ['尚未安裝','warning']}
function renderModelCatalog(){
  const catalog=state.modelCatalog;if(!catalog)return;const search=$('modelCatalogSearch').value.trim().toLowerCase(),filter=$('modelCatalogFilter').value;
  const models=(catalog.engines||catalog.models||[]).filter(model=>{
    if(filter==='ready'&&!model.train&&!model.predict)return false;if(filter==='unavailable'&&(model.train||model.predict))return false;
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
  const capability=model.inference_only&&model.predict?'可匯入外部權重 · 試跑／預標註／匯出 · 不提供續訓':model.train?`訓練可使用 · ${model.predict?'可產生預標註':'不產生物件預標註'} · 可匯出`:model.unavailable_reason||'尚未準備';
  const facts=element('dl',undefined,'catalog-facts');
  for(const [label,value] of [['任務',model.task_name||model.task],['所需標註',model.annotation||'依模型說明'],['評估指標',(model.metrics||[]).join('、')||'待定'],['執行元件',component?.name||model.component],['元件狀態',stateNames[model.runtime_state]||model.runtime_state],['模型能力',capability]])facts.append(element('dt',label),element('dd',value));
  if(component?.install_path){
    const path=element('dd');path.append(element('code',component.install_path));
    facts.append(element('dt','預設安裝位置'),path);
  }
  root.append(facts,element('p',component?.description||'','field-note'));
  if(component?.install_path)root.append(element('p','移到新電腦後，請在新位置重新安裝此執行環境。','field-note'));
  root.append(element('div',`授權：${model.license||'依來源'}`,'catalog-license'));
  const actions=element('div',undefined,'catalog-actions');
  if(model.train){const use=button('回到訓練並選用','primary',()=>{if(!state.project){toast('請先開啟專案。',true);return}closeSettings();switchStage('train').then(()=>{$('trainingEngine').value=model.key;renderTraining()})});actions.append(use)}
  else if(model.integration==='ready'&&component?.installable){const install=button(component.state==='broken'?'修復必要元件':'安裝必要元件','primary',()=>safe(()=>installModelComponent(component.id)));actions.append(install)}
  else {const planned=button('查看開發狀態','secondary',()=>toast('此模型已納入交付待辦；完成資料、評估與推論 adapter 後才會開放安裝。'));actions.append(planned)}
  root.append(actions);
}
async function installModelComponent(componentId,payload={}){
  if(state.settingsJob)return;
  const isAnnotation=componentId==='annotation-assist';
  const component=(state.modelCatalog.components||[]).find(item=>item.id===componentId);
  if(isAnnotation&&!state.modelCatalog.annotation_models?.installable)throw Error('目前無法從設定安裝標註輔助模型。');
  if(!isAnnotation&&!component)throw Error('找不到模型元件');
  const name=isAnnotation?'標註輔助模型':component.name;
  const action=$('startAnnotationModelInstall'),status=$('settingsJobText'),progress=$('settingsJobProgress');
  state.settingsJob=true;action.disabled=true;
  status.textContent=`正在準備 ${name}`;progress.value=0;progress.hidden=false;
  try{
    let current=await api(`/api/model-components/${componentId}/install`,'POST',payload);
    const jobId=current.id,timing=new TaskTiming();
    state.settingsJob=jobId;timing.update(current.progress,current.state);
    progress.value=Number.isFinite(current.progress)?current.progress:0;
    while(!['succeeded','failed','cancelled'].includes(current.state)){
      await new Promise(resolve=>setTimeout(resolve,650));
      current=await api(`/api/jobs/${jobId}`);
      timing.update(current.progress,current.state,current.progress_phase);
      status.textContent=`${timing.text()} · ${taskStage(current.message||`正在準備 ${name}`)}`;
      progress.value=Number.isFinite(current.progress)?current.progress:0;
    }
    if(current.state!=='succeeded')throw Error(current.error||current.message||'安裝未完成');
  }catch(error){
    status.textContent=`${name} 安裝未完成`;
    throw error;
  }finally{
    state.settingsJob=null;progress.hidden=true;action.disabled=!state.modelCatalog?.annotation_models?.installable;
  }
  status.textContent=isAnnotation?'標註輔助模型檔案已安裝':`${name} 已安裝並通過檢查`;
  toast(isAnnotation?'標註輔助模型檔案已安裝。':`${name} 已可使用。`);
  await loadModelCatalog(!isAnnotation);
  if(!isAnnotation&&state.project)await loadTraining({quiet:true});
}
return {settingValue, saveSetting, applySettings, showSettingsPage, openSettings, closeSettings, renderSettingsSummary, parseNativeResult, invokeNativeUpdate, renderRemoteVersion, refreshRemoteVersion, runRemoteUpdate, renderDesktopUpdate, refreshDesktopUpdate, runDesktopUpdate, loadModelCatalog, catalogState, renderModelCatalog, renderCatalogDetail, installModelComponent};
}
