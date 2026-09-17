// Settings and model-catalog page; dependencies are injected for browser tests.
export function createSettingsPage(context) {
const {$, state, api, element, button, safe, switchStage, toast, setUpdateIndicators, renderTraining, loadTraining, nativeCallbacks}=context;
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
nativeCallbacks.updateProgress=snapshot=>{renderDesktopUpdate(snapshot);if(snapshot?.state==='error')toast(snapshot.message||'更新未完成。',true)};
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
return {settingValue, saveSetting, applySettings, showSettingsPage, openSettings, closeSettings, renderSettingsSummary, parseNativeResult, invokeNativeUpdate, renderDesktopUpdate, refreshDesktopUpdate, runDesktopUpdate, loadModelCatalog, catalogState, renderModelCatalog, renderCatalogDetail, installModelComponent};
}
