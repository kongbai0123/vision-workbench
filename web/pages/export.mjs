export function createExportPage(context) {
const {$, state, formatDescriptions, element, formatNames, date, button, safe, api, number, flushAllEdits, projectPath, toast, readable, pollJob, refreshProject, openSplitManager}=context;
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
async function autoSplitProject(){await openSplitManager();}
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
return {renderExport, openReleaseDrawer, closeReleaseDrawer, renderBatchTable, autoSplitProject, resetValidation, renderValidation, renderValidationView, validateProject, exportProject};
}
