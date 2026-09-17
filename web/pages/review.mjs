export function createReviewPage(context) {
const {$, state, stats, element, number, date, thumbnailURL, reviewNames, button, safe, selectAsset, switchStage, renderYoloCompatibility, api, projectPath, kind, colorFor, decodeMask, flushAllEdits, updateAssetHeader, toast, checkReviewYoloCompatibility, formDialog, imageURL, saver, editor, renderAssetList}=context;
function filteredReview() {
  const query=$('reviewSearch').value.toLowerCase(),filter=$('reviewFilter').value;
  const reason=$('reviewReasonFilter').value;
  return (state.project?.assets||[]).filter(a=>a.name.toLowerCase().includes(query)&&(filter==='all'||(filter==='correction'?a.review_state==='pending'&&a.source?.review?.needs_correction:a.review_state===filter))&&(!reason||(reason==='疑似模糊'?a.source?.quality?.suspected_blur:a.source?.review?.reason===reason)));
}
function reviewPageItems() {return filteredReview().slice(state.reviewPage*48,(state.reviewPage+1)*48);}
function updateReviewSelection() {
  $('reviewSelectedCount').textContent=`已選 ${state.reviewSelection.size} 張`;
  const page=filteredReview();$('reviewSelectAll').checked=!!page.length&&page.every(a=>state.reviewSelection.has(a.id));
  $('reviewSelectAll').indeterminate=page.some(a=>state.reviewSelection.has(a.id))&&!$('reviewSelectAll').checked;
  for(const id of ['reviewApprove','reviewReject','reviewPending','reviewCorrection','reviewTrash','assignSelected'])$(id).disabled=!state.reviewSelection.size||state.busy;
}
function renderReview() {
  const counts=stats(state.project);const summary=$('reviewStats');summary.replaceChildren();
  for(const [key,name]of Object.entries({total:'全部影像',pending:'待審核',approved:'已核准',rejected:'已排除訓練'})){const chip=element('div',undefined,'stat-chip');chip.append(element('span',name),element('b',number(counts[key])));summary.append(chip);}
  const independence=state.project?.independence_review||{},independencePanel=$('independenceReview');
  independencePanel.classList.toggle('current',!!independence.current);
  $('confirmIndependentAssets').checked=!!independence.current;
  $('independenceReviewTitle').textContent=independence.current?`已確認 ${number(independence.asset_count)} 張核准樣本彼此獨立`:'樣本獨立性尚未確認';
  $('independenceReviewText').textContent=independence.current
    ?`確認時間 ${date(independence.confirmed_at)}；圖片、標註、來源資訊或審核狀態改變時會自動失效。`
    :(independence.reason||'確認目前已核准圖片不是連拍近似影格或同一原圖的重複版本，才可使用圖片層級平衡分割。');
  const pages=Math.max(1,Math.ceil(filteredReview().length/48));state.reviewPage=Math.max(0,Math.min(state.reviewPage,pages-1));
  $('reviewPageLabel').textContent=`第 ${state.reviewPage+1} / ${pages} 頁 · ${number(filteredReview().length)} 張`;
  $('reviewPrevious').disabled=state.reviewPage<=0;$('reviewNext').disabled=state.reviewPage>=pages-1;
  const grid=$('reviewGrid');grid.replaceChildren();const generation=++state.reviewGeneration;
  for(const asset of reviewPageItems()) {
    const card=element('article',undefined,'review-card'+(state.reviewSelection.has(asset.id)?' selected':''));
    const media=element('div',undefined,'review-card-media'),img=document.createElement('img');img.src=thumbnailURL(asset);img.alt=asset.name;img.loading='lazy';
    const check=document.createElement('input');check.type='checkbox';check.checked=state.reviewSelection.has(asset.id);check.setAttribute('aria-label',`選取 ${asset.name}`);
    const label=element('label');label.append(check);label.onclick=e=>e.stopPropagation();
    check.onchange=()=>{if(state.busy){check.checked=state.reviewSelection.has(asset.id);return}if(check.checked)state.reviewSelection.add(asset.id);else state.reviewSelection.delete(asset.id);card.classList.toggle('selected',check.checked);updateReviewSelection()};
    const overlay=document.createElementNS('http://www.w3.org/2000/svg','svg');overlay.setAttribute('viewBox',`0 0 ${asset.width} ${asset.height}`);overlay.setAttribute('preserveAspectRatio','xMidYMid meet');
    media.append(img,overlay,label,element('span',reviewNames[asset.review_state],'badge '+asset.review_state));
    card.tabIndex=0;card.setAttribute('aria-label',`選取 ${asset.name}`);
    card.onclick=event=>{if(event.target.closest('button,input,label'))return;check.checked=!check.checked;check.onchange()};
    card.onkeydown=event=>{if(event.target===card&&[' ','Enter'].includes(event.key)){event.preventDefault();check.checked=!check.checked;check.onchange()}};
    const body=element('div',undefined,'review-card-body');body.append(element('b',asset.name),element('small',`${asset.width} × ${asset.height} · ${asset.shape_count||0} 個物件`));
    const review=asset.source?.review;
    if(review?.reason)body.append(element('small',`${review.reason}${review.note?' · '+review.note:''}`));
    if(asset.source?.quality?.suspected_blur)body.append(element('small','疑似模糊 · 請人工確認'));
    const footer=element('div',undefined,'review-card-footer');footer.append(element('small',asset.split?`${asset.split} · ${asset.batch_id||''}`:asset.batch_id||'尚未指定批次'),button('預覽','text-button',()=>safe(()=>previewReviewAsset(asset))),button('編輯標註 →','text-button',()=>safe(async()=>{await selectAsset(asset.id);await switchStage('annotate')})));body.append(footer);card.append(media,body);grid.append(card);
    queueReviewPreview({asset,overlay,generation});
  }
  if(!grid.children.length)grid.append(element('p','沒有符合篩選條件的圖片。','muted'));
  updateReviewSelection();renderYoloCompatibility();
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
async function reviewSelection(reviewState, reason='', note='') {
  await flushAllEdits();const ids=[...state.reviewSelection];if(!ids.length)throw Error('請先選取需要更新的圖片。');
  const revisions=Object.fromEntries(state.project.assets.filter(a=>state.reviewSelection.has(a.id)).map(a=>[a.id,a.revision]));
  state.project=await api(projectPath('/review'),'POST',{asset_ids:ids,state:reviewState,revisions,reason,note});
  const selectedMeta=state.project.assets.find(a=>a.id===state.asset?.id);if(selectedMeta)Object.assign(state.asset,{review_state:selectedMeta.review_state,revision:selectedMeta.revision});
  state.reviewSelection.clear();state.reviewYoloCompatibility=null;renderReview();updateAssetHeader();toast(`${ids.length} 張圖片已設為${reviewNames[reviewState]}。`);await checkReviewYoloCompatibility();
}
async function assignSelected() {
  const ids=[...state.reviewSelection];if(!ids.length)return;
  const body=element('div');body.append(element('p',`將更新已選取的 ${ids.length} 張圖片。相同拍攝批次應放在同一訓練分割。`));
  const batch=document.createElement('input');batch.placeholder='例如：camera-A-2026-09-09';batch.maxLength=120;
  const split=document.createElement('select');for(const [value,text]of [['','未指定'],['train','train · 訓練'],['val','val · 驗證'],['test','test · 測試']])split.append(new Option(text,value));
  body.append(element('label','來源批次（留空保留原批次）'),batch,element('label','資料分割'),split);
  const result=await formDialog({title:'指定批次與資料分割',body,confirm:'儲存設定',onSubmit:()=>api(projectPath('/assign'),'POST',{asset_ids:ids,split:split.value,...(batch.value.trim()?{batch_id:batch.value.trim()}:{})})});
  if(result){state.project=result;state.reviewYoloCompatibility=null;renderReview();toast('批次與分割設定已更新。');await checkReviewYoloCompatibility()}
}
async function previewReviewAsset(asset) {
  const full=await api(projectPath(`/assets/${asset.id}`)),body=element('div');
  const media=element('div',undefined,'review-card-media'),img=document.createElement('img');
  img.src=imageURL(asset);img.alt=asset.name;img.style.cssText='width:100%;height:auto;max-height:70vh;object-fit:contain';
  const overlay=document.createElementNS('http://www.w3.org/2000/svg','svg');
  overlay.setAttribute('viewBox',`0 0 ${asset.width} ${asset.height}`);overlay.setAttribute('preserveAspectRatio','xMidYMid meet');
  media.append(img,overlay);body.append(media);drawReviewOverlay(overlay,full);
  await formDialog({title:asset.name,body,wide:true});
}
async function excludeReview() {
  const body=element('div'),reason=document.createElement('select'),note=document.createElement('textarea');
  for(const text of ['模糊','曝光問題','重複','無關','遮擋過重','其他'])reason.append(new Option(text,text));
  note.maxLength=2000;body.append(element('p',`排除選取的 ${state.reviewSelection.size} 張圖片；之後可恢復為待審核。`),element('label','排除原因'),reason,element('label','備註'),note);
  await formDialog({title:'排除訓練',body,confirm:'排除選取',onSubmit:async()=>{await reviewSelection('rejected',reason.value,note.value);return true}});
}
async function trashReview() {
  await flushAllEdits();const ids=[...state.reviewSelection];
  const body=element('p',`將 ${ids.length} 張圖片移到可還原垃圾桶。外部原始檔案與既有訓練版本會保留。`);
  const result=await formDialog({title:'從專案移除',body,confirm:'移到垃圾桶',onSubmit:()=>api(projectPath('/review-trash'),'POST',{asset_ids:ids,revisions:Object.fromEntries(state.project.assets.filter(a=>ids.includes(a.id)).map(a=>[a.id,a.revision]))})});
  if(!result)return;
  state.project=result;for(const id of ids)state.reviewSelection.delete(id);
  if(ids.includes(state.asset?.id)){saver.load(null);state.asset=null;editor.clear()}
  renderReview();renderAssetList();
}
async function restoreReview() {
  const items=await api(projectPath('/review-trash-list'),'POST',{}),body=element('div'),selected=new Set();
  body.append(element('p',items.length?'勾選要還原的圖片，還原後會回到待審核。':'垃圾桶目前沒有圖片。'));
  for(const item of items){const row=element('label'),check=document.createElement('input');check.type='checkbox';check.onchange=()=>{if(check.checked)selected.add(item.id);else selected.delete(item.id)};row.append(check,document.createTextNode(` ${item.name} · ${date(item.removed_at)}`));body.append(row)}
  const result=await formDialog({title:'可還原垃圾桶',body,confirm:'還原勾選圖片',onSubmit:items.length?()=>{if(!selected.size)throw Error('請勾選要還原的圖片');return api(projectPath('/review-restore'),'POST',{asset_ids:[...selected],revisions:Object.fromEntries(items.filter(a=>selected.has(a.id)).map(a=>[a.id,a.revision]))})}:null});
  if(result){state.project=result;renderReview();renderAssetList();toast('已還原為待審核圖片。')}
}
return {filteredReview, reviewPageItems, updateReviewSelection, renderReview, queueReviewPreview, runReviewPreview, drawReviewOverlay, reviewSelection, assignSelected, previewReviewAsset, excludeReview, trashReview, restoreReview};
}
