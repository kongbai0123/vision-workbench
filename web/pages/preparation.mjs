export function createPreparationPage(context) {
const {$,state,SplitManager,api,renderBatchTable,renderReview,resetValidation,createDatasetVersion,loadTraining,toast,flushAllEdits,projectPath,number,element,formDialog,thumbnailURL,drawReviewOverlay}=context;
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
  const splitPurpose=state.project.split_plan?.purpose,imageLevel=splitPurpose==='reviewed_independent'||(!splitPurpose&&info.independence_review?.current);
  const allocationCard=imageLevel
    ?['可分配圖片',number(stats.approved||0),`來源紀錄 ${number(info.groups?.length||0)} 組；建議模式按圖片平衡`]
    :['不可拆來源群組',number(info.groups?.length||0),'正式模式不拆開同來源群組'];
  for(const [label,value,note] of [
    ['已核准圖片',number(stats.approved||0),`排除 ${number(stats.excluded||0)} 張未核准圖片`],
    allocationCard,
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
  if(stats.class_counts){const table=element('table'),head=element('tr');for(const label of ['類別','Train','Validation','Test'])head.append(element('th',label));table.append(head);for(const label of Object.keys(stats.classes||{})){const row=element('tr');row.append(element('td',label));for(const split of ['train','val','test']){const count=stats.class_counts[split]?.[label]||0;const cell=element('td',String(count));if(!count)cell.className=split==='train'?'error':'warning';row.append(cell)}table.append(row)}const wrap=element('div',undefined,'training-table-wrap');wrap.append(table);statusRoot.append(wrap)}
  if(readiness.ready)statusRoot.append(element('div','資料分割符合建立固定資料版本的條件。','readiness-item'));
  for(const item of readiness.blockers||[]){const row=element('div',undefined,'readiness-item error');row.append(element('b',item.message),element('span',item.action));statusRoot.append(row)}
  for(const item of readiness.warnings||[]){const row=element('div',undefined,'readiness-item warning');row.append(element('b',item.message),element('span',item.action));statusRoot.append(row)}
  $('openSplitFlowManager').disabled=!(stats.approved>0);
  $('continueToTraining').disabled=!readiness.ready;
  $('splitFlowNextHint').textContent=readiness.ready?'分割已具備訓練條件；下一步建立固定資料版本。':'請先處理上方阻擋項目。';
  renderAugmentationPreparation();
}
async function setIndependentAssets(confirmed) {
  if(confirmed){
    const body=element('div');body.append(
      element('p','此確認只套用目前已核准圖片。請確認它們不是連拍近似影格、同一原圖的裁切／增強版本，且可視為彼此獨立樣本。'),
      element('p','確認後可依圖片進行多類別平衡；完全相同的圖片仍強制留在同一集合。任何圖片、標註、來源資訊或審核狀態變動都會使確認失效。','muted'));
    const accepted=await formDialog({title:'確認樣本獨立性',body,confirm:'確認目前樣本獨立',onSubmit:()=>true});
    if(!accepted){$('confirmIndependentAssets').checked=false;return;}
  }
  state.project=await api(projectPath('/independence-review'),'POST',{confirmed,revision:state.project.revision});
  renderReview();resetValidation('樣本獨立性確認已更新，請重新驗證資料分割。');
  toast(confirmed?'已保存目前核准樣本的獨立性確認。':'已取消樣本獨立性確認。');
}

const augmentationPresets={
  off:{brightness:0,contrast:0,fliplr:0,flipud:0,degrees:0,translate:0,scale:0,mosaic:0,mixup:0,copy_paste:0,close_mosaic:0},
  light:{brightness:.1,contrast:.1,fliplr:.5,flipud:0,degrees:2,translate:.03,scale:.1,mosaic:0,mixup:0,copy_paste:0,close_mosaic:0},
  standard:{brightness:.2,contrast:.2,fliplr:.5,flipud:0,degrees:0,translate:.1,scale:.5,mosaic:1,mixup:0,copy_paste:0,close_mosaic:10}
};
function augmentationProfile(){
  const preset=$('augmentationPreset')?.value||state.augmentationPreset||'light',base={...(augmentationPresets[preset]||augmentationPresets.light)};
  if(preset==='custom')Object.assign(base,{brightness:Number($('augmentationBrightness').value),contrast:Number($('augmentationContrast').value),fliplr:Number($('augmentationFlipLR').value),flipud:Number($('augmentationFlipUD').value)});
  for(const [key,value] of Object.entries(base))if(!Number.isFinite(value))throw Error(`資料增強 ${key} 不是有效數值。`);
  const expansion=Number($('augmentationExpansion')?.value??state.augmentationExpansion??0);
  if(!Number.isInteger(expansion)||expansion<0||expansion>50)throw Error('每張原圖擴充份數必須是 0 到 50 的整數。');
  return {schema_version:2,preset,apply_to:'train',mode:'online',expansion_count:preset==='off'?0:expansion,...base};
}
function renderAugmentationPreparation(){
  const preset=$('augmentationPreset');if(!preset)return;preset.value=state.augmentationPreset||preset.value;
  const expansionInput=$('augmentationExpansion');expansionInput.disabled=preset.value==='off';expansionInput.value=preset.value==='off'?0:state.augmentationExpansion;
  const profile=augmentationProfile();$('augmentationCustomFields').hidden=profile.preset!=='custom';
  const sample=(state.project?.assets||[]).find(asset=>asset.review_state==='approved'&&asset.split==='train')||(state.project?.assets||[]).find(asset=>asset.review_state==='approved');
  for(const id of ['augmentationOriginalPreview','augmentationResultPreview']){
    const image=$(id);let stage=image.parentElement;
    stage.hidden=!sample;image.hidden=!sample;if(sample)image.src=thumbnailURL(sample);else stage.querySelector('svg')?.remove();
  }
  const result=$('augmentationResultPreview');result.style.filter=`brightness(${1+profile.brightness*.45}) contrast(${1+profile.contrast*.55})`;result.style.transform=profile.fliplr>0?'scaleX(-1)':'none';
  if(sample)api(projectPath(`/assets/${sample.id}`)).then(full=>{
    if($('augmentationOriginalPreview').src!==new URL(thumbnailURL(sample),location.href).href)return;
    for(const [id,flipped] of [['augmentationOriginalPreview',false],['augmentationResultPreview',profile.fliplr>0]]){
      const image=$(id),stage=image.parentElement;let overlay=stage.querySelector('svg');
      if(!overlay){overlay=document.createElementNS('http://www.w3.org/2000/svg','svg');overlay.classList.add('augmentation-annotation-overlay');stage.append(overlay)}
      overlay.replaceChildren();overlay.setAttribute('viewBox',`0 0 ${full.width} ${full.height}`);overlay.setAttribute('preserveAspectRatio','xMidYMid meet');overlay.style.transform=flipped?'scaleX(-1)':'none';drawReviewOverlay(overlay,full);
    }
  }).catch(()=>{});
  const names={off:'關閉',light:'輕量',standard:'標準',custom:'自訂'};
  const trainCount=Number(state.training?.readiness?.stats?.splits?.train||0),expanded=trainCount*profile.expansion_count,total=trainCount+expanded;
  $('augmentationSummary').textContent=`${names[profile.preset]}配方 · 原始 Train ${number(trainCount)} 張 · 每張增加 ${number(profile.expansion_count)} 份 · 每輪共 ${number(total)} 個訓練事件（新增 ${number(expanded)}）· 每份獨立抽樣。水平翻轉 ${(profile.fliplr*100).toFixed(0)}% · 垂直翻轉 ${(profile.flipud*100).toFixed(0)}% · 亮度 ±${(profile.brightness*100).toFixed(0)}% · 對比 ±${(profile.contrast*100).toFixed(0)}%${profile.mosaic?` · Mosaic ${(profile.mosaic*100).toFixed(0)}%（YOLO）`:''}。Validation／Test 保持原始資料。`;
}
return {openSplitManager, renderSplitPage, setIndependentAssets, augmentationProfile, renderAugmentationPreparation};
}
