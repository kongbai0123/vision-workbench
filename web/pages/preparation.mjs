export function augmentationProjection(splits={},expansionCount=0){
  const originals={train:Number(splits.train)||0,val:Number(splits.val)||0,test:Number(splits.test)||0};
  const expansion=Math.max(0,Number(expansionCount)||0),expanded=originals.train*expansion;
  const rows={
    train:{originals:originals.train,expanded,events:originals.train+expanded},
    val:{originals:originals.val,expanded:0,events:originals.val},
    test:{originals:originals.test,expanded:0,events:originals.test},
  };
  return {rows,originals:Object.values(originals).reduce((sum,value)=>sum+value,0),expanded,
    events:Object.values(rows).reduce((sum,row)=>sum+row.events,0),trainEvents:rows.train.events};
}

export function createPreparationPage(context) {
const {$,state,SplitManager,api,renderBatchTable,renderReview,resetValidation,createDatasetVersion,loadTraining,toast,flushAllEdits,projectPath,number,element,thumbnailURL,drawReviewOverlay}=context;
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
  const splitPurpose=state.project.split_plan?.purpose,imageLevel=!splitPurpose||splitPurpose==='reviewed_independent';
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
  const names={off:'關閉',light:'輕量',standard:'標準',custom:'自訂'},splits=state.training?.readiness?.stats?.splits||{};
  const projection=augmentationProjection(splits,profile.expansion_count),projectionRoot=$('augmentationSplitProjection');projectionRoot.replaceChildren();
  for(const [key,label,note] of [['train','Train','每輪訓練'],['val','Validation','驗證保持原始'],['test','Test','最終評估保持原始']]){
    const row=projection.rows[key],card=element('article',undefined,`augmentation-split-card ${key}`),heading=element('div',undefined,'augmentation-split-card-heading');
    heading.append(element('strong',label),element('span',note));
    const equation=element('div',undefined,'augmentation-equation');
    equation.append(element('div',undefined,'augmentation-equation-item'),element('i','＋'),element('div',undefined,'augmentation-equation-item'),element('i','＝'),element('div',undefined,'augmentation-equation-item result'));
    const parts=equation.querySelectorAll('div');parts[0].append(element('b',number(row.originals)),element('small','原始圖片'));parts[1].append(element('b',number(row.expanded)),element('small','新增事件'));parts[2].append(element('b',number(row.events)),element('small',key==='train'?'事件／Epoch':'原始圖片'));
    card.append(heading,equation);projectionRoot.append(card);
  }
  $('augmentationEffectiveTotal').textContent=`${number(projection.events)} 個資料事件`;
  const summary=$('augmentationSummary');summary.replaceChildren(
    element('strong',`${names[profile.preset]}配方 · Train 每輪 ${number(projection.trainEvents)} 個事件`),
    element('span',`${number(projection.originals)} 張原圖的分割不變；新增 ${number(projection.expanded)} 個事件全部留在 Train，Validation／Test 零擴充。`),
    element('small','擴充事件每次載入重新抽樣，但仍來自相同原圖，可能產生相近結果。'));
  const transforms=$('augmentationTransformList');transforms.replaceChildren();
  const badges=profile.preset==='off'?['未啟用隨機增強']:[`水平翻轉 ${(profile.fliplr*100).toFixed(0)}%`,`垂直翻轉 ${(profile.flipud*100).toFixed(0)}%`,`亮度 ±${(profile.brightness*100).toFixed(0)}%`,`對比 ±${(profile.contrast*100).toFixed(0)}%`,`旋轉 ±${profile.degrees.toFixed(0)}°`,`平移 ${(profile.translate*100).toFixed(0)}%`,`縮放 ${(profile.scale*100).toFixed(0)}%`,...(profile.mosaic?[`Mosaic ${(profile.mosaic*100).toFixed(0)}%`]:[])];
  for(const text of badges)transforms.append(element('span',text));
  const currentRevision=state.training?.readiness?.project_revision,latest=state.training?.datasets?.[0];
  const sameSplits=latest&&['train','val','test'].every(key=>Number(latest.splits?.[key]||0)===Number(splits[key]||0));
  const profileKeys=['preset','expansion_count','brightness','contrast','fliplr','flipud','degrees','translate','scale','mosaic','mixup','copy_paste','close_mosaic'];
  const sameProfile=latest&&profileKeys.every(key=>latest.augmentation?.[key]===profile[key]);
  const fixed=!!latest&&latest.project_revision===currentRevision&&sameSplits&&sameProfile;
  const stateBadge=$('augmentationVersionState');stateBadge.textContent=fixed?`已固定於 ${latest.id}`:'尚未固定';stateBadge.className=`augmentation-state ${fixed?'fixed':'draft'}`;
  $('augmentationVersionHint').textContent=fixed?`${latest.id} 已保存目前分割與配方；變更任一設定後需建立新版本。`:'目前只是預覽；建立固定資料版本後，新的 Run 才會使用這些數量。';
  $('createPreparationDatasetVersion').disabled=!state.training?.readiness?.ready;
}
async function createPreparedDatasetVersion(){const created=await createDatasetVersion();await renderSplitPage();return created}
return {openSplitManager, renderSplitPage, augmentationProfile, renderAugmentationPreparation,createPreparedDatasetVersion};
}
