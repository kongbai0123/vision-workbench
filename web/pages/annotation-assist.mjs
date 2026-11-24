const verdictNames = {accepted:'建議採用', rejected:'建議拒絕', unresolved:'需覆核'};
const runNames = {queued:'等待', running:'執行中', complete:'完成', failed:'失敗', cancelled:'已停止',
  stopped:'已停止', interrupted:'已中斷'};
const voteNames = {A:'符合類別與框界', B:'框界或併框問題', C:'其他物件或類別', D:'證據不足'};
const terminalJobStates = new Set(['succeeded','failed','cancelled']);
const candidatePriority = {unresolved:0, accepted:1, rejected:2};
const archiveNames = {adopted:'已成為人工標註', alternative:'同一物件已有人工標註', rejected:'人工不採用',
  below_floor:'低於候選門檻'};
const samNames = {aligned:'框界一致', mismatch:'框界差異', unavailable:'未取得可靠遮罩框',
  skipped_budget:'超出檢查額度', not_eligible:'未列入框界檢查'};
const healthNames = {passed:'控制案例通過', failed:'控制案例有誤判', inconclusive:'控制案例不足', not_run:'未執行健全性檢查'};
const comparisonNames = {agreement:'與 Qwen 判讀一致', disagreement:'與 Qwen 判讀不同'};
const manualBoxReasons = new Set(['unmatched_scene_target','bad_extent','uncertain_review','low_recommendation_score',
  'competing_extents','conflicting_classes','reference_manual']);

function origin(shape) {
  return shape.metadata?.annotation_assist;
}

function overlaps(first, second) {
  const width = Math.max(0, Math.min(first.x+first.width, second.x+second.width) - Math.max(first.x, second.x));
  const height = Math.max(0, Math.min(first.y+first.height, second.y+second.height) - Math.max(first.y, second.y));
  return width*height / (first.width*first.height + second.width*second.height - width*height);
}

function alreadyOnCanvas(shapes, runId, candidate) {
  if(!candidate.shape)return false;
  return shapes.some(shape => {
    const source = origin(shape);
    if(source?.run_id===runId && source.candidate_id===candidate.id)return true;
    return shape.type!=='mask' && shape.label===candidate.label && overlaps(shape,candidate.shape)>=.85;
  });
}

export function groupAssistCandidates(candidates, shapes, runId, decisions = new Map(), {sameClassOnly=false}={}) {
  const byId = new Map(candidates.map(candidate => [candidate.id, candidate]));
  const confirmed = new Set();
  const anchors = [];
  for(const shape of shapes){
    const source=origin(shape);
    if(!source || source.run_id!==runId || !byId.has(source.candidate_id))continue;
    const original = byId.get(source.candidate_id).shape;
    confirmed.add(source.candidate_id);
    if(original && shape.type!=='mask' && overlaps(shape,original)>=.5)anchors.push(original);
  }
  const active=[], archived=[];
  for(const candidate of candidates){
    let reason;
    if(confirmed.has(candidate.id))reason='adopted';
    else if(decisions.has(candidate.id))reason='rejected';
    else if(candidate.reason_code==='low_score_proposal')reason='below_floor';
    // Each alternative must directly match an adopted origin. An alternative never masks another object.
    else if(candidate.shape && anchors.some(anchor => (!sameClassOnly || anchor.label===candidate.label)
      && overlaps(anchor,candidate.shape)>=.85))reason='alternative';
    if(reason)archived.push({candidate,reason});
    else active.push(candidate);
  }
  return {active,archived};
}

function samShape(candidate,asset) {
  const coordinates=candidate?.sam?.mask_box_xyxy;
  if(!Array.isArray(coordinates)||coordinates.length!==4||coordinates.some(value=>!Number.isFinite(value)))return null;
  const [x,y,right,bottom]=coordinates;
  if(x<0||y<0||right>asset.width||bottom>asset.height||right<=x||bottom<=y)return null;
  return {id:`${candidate.shape?.id||candidate.id}-sam`,label:candidate.label,type:'rectangle',x,y,width:right-x,height:bottom-y};
}

function modelScore(value) {
  return typeof value==='number' && Number.isFinite(value) ? value.toFixed(3) : '—';
}

export function manualBoxRequests(run) {
  const requests=new Map();
  if(run?.status!=='complete')return requests;
  const labels=new Set((run.classes||[]).map(item=>item.label));
  const accepted=new Set(), requested=new Set();
  const usable=(run.frames||[]).filter(item=>item.available!==false);
  for(const frame of usable){
    for(const box of frame.protected_boxes||[])accepted.add(box.label);
    for(const candidate of frame.candidates||[]){
      if(candidate.verdict==='accepted')accepted.add(candidate.label);
    }
    const evidence=new Set((frame.exceptions||[]).filter(issue=>labels.has(issue.label) && manualBoxReasons.has(issue.code)).map(issue=>issue.label));
    for(const label of evidence)requested.add(label);
    if(evidence.size && frame.review?.status!=='resolved'){
      requests.set(frame.asset_id,[...evidence].map(label=>({label,reason:'evidence'})));
    }
  }
  const fallback=usable[0];
  if(!fallback || fallback.review?.status==='resolved')return requests;
  const missing=requests.get(fallback.asset_id)||[];
  for(const label of labels){
    if(!accepted.has(label) && !requested.has(label))missing.push({label,reason:'missing'});
  }
  if(missing.length)requests.set(fallback.asset_id,missing);
  return requests;
}

export function createAnnotationAssistPage({$, state, editor, api, button, element, safe, toast, filteredAssets,
  selectAsset, switchStage, flushAllEdits, date, openAnnotationMessages, returnToAnnotationEditor}) {
  const humanRequests = element('section',undefined,'annotation-assist-human-requests');
  humanRequests.hidden=true;
  $('annotationAssistCandidates').before(humanRequests);
  const useReferences=element('input');
  useReferences.type='checkbox';
  const referenceOption=element('label',undefined,'check-label');
  referenceOption.append(useReferences,document.createTextNode(' 使用已確認的人工範例'));
  $('annotationAssistEnableSam').closest('.annotation-assist-checks').append(referenceOption);
  const jobs = new Map();
  let projectId = null;
  let runs = [];
  let run = null;
  let selectedId = null;
  let selectedAssetId = null;
  let classLabels = null;
  let reviewing = false;
  let fetchSequence = 0;
  let refreshing = null;
  let statusText = '';
  let requestsByAsset = new Map();
  const draftDecisions = new Map();

  const endpoint = pid => `/api/projects/${encodeURIComponent(pid)}/annotation-assist`;
  const frameForAsset = () => run?.frames?.find(frame => frame.asset_id===state.asset?.id);
  const currentCandidate = () => frameForAsset()?.candidates?.find(candidate => candidate.id===selectedId);
  const decisionsFor = (id, frame) => draftDecisions.get(`${id}/${frame.asset_id}`)
    ?? new Map((frame.review?.decisions||[]).map(item=>[item.candidate_id,item.reason]));
  const assetFor = id => state.project?.assets.find(asset => asset.id===id);
  const revisionFor = frame => state.asset?.id===frame.asset_id ? state.asset.revision : assetFor(frame.asset_id)?.revision;
  const isResolved = frame => frame.review?.status==='resolved';
  const pendingFrames = () => (run?.frames || []).filter(frame => frame.available!==false &&
    (frame.exceptions?.length || requestsByAsset.has(frame.asset_id)) && assetFor(frame.asset_id) && !isResolved(frame));
  const groupsFor = (frame, candidates) => groupAssistCandidates(candidates,state.asset?.shapes||[],run.id,
    decisionsFor(run.id,frame),{sameClassOnly:run.use_human_references===true});

  function nextPendingFrame() {
    const pending = new Set(pendingFrames().map(frame => frame.asset_id));
    const frames = run?.frames || [];
    const index = frames.findIndex(frame => frame.asset_id===state.asset?.id);
    return [...frames.slice(index+1),...frames.slice(0,index+1)].find(frame => pending.has(frame.asset_id));
  }

  function renderClasses() {
    const labels = state.project?.classes||[];
    const key = JSON.stringify(labels);
    if(classLabels===key)return;
    classLabels=key;
    const container = $('annotationAssistClasses');
    const previous = new Map([...container.children].map(row => [row.dataset.label, {
      checked:row.querySelector('input[data-assist-class]')?.checked,
      prompt:row.querySelector('input[data-assist-prompt]')?.value,
      description:row.querySelector('input[data-assist-description]')?.value,
    }]));
    container.replaceChildren();
    if(!labels.length){container.append(element('p','請先在物件類別新增要找的類別。','empty-list'));return;}
    for(const label of labels){
      const remembered=previous.get(label);
      const row=element('div',undefined,'annotation-assist-class');
      row.dataset.label=label;
      const selection=element('label',undefined,'check-label');
      const check=element('input');
      check.type='checkbox';
      check.dataset.assistClass=label;
      check.checked=remembered?.checked ?? true;
      check.onchange=renderControls;
      selection.append(check,document.createTextNode(label));
      const prompt=element('input');
      prompt.dataset.assistPrompt=label;
      prompt.value=remembered?.prompt ?? label;
      prompt.placeholder='MM 找框提示詞';
      prompt.setAttribute('aria-label',`${label} 的 MM 提示詞`);
      const description=element('input');
      description.dataset.assistDescription=label;
      description.value=remembered?.description ?? '';
      description.placeholder='Qwen 判斷規則（選填）';
      description.setAttribute('aria-label',`${label} 的 Qwen 判斷規則`);
      row.append(selection,prompt,description);
      container.append(row);
    }
  }

  function selectedClasses() {
    return [...$('annotationAssistClasses').querySelectorAll('[data-assist-class]:checked')].map(input => {
      const row = input.closest('.annotation-assist-class');
      return {label:input.dataset.assistClass,prompt:row.querySelector('[data-assist-prompt]').value.trim(),
        description:row.querySelector('[data-assist-description]').value.trim()};
    });
  }

  function scopeAssets() {
    return $('annotationAssistScope').value==='visible' ? filteredAssets() : state.asset ? [state.asset] : [];
  }

  function renderControls() {
    const job=jobs.get(projectId);
    const selected=selectedClasses();
    const images=scopeAssets().length;
    $('annotationAssistClassCount').textContent = `已選 ${selected.length} 類`;
    $('annotationAssistStart').disabled=!projectId||!images||!selected.length||!!job;
    $('annotationAssistStart').textContent=images ? `開始找框 · ${images} 張` : '開始批量找框';
    $('annotationAssistStop').hidden=!job?.id;
    $('annotationAssistStop').disabled=!!job?.stopping;
    $('annotationAssistEnableSam').disabled=!!job;
    $('annotationAssistEnableGemma').disabled=!!job;
    useReferences.disabled=!!job;
    $('annotationAssistStatus').textContent=job?.message||statusText||run?.error||run?.message||
      (run ? `批次${runNames[run.status]||run.status}。` : '尚無執行紀錄。');
    $('annotationAssistResult').hidden=!!job||!runs.length;
    const progress=$('annotationAssistProgress');
    progress.hidden=!job;
    progress.value=Number(job?.progress)||0;
  }

  function renderRuns() {
    const select=$('annotationAssistRuns');
    const activeId=run?.id || select.value;
    select.replaceChildren();
    for(const item of runs){
      const title=`${date(item.created_at)} · ${runNames[item.status]||item.status} · ${item.summary?.images ?? 0} 張`;
      select.append(new Option(title,item.id));
    }
    if(runs.some(item=>item.id===activeId))select.value=activeId;
    $('annotationAssistResult').hidden=jobs.has(projectId)||!runs.length;
  }

  function renderReviewLink() {
    const count=pendingFrames().length;
    $('annotationAssistReviewEntry').hidden=!count;
    $('annotationAssistReviewCount').textContent=`目前 AI 批次有 ${count} 張影像待覆核`;
  }

  function renderMessageFrames() {
    const root=$('annotationMessageBatchFrames');
    if(!root)return;
    const assetNames=new Map((state.project?.assets||[]).map(asset=>[asset.id,asset.name]));
    const running=jobs.get(projectId);
    const requested=running ? running.imageIds.map(asset_id=>({asset_id})) : run?.frames||[];
    const frames=requested.filter(frame=>frame.available!==false && assetNames.has(frame.asset_id));
    const scroll=root.scrollTop;
    root.replaceChildren();
    for(const frame of frames){
      const pending=!running && frame.review?.status!=='resolved' && (frame.exceptions?.length || requestsByAsset.has(frame.asset_id));
      const row=button('', 'asset-row annotation-message-frame',()=>safe(()=>selectAsset(frame.asset_id)));
      row.dataset.assetId=frame.asset_id;
      if(frame.asset_id===state.asset?.id){row.classList.add('active');row.setAttribute('aria-current','true');}
      row.append(element('b',frame.name||assetNames.get(frame.asset_id)));
      if(pending)row.append(element('small','待人工處理'));
      root.append(row);
    }
    root.scrollTop=scroll;
  }

  function renderHumanRequests(frame) {
    const requests=(requestsByAsset.get(frame?.asset_id)||[]).filter(item=>state.project?.classes.includes(item.label));
    const preparation=frame ? run?.reference_preparation?.classes||[] : [];
    humanRequests.replaceChildren();
    humanRequests.hidden=!requests.length && !preparation.length;
    if(humanRequests.hidden)return;
    humanRequests.append(element('b',requests.length?'待人工確認類別':'人工範例'));
    for(const {label,status} of preparation){
      const message = status==='applied' ? '已使用獨立來源的人工範例' : '人工範例不足或描述無法確認，需人工處理';
      humanRequests.append(element('p',`${label} · ${message}`,'field-note'));
    }
    for(const {label,reason} of requests){
      const line=element('div',undefined,'annotation-assist-human-request');
      const text=reason==='missing' ? `這批未找到可靠的「${label}」。若圖中有這個類別，請先框一個範例。`
        : `「${label}」的類別或框界無法確認。請在人眼確認後補框或修框。`;
      line.append(element('p',text));
      const actions=element('div',undefined,'inline');
      const draw=button(`框選 ${label}`,'secondary',()=>safe(()=>returnToAnnotationEditor(label)));
      draw.dataset.assistManualLabel=label;
      actions.append(draw);
      const repeat=button('以人工範例重跑','secondary',()=>safe(()=>{
        const definition=run.classes.find(item=>item.label===label);
        return start({classes:[definition],imageIds:[frame.asset_id],useHumanReferences:true});
      }));
      repeat.disabled=jobs.has(projectId);
      actions.append(repeat);
      line.append(actions);
      humanRequests.append(line);
    }
  }

  function showRunHealth() {
    const root=$('annotationAssistGemmaHealth');
    root.replaceChildren();
    root.hidden=!run?.gemma;
    if(!run?.gemma)return;
    const {health,status,error}=run.gemma;
    const outcome=status==='failed' ? '執行失敗' : healthNames[health?.status] || '沒有控制案例結果';
    root.append(element('b',`Gemma 4 · 僅觀察 · ${outcome}`));
    if(error||health?.reason)root.append(element('div',error||health.reason));
  }

  function showCandidateModels(candidate) {
    const root=$('annotationAssistCandidateModels');
    root.replaceChildren();
    if(!candidate)return;
    const qwenVotes=candidate.votes;
    if(Array.isArray(qwenVotes) && qwenVotes.length){
      const choices=qwenVotes.map(vote=>voteNames[vote?.winner]||'未定');
      root.append(element('div',`Qwen 兩次判讀 · ${choices.join(' ／ ')}`));
    }
    const sam=candidate.sam;
    if(sam){
      const metrics=[];
      if(Number.isFinite(sam.iou))metrics.push(`外接框 IoU ${modelScore(sam.iou)}`);
      if(Number.isFinite(sam.mask_score))metrics.push(`遮罩分數 ${modelScore(sam.mask_score)}`);
      root.append(element('b',`SAM 2.1 · ${samNames[sam.status]||'未檢查'}`));
      if(metrics.length)root.append(element('div',metrics.join(' · ')));
      if(sam.reason)root.append(element('div',sam.reason));
    }
    const gemma=candidate.gemma;
    if(gemma?.verdict){
      root.append(element('b',`Gemma 4 · 僅觀察 · ${verdictNames[gemma.verdict]||'未定'}`));
      const health=run?.gemma?.health?.status;
      if(health==='failed')root.append(element('div','控制案例有誤判，此判讀僅供觀察。'));
      else if(health==='passed'&&comparisonNames[gemma.qwen_comparison])
        root.append(element('div',comparisonNames[gemma.qwen_comparison]));
      if(gemma.reason)root.append(element('div',gemma.reason));
      if(Array.isArray(gemma.votes) && gemma.votes.length)
        root.append(element('div',gemma.votes.map(vote=>voteNames[vote?.winner]||'未定').join(' ／ ')));
    }else if(run?.gemma?.status==='observed')root.append(element('div','Gemma 4 · 本候選未抽查'));
  }

  function renderEvidence(frame, candidate) {
    const root=$('annotationAssistEvidence');
    root.replaceChildren();
    const sam=state.asset ? samShape(candidate,state.asset) : null;
    const layers=$('annotationAssistLayers');
    layers.hidden=!candidate?.shape && !sam;
    $('annotationAssistShowOriginal').disabled=!candidate?.shape;
    $('annotationAssistShowSam').disabled=!sam;
    const showing=editor.activeAiTool==='batch' && state.stage==='annotate' && !jobs.has(projectId);
    editor.setAssistPreview(
      showing && $('annotationAssistShowOriginal').checked ? candidate?.shape ?? null : null,
      showing && $('annotationAssistShowSam').checked ? sam : null,
    );
    showCandidateModels(candidate);
    const adopt=$('annotationAssistAdopt');
    const reject=$('annotationAssistReject');
    adopt.disabled=true;
    reject.disabled=true;
    if(!candidate || !state.asset)return;
    const present=alreadyOnCanvas(state.asset.shapes||[],run.id,candidate);
    const previouslyRejected=decisionsFor(run.id,frame).has(candidate.id);
    const labelExists=state.project.classes.includes(candidate.label);
    const archived=groupsFor(frame,frame.candidates||[]).archived.find(item=>item.candidate.id===candidate.id);
    adopt.disabled=!candidate.shape||present||!labelExists||reviewing||frame.available===false||
      (archived && archived.reason!=='rejected');
    reject.disabled=!!archived||reviewing||frame.available===false;
    if(candidate.reason)root.append(element('div',candidate.reason));
    if(archived)root.append(element('div',archiveNames[archived.reason]));
    if(present&&!archived)root.append(element('div','畫布已有對應標註。可直接選取並編輯現有框。'));
    if(!present&&!archived&&candidate.verdict==='accepted')root.append(element('div','候選待確認，可直接採用。'));
    if(!labelExists)root.append(element('div','此候選的類別已變更，請依目前類別手動覆核。'));
    if(previouslyRejected)root.append(element('div','已標記為不採用，確認整張後記錄。'));
  }

  function renderFrame() {
    requestsByAsset=manualBoxRequests(run);
    renderMessageFrames();
    renderReviewLink();
    const frame=frameForAsset();
    const summary=run?.summary;
    const exceptions=$('annotationAssistExceptions');
    const list=$('annotationAssistCandidates');
    const archive=$('annotationAssistArchive');
    const archivedList=$('annotationAssistArchivedCandidates');
    const previousAsset=selectedAssetId;
    selectedAssetId=frame?.asset_id??null;
    const previousScroll=previousAsset===selectedAssetId ? list.scrollTop : 0;
    if(previousAsset!==selectedAssetId){selectedId=null;archive.open=false;}
    exceptions.replaceChildren();
    list.replaceChildren();
    archivedList.replaceChildren();
    archive.hidden=true;
    renderHumanRequests(frame);
    showRunHealth();
    $('annotationAssistSummary').textContent=summary
      ? `已處理 ${summary.completed_images ?? 0}／${summary.images ?? 0} 張 · 待覆核 ${pendingFrames().length} 張`
      : '';
    $('annotationAssistResolve').disabled=run?.status!=='complete'||!frame||frame.available===false||reviewing||isResolved(frame);
    const next=nextPendingFrame(), nextButton=$('annotationAssistNext');
    nextButton.disabled=!next;
    nextButton.dataset.assistAssetId=next?.asset_id||'';
    exceptions.dataset.assistAssetId=frame?.asset_id||'';
    if(!frame||frame.available===false){
      selectedId=null;
      if(run)exceptions.textContent='目前影像沒有這批的結果。';
      renderEvidence(null,null);
      return;
    }
    if(run.status!=='complete'){
      exceptions.textContent=run.error||run.message||'這批尚未完成。';
      renderEvidence(null,null);
      return;
    }
    const candidates=[...(frame.candidates||[])].sort((first,second)=>
      candidatePriority[first.verdict]-candidatePriority[second.verdict]);
    const {active,archived}=groupsFor(frame,candidates);
    if(!candidates.some(candidate=>candidate.id===selectedId)){
      selectedId=active.find(item=>item.verdict==='accepted')?.id??active.find(item=>
        item.verdict==='unresolved' && item.reason_code!=='low_recommendation_score')?.id??null;
    }
    if(isResolved(frame))exceptions.append(element('div','已完成人工覆核'));
    else if(frame.exceptions?.length){
      const reasons=[...new Set(frame.exceptions.map(item=>item.reason||item.code))];
      exceptions.append(element('b','待人工覆核'));
      const reasonList=element('ul');
      for(const reason of reasons)reasonList.append(element('li',reason));
      exceptions.append(reasonList);
    }else exceptions.append(element('div',candidates.length?'已產生候選，確認畫布後可提交覆核。':'這張影像沒有可用候選。'));
    if(frame.base_revision!==revisionFor(frame) && !isResolved(frame))exceptions.append(element('div','標註已更新。請以目前畫布為準，比對後再提交。','muted'));
    if(frame.confidence?.accepted_count && frame.confidence.score!=null){
      const evidence=element('div',`建議框最低支持值 ${modelScore(frame.confidence.score)}`,'muted');
      evidence.title='綜合 MM 與 Qwen 對已建議框的支持值，未校準為準確率，且不含漏框風險。';
      exceptions.append(evidence);
    }
    function appendCandidate(candidate, target, archivedReason){
      const row=button('','annotation-assist-candidate',()=>{
        selectedId=candidate.id;
        for(const item of [...list.children,...archivedList.children])
          item.setAttribute('aria-pressed',String(item===row));
        renderEvidence(frame,candidate);
      });
      row.dataset.assistCandidateId=candidate.id;
      row.dataset.assistLabel=candidate.label;
      row.dataset.assistVerdict=candidate.verdict;
      row.setAttribute('aria-pressed',String(candidate.id===selectedId));
      const present=candidate.shape && alreadyOnCanvas(state.asset.shapes||[],run.id,candidate);
      const verdict=verdictNames[candidate.verdict]||verdictNames.unresolved;
      const decision=candidate.reason_code==='low_score_proposal' ? '低於門檻，未送 Qwen'
        : candidate.reason_code==='low_recommendation_score' ? '低於建議採用門檻'
        : candidate.votes?.length ? `Qwen ${verdict}` : verdict;
      const source=candidate.query_source==='reference'?' · 人工範例':'';
      row.append(element('b',candidate.label),element('small',`MM ${modelScore(candidate.score)} · ${decision}${source}`));
      if(archivedReason!=='below_floor' && (archivedReason||present))row.append(element('small',archiveNames[archivedReason]||'畫布已有標註'));
      target.append(row);
    }
    for(const candidate of active)appendCandidate(candidate,list);
    for(const {candidate,reason} of archived)appendCandidate(candidate,archivedList,reason);
    archive.hidden=!archived.length;
    $('annotationAssistArchiveLabel').textContent=`候選紀錄（${archived.length}）`;
    if(candidates.length&&!active.length)list.append(element('div','這張影像沒有待處理的候選框。','muted'));
    list.scrollTop=previousScroll;
    renderEvidence(frame,currentCandidate());
  }

  function onContextChange() {
    const nextProject=state.project?.id??null;
    if(nextProject!==projectId){
      projectId=nextProject;
      fetchSequence++;
      refreshing=null;
      runs=[];
      run=null;
      selectedId=null;
      selectedAssetId=null;
      classLabels=null;
      $('annotationAssistClassSetup').open=true;
      $('annotationAssistClasses').replaceChildren();
      draftDecisions.clear();
      statusText='';
      renderRuns();
    }
    renderClasses();
    renderControls();
    renderFrame();
  }

  async function loadRun(pid,id) {
    const sequence=++fetchSequence;
    const result=await api(`${endpoint(pid)}/${encodeURIComponent(id)}`);
    if(projectId!==pid||sequence!==fetchSequence)return;
    if(run?.id!==result.id){
      selectedId=null;
      if(result.status==='complete')$('annotationAssistClassSetup').open=false;
    }
    run=result;
    statusText='';
    renderRuns();
    renderControls();
    renderFrame();
  }

  function onSaved(asset) {
    const frame=run?.frames?.find(item=>item.asset_id===asset.id);
    if(!reviewing && frame?.review && frame.revision!==asset.revision){
      void safe(()=>loadRun(projectId,run.id));
    }
  }

  async function refresh(preferredId) {
    if(!projectId)return;
    if(refreshing){await refreshing;if(!preferredId)return;}
    const pid=projectId;
    let task;
    task=(async()=>{
      const payload=await api(endpoint(pid));
      if(projectId!==pid)return;
      runs=payload.runs||[];
      const previous=preferredId||run?.id;
      const current=runs.find(item=>item.id===previous)||runs[0];
      if(current)await loadRun(pid,current.id);
      else{run=null;renderRuns();renderFrame();}
    })();
    refreshing=task;
    try{await task;}finally{if(refreshing===task)refreshing=null;}
  }

  async function start({classes=selectedClasses(),imageIds=scopeAssets().map(asset=>asset.id),
    useHumanReferences=useReferences.checked}={}) {
    if(!projectId||jobs.has(projectId))return;
    const pid=projectId;
    if(!classes.length||!imageIds.length)throw Error('請選取影像及至少一個類別。');
    if(classes.some(item=>!item.prompt))throw Error('已選取的類別都需要 MM 提示詞。');
    const modelChecks={sam:$('annotationAssistEnableSam').checked,gemma:$('annotationAssistEnableGemma').checked};
    const active={id:null,progress:0,message:'準備批次…',imageIds};
    jobs.set(pid,active);
    renderControls();
    renderFrame();
    try{
      await flushAllEdits();
      if(projectId!==pid)return;
      if(!imageIds.includes(state.asset?.id))await selectAsset(imageIds[0]);
      if(projectId!==pid)return;
      if(!openAnnotationMessages('batch'))return;
      let job=await api(endpoint(pid),'POST',{asset_ids:imageIds,classes,model_checks:modelChecks,
        ...(useHumanReferences && {use_human_references:true})});
      active.id=job.id||job.job_id;
      const runId=job.run_id;
      if(!active.id)throw Error('背景作業未回傳識別碼。');
      while(job){
        active.progress=job.progress;
        if(!active.stopping)active.message=job.message||'執行 MM／Qwen…';
        if(projectId===pid)renderControls();
        if(terminalJobStates.has(job.state))break;
        await new Promise(resolve=>setTimeout(resolve,600));
        job=await api(`/api/jobs/${active.id}`);
      }
      if(projectId!==pid)return;
      if(job.state==='failed')throw Error(job.error?.message||job.error||job.message||'批量找框失敗。');
      statusText=job.state==='cancelled' ? '批量找框已停止。' : '找框與裁決已完成。';
      await refresh(job.result?.run_id||job.result?.id||runId);
    }catch(error){
      if(projectId===pid){statusText=`批量找框失敗 · ${error.message}`;throw error;}
    }finally{
      jobs.delete(pid);
      if(projectId===pid){renderControls();renderFrame();}
    }
  }

  async function stop() {
    const job=jobs.get(projectId);
    if(!job?.id||job.stopping)return;
    job.stopping=true;
    job.message='正在停止…';
    renderControls();
    try{await api(`/api/jobs/${job.id}/cancel`,'POST',{});}
    catch(error){job.stopping=false;job.message=error.message;renderControls();throw error;}
  }

  function adopt() {
    const candidate=currentCandidate();
    if(!candidate?.shape || !state.asset || !run || frameForAsset()?.available===false || !state.project.classes.includes(candidate.label))return;
    if(alreadyOnCanvas(state.asset.shapes||[],run.id,candidate))return;
    const archived=groupsFor(frameForAsset(),frameForAsset().candidates||[]).archived.find(item=>item.candidate.id===candidate.id);
    if(archived&&archived.reason!=='rejected')return;
    const shape=editor.add({...candidate.shape,metadata:{...candidate.shape.metadata,source:'manual',annotation_assist:{
      ...candidate.shape.metadata?.annotation_assist,run_id:run.id,candidate_id:candidate.id,
      generated_label:candidate.label,score:candidate.score,human_confirmed:true,
    }}});
    if(!shape)return;
    const frame=frameForAsset();
    const decisions=decisionsFor(run.id,frame);
    if(decisions.delete(candidate.id))draftDecisions.set(`${run.id}/${frame.asset_id}`,decisions);
    editor.setTool('select');
    renderFrame();
    toast('已採用為人工標註。同一物件的替代框已收起。');
  }

  function reject() {
    const candidate=currentCandidate();
    const frame=frameForAsset();
    if(!candidate||!frame||!run)return;
    const decisions=decisionsFor(run.id,frame);
    decisions.set(candidate.id,'人工不採用候選');
    draftDecisions.set(`${run.id}/${frame.asset_id}`,decisions);
    renderFrame();
  }

  async function resolve() {
    const pid=projectId;
    const frame=frameForAsset();
    const runId=run?.id;
    if(!frame||frame.available===false||run?.status!=='complete'||!runId||reviewing||isResolved(frame))return;
    reviewing=true;
    renderFrame();
    try{
      await flushAllEdits();
      if(projectId!==pid||run?.id!==runId||state.asset?.id!==frame.asset_id)return;
      const decisions=[...decisionsFor(runId,frame)].map(([candidate_id,reason])=>({candidate_id,reason}));
      const result=await api(`${endpoint(pid)}/${encodeURIComponent(runId)}/review`,'POST',{
        asset_id:frame.asset_id,revision:state.asset.revision,candidate_ids:[],decisions,
      });
      if(projectId!==pid||run?.id!==runId)return;
      frame.review=result.review;
      frame.revision=result.asset.revision;
      run.summary=result.summary||run.summary;
      draftDecisions.delete(`${runId}/${frame.asset_id}`);
      toast('已確認這張影像的例外。');
    }finally{reviewing=false;renderFrame();}
  }

  async function nextException() {
    const next=nextPendingFrame();
    if(!next){toast('這批沒有未覆核的例外。');return;}
    if(state.stage==='review')await switchStage('annotate');
    if(state.stage==='annotate'){
      await selectAsset(next.asset_id);
      openPanel();
    }
  }

  function openPanel() {
    openAnnotationMessages('batch');
    renderFrame();
  }

  $('annotationAssistScope').onchange=renderControls;
  $('annotationAssistStart').onclick=()=>safe(start);
  $('annotationAssistStop').onclick=()=>safe(stop);
  $('annotationAssistRuns').onchange=()=>safe(()=>loadRun(projectId,$('annotationAssistRuns').value));
  for(const id of ['annotationAssistShowOriginal','annotationAssistShowSam'])
    $(id).onchange=()=>renderEvidence(frameForAsset(),currentCandidate());
  $('annotationAssistAdopt').onclick=()=>safe(adopt);
  $('annotationAssistReject').onclick=()=>safe(reject);
  $('annotationAssistResolve').onclick=()=>safe(resolve);
  $('annotationAssistNext').onclick=$('annotationAssistReviewLink').onclick=()=>safe(nextException);
  return {onContextChange,onSaved,refresh,renderFrame};
}
