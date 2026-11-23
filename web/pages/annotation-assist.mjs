const verdictNames = {accepted:'建議採用', rejected:'建議拒絕', unresolved:'需覆核'};
const runNames = {queued:'等待', running:'執行中', complete:'完成', failed:'失敗', cancelled:'已停止',
  stopped:'已停止', interrupted:'已中斷'};
const voteNames = {A:'符合類別與框界', B:'框界或併框問題', C:'其他物件或類別', D:'證據不足'};
const terminalJobStates = new Set(['succeeded','failed','cancelled']);
const candidatePriority = {unresolved:0, accepted:1, rejected:2};

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
    return shape.type!=='mask' && overlaps(shape,candidate.shape)>=.85;
  });
}

function modelScore(value) {
  return typeof value==='number' && Number.isFinite(value) ? value.toFixed(3) : '—';
}

export function createAnnotationAssistPage({$, state, editor, api, button, element, safe, toast, filteredAssets,
  selectAsset, switchStage, flushAllEdits, date}) {
  const panel = $('annotationAssistPanel');
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
  const draftDecisions = new Map();

  const endpoint = pid => `/api/projects/${encodeURIComponent(pid)}/annotation-assist`;
  const frameForAsset = () => run?.frames?.find(frame => frame.asset_id===state.asset?.id);
  const currentCandidate = () => frameForAsset()?.candidates?.find(candidate => candidate.id===selectedId);
  const decisionsFor = (id, frame) => draftDecisions.get(`${id}/${frame.asset_id}`)
    ?? new Map((frame.review?.decisions||[]).map(item=>[item.candidate_id,item.reason]));
  const assetFor = id => state.project?.assets.find(asset => asset.id===id);
  const revisionFor = frame => state.asset?.id===frame.asset_id ? state.asset.revision : assetFor(frame.asset_id)?.revision;
  const isResolved = frame => frame.review?.status==='resolved';
  const pendingFrames = () => (run?.frames || []).filter(frame => frame.available!==false && frame.exceptions?.length && assetFor(frame.asset_id) && !isResolved(frame));

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
    $('annotationAssistStatus').textContent=job?.message||statusText||run?.error||run?.message||
      (run ? `批次${runNames[run.status]||run.status}。` : '尚無執行紀錄。');
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
    $('annotationAssistResult').hidden=!runs.length;
  }

  function renderReviewLink() {
    const count=pendingFrames().length;
    $('annotationAssistReviewEntry').hidden=!count;
    $('annotationAssistReviewCount').textContent=`目前 AI 批次有 ${count} 張影像待覆核`;
  }

  function renderEvidence(frame, candidate) {
    const root=$('annotationAssistEvidence');
    root.replaceChildren();
    editor.setAssistPreview(panel.open && state.stage==='annotate' ? candidate?.shape ?? null : null);
    const adopt=$('annotationAssistAdopt');
    const reject=$('annotationAssistReject');
    adopt.disabled=true;
    reject.disabled=true;
    if(!candidate || !state.asset)return;
    const present=alreadyOnCanvas(state.asset.shapes||[],run.id,candidate);
    const previouslyRejected=decisionsFor(run.id,frame).has(candidate.id);
    const labelExists=state.project.classes.includes(candidate.label);
    adopt.disabled=!candidate.shape||present||!labelExists||reviewing||frame.available===false;
    const adopted=(state.asset.shapes||[]).some(shape=>origin(shape)?.run_id===run.id && origin(shape)?.candidate_id===candidate.id);
    reject.disabled=adopted||previouslyRejected||reviewing||frame.available===false;
    if(candidate.reason)root.append(element('div',candidate.reason));
    if(Array.isArray(candidate.votes) && candidate.votes.length===2){
      const choices=candidate.votes.map(vote=>voteNames[vote?.winner]||'未定');
      root.append(element('div',`Qwen 兩次判讀 · ${choices.join(' ／ ')}`));
    }
    if(present)root.append(element('div','畫布已有對應標註。可直接選取並編輯現有框。'));
    if(!labelExists)root.append(element('div','此候選的類別已變更，請依目前類別手動覆核。'));
    if(previouslyRejected)root.append(element('div','已標記為不採用，確認整張後記錄。'));
    if(candidate.shape)root.append(element('div','黃色虛線為候選原框。'));
  }

  function renderFrame() {
    renderReviewLink();
    const frame=frameForAsset();
    const summary=run?.summary;
    const exceptions=$('annotationAssistExceptions');
    const list=$('annotationAssistCandidates');
    const previousAsset=selectedAssetId;
    selectedAssetId=frame?.asset_id??null;
    const previousScroll=previousAsset===selectedAssetId ? list.scrollTop : 0;
    if(previousAsset!==selectedAssetId)selectedId=null;
    exceptions.replaceChildren();
    list.replaceChildren();
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
    if(!candidates.some(candidate=>candidate.id===selectedId)){
      selectedId=candidates[0]?.id??null;
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
    for(const candidate of candidates){
      const row=button('','annotation-assist-candidate',()=>{
        selectedId=candidate.id;
        for(const item of list.children)item.setAttribute('aria-pressed',String(item===row));
        renderEvidence(frame,candidate);
      });
      row.dataset.assistCandidateId=candidate.id;
      row.dataset.assistLabel=candidate.label;
      row.dataset.assistVerdict=candidate.verdict;
      row.setAttribute('aria-pressed',String(candidate.id===selectedId));
      const manuallyRejected=decisionsFor(run.id,frame).has(candidate.id);
      const present=candidate.shape && alreadyOnCanvas(state.asset.shapes||[],run.id,candidate);
      const verdict=verdictNames[candidate.verdict]||verdictNames.unresolved;
      const decision=candidate.reason_code==='low_score_proposal' ? '低於門檻，未送 Qwen'
        : candidate.votes?.length ? `Qwen ${verdict}` : verdict;
      row.append(element('b',candidate.label),element('small',`MM ${modelScore(candidate.score)} · ${decision}`));
      if(manuallyRejected||present)row.append(element('small',manuallyRejected?'已標記不採用':'畫布已有標註'));
      list.append(row);
    }
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
    if(run?.id!==result.id)selectedId=null;
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

  async function start() {
    if(!projectId||jobs.has(projectId))return;
    const pid=projectId;
    const classes=selectedClasses();
    const imageIds=scopeAssets().map(asset=>asset.id);
    if(!classes.length||!imageIds.length)throw Error('請選取影像及至少一個類別。');
    if(classes.some(item=>!item.prompt))throw Error('已選取的類別都需要 MM 提示詞。');
    const active={id:null,progress:0,message:'準備批次…'};
    jobs.set(pid,active);
    renderControls();
    try{
      await flushAllEdits();
      if(projectId!==pid)return;
      let job=await api(endpoint(pid),'POST',{asset_ids:imageIds,classes});
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
      if(projectId===pid)renderControls();
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
    const shape=editor.add({...candidate.shape,metadata:{...candidate.shape.metadata,annotation_assist:{
      run_id:run.id,candidate_id:candidate.id,generated_label:candidate.label,score:candidate.score,
    }}});
    if(!shape)return;
    const frame=frameForAsset();
    const decisions=decisionsFor(run.id,frame);
    if(decisions.delete(candidate.id))draftDecisions.set(`${run.id}/${frame.asset_id}`,decisions);
    editor.setTool('select');
    renderFrame();
    toast('候選已加入標註，可在畫布修框。');
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
    if($('propertiesSidebar').hidden)$('toggleProperties').click();
    panel.open=true;
    panel.scrollIntoView({block:'nearest'});
    renderFrame();
  }

  $('openAnnotationAssist').onclick=()=>{openPanel();void safe(()=>refresh());};
  panel.ontoggle=()=>{renderFrame();if(panel.open)void safe(()=>refresh());};
  $('annotationAssistScope').onchange=renderControls;
  $('annotationAssistStart').onclick=()=>safe(start);
  $('annotationAssistStop').onclick=()=>safe(stop);
  $('annotationAssistRuns').onchange=()=>safe(()=>loadRun(projectId,$('annotationAssistRuns').value));
  $('annotationAssistAdopt').onclick=()=>safe(adopt);
  $('annotationAssistReject').onclick=()=>safe(reject);
  $('annotationAssistResolve').onclick=()=>safe(resolve);
  $('annotationAssistNext').onclick=$('annotationAssistReviewLink').onclick=()=>safe(nextException);
  return {onContextChange,onSaved,refresh,renderFrame};
}
