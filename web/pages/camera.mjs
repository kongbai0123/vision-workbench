export function createCameraPage(context) {
const {$,state,decodeMask,shapeNames,toast,brush,encodeMask,api,setSourceInspector,updateAcquisitionControls,renderAssetList,renderAcquisitionAssets,flushAllEdits}=context;
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
return {cameraTargetDimensions, cameraTargetShape, cameraTargetPayload, targetPoint, targetSvgShape, renderTargetMask, renderCameraTarget, updateCameraTargetTool, commitCameraTarget, clearCameraTarget, updateCameraTargetEditingSurface, installCameraTargetEvents, cameraFlags, loadCameraModes, selectedCameraModes, updateCameraMode, cameraConfiguration, receiveCameraStatus, cameraStatus, cameraCommand, stopPreview, updatePreviewLayout, setCameraPreviewExpanded, updateCameraControls, startPreview, stopAutoCapture, runAutoCapture, startAutoCapture};
}
