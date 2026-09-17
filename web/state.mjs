// Shared session state. Page modules receive this instance explicitly.
export function mergeProjectDelta(current, response) {
  if (!response?.delta) return response;
  if (current?.id !== response.id || current.revision !== response.delta.base_revision) return null;
  const changed = new Set(response.delta.changed_ids);
  const updates = new Map(response.assets.map(asset => [asset.id, asset]));
  const assets = current.assets.flatMap(asset => {
    if (!changed.has(asset.id)) return [asset];
    const updated = updates.get(asset.id);updates.delete(asset.id);
    return updated ? [updated] : [];
  });
  assets.push(...updates.values());
  const {delta, ...project} = response;
  return {...project, assets};
}
export const state = {projects:[],project:null,asset:null,stage:'library',acquireSource:'camera',busy:false,transitioning:false,
  system:null,editorMode:'builtin',cvatPoll:null,cvatElapsedTimer:null,cvatPollStartedAt:null,cvatPollBaseText:'',cvatPollBusy:false,cvatWasOpened:false,nativeBridge:null,reviewSelection:new Set(),acquireSelection:new Set(),reviewPage:0,reviewGeneration:0,previewRunning:false,previewTimer:null,camera:false,recording:false,cameraDetails:null,cameraPoll:0,autoCapture:null,cameraTarget:null,cameraTargetDraft:[],cameraTargetGesture:null,
  splitTab:'train',splitPage:0,validationResult:null,validationTab:null,validationPage:0,releaseDrawerFocus:null,
  training:null,trainingTimer:null,trainingMetrics:null,selectedRun:null,selectedModel:null,modelCatalog:null,selectedCatalogModel:null,yoloCompatibility:null,yoloCompatibilitySignature:'',reviewYoloCompatibility:null,reviewYoloCompatibilityLoading:false,augmentationPreset:'light',augmentationExpansion:0,trainingConfigTab:'basic',
  settingsPage:'general',settingsFocus:null,settingsJob:null,desktopUpdate:null};
