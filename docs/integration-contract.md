# Vision Workbench internal contract

Independent Windows desktop app: PySide6 WebEngine shell, local Python HTTP service, one durable project store, and an optional isolated TorchVision runtime. No imports, launchers, symlinks or runtime dependencies on the original project directories. Reused behavior lives in internal modules. Model training consumes immutable Workbench DatasetVersions rather than exported exchange folders.

## Shared data

Project: `{id,name,revision,created_at,updated_at,classes:[string],assets:[Asset],stats:{total,pending,approved,rejected},exports:[]}`.
Asset: `{id,name,width,height,sha256,batch_id,split,source,revision,review_state,shape_count,url}`. Full asset detail adds `shapes`. Internal export snapshot adds absolute `image_path` and full shapes. `review_state` is `pending|approved|rejected`; `split` is empty or `train|val|test`. Editing resets approval to pending. Original image bytes are immutable and copied into the new project's image store.
Shape schema is the existing annotation studio native schema: `{id,type,label,hidden,x,y,width,height,points?,counts?,metadata?}`. Types rectangle, obb, polygon, linestrip, point, mask. Mask counts = full-image uncompressed COCO RLE, column-major. Stable IDs. Preserve source metadata. No lossy conversion during editing.

DatasetVersion: immutable `{dataset_version_id,project_id,project_revision,classes,class_mapping,assets:[{asset_id,image_file,sha256,annotation_revision,annotation_sha256,split,batch_id,shapes,objects}]}`. Each image is hash checked and hard-linked to the immutable project image when supported, otherwise copied, before atomic publication.

TrainingRun: persistent `{run_id,project_id,dataset_version_id,model_version_id,engine,config,status,progress,metrics?,evaluation?}`. Active states are `queued|preparing|running|stopping`; terminal states are `completed|failed|stopped`. A worker writes metrics, logs, evaluation and artifact manifest below the owning project at `runs/{run_id}`.

ModelVersion: immutable model metadata and artifacts below the owning project at `models/{model_id}`. A ModelExport is an atomic ZIP at `model-exports/{export_id}` containing model artifacts, run/evaluation/metric lineage and SHA-256 metadata, without dataset images. PredictionCandidate keeps model/run lineage, the source asset revision and native shapes. Accepting a candidate uses the normal revision check and resets the image to `pending` only when shapes changed. SQLite indexes these filesystem artifacts through foreign-keyed DatasetVersion, TrainingRun, ModelVersion, ModelExport and PredictionCandidate tables; stored paths are project-relative.

## HTTP API for web UI

GET `/api/projects` => `{projects:[{id,name,...stats}]}`
POST `/api/projects` `{name}` => full project (asset metadata only)
GET `/api/projects/{pid}` => project
GET `/api/projects/{pid}/training` => readiness, DatasetVersions, Runs, ModelVersions, ModelExports, PredictionCandidates and engine capabilities
GET `/api/projects/{pid}/training-runs/{rid}/metrics` => persistent run and metric rows
PATCH `/api/projects/{pid}` `{name?,classes?}` => project
POST `/api/projects/{pid}/import` `{paths:[absolute paths]}` => job
POST `/api/projects/{pid}/merge` `{project_ids:[ids]}` => job
GET `/api/projects/{pid}/assets/{aid}` => full asset
GET `/api/projects/{pid}/assets/{aid}/image` => original bytes
PUT `/api/projects/{pid}/assets/{aid}` `{shapes,revision}` => full asset (409 stale revision)
POST `/api/projects/{pid}/review` `{asset_ids:[ids],state:'approved'|'pending'|'rejected'}` => project
POST `/api/projects/{pid}/validate` `{format:'native'|'coco'|'yolo_detection'|'yolo_segmentation'|'labelme',tolerance:0}` => job
POST `/api/projects/{pid}/export` `{format,version,output_dir?,tolerance:0,acknowledge_loss:false}` => job
POST `/api/projects/{pid}/ai` `{asset_id,revision,engine:'sam2'|'grabcut',label,points:[[x,y]],negative_points:[[x,y]],box:[x1,y1,x2,y2]}` => job. AI returns candidate shape in result; UI accepts via normal PUT (not automatically approved).
POST `/api/projects/{pid}/dataset-versions` => create immutable approved training snapshot
POST `/api/projects/{pid}/training-runs` `{dataset_version_id,config:{engine,epochs,seed,device}}` => start isolated worker
POST `/api/projects/{pid}/training-runs/{rid}/stop` => request safe stop
POST `/api/projects/{pid}/model-exports` `{model_version_id}` => atomic portable model-bundle job
POST `/api/projects/{pid}/predictions` `{model_version_id,asset_ids?}` => prediction job
POST `/api/predictions/{candidate_id}/accept` `{asset_ids?}` => append candidates through revision-checked project store
GET `/api/jobs/{id}` => `{id,kind,state:'queued'|'running'|'succeeded'|'failed',message,progress:0..100|null,result?,error?}`
GET `/api/system` => app/version/capabilities/default export path.
GET `/api/model-catalog?refresh=1` => all ready/unavailable/planned model entries plus component state; entries remain visible when unavailable.
POST `/api/model-components/{component_id}/install` => bounded background install job for components whose Workbench adapter is ready; currently `torchvision` and `ultralytics`.
GET `/api/camera/devices` => `{devices:[{index,name}]}`
POST `/api/camera/start` `{index,width,height,fps}` => status
POST `/api/camera/stop` => status
GET `/api/camera/status` => status
GET `/api/camera/frame` => JPEG (cache bust query supported)
POST `/api/projects/{pid}/capture` `{label?,batch_id?}` => asset
POST `/api/projects/{pid}/screen` => asset
POST `/api/camera/record/start` `{project_id}` => status
POST `/api/camera/record/stop` => status
POST `/api/projects/{pid}/video` `{path,interval_seconds:1}` => job extracting frames.
POST `/api/dialog` `{kind:'images'|'folder'|'video'|'output'}` => `{paths:[...]}` native chooser (desktop only; browser UI supports path entry fallback). POST `/api/open-folder` `{project_id?,export_id?,model_export_id?}` opens an app-owned project, dataset export, or model export folder.

All requests with JSON body use Content-Type application/json and X-Workbench: 1. Responses are direct objects, errors `{error,message}` with non-2xx status. UI must show failures, await autosave before asset/project/stage switches, and disable repeated in-flight actions. Saved state remains authoritative. Poll jobs ~500ms, do not refresh canvas during editing. API only binds loopback.

## Training runtime

The desktop runtime remains in `.venv`; the optional TorchVision runtime lives in `.venv-training` and is installed with `bootstrap.ps1 -Training` or the settings model center. RT-DETR and YOLO26 Detect/Seg use `.venv-models/ultralytics`, installed only by the settings model center. `VISION_WORKBENCH_TRAINING_PYTHON` may select another versioned TorchVision runtime. Every training and inference subprocess uses the runtime registered for its model component. The built-in `pixel_prototype_v1` engine remains available without Torch; optional engines are advertised as usable only after their runtime probe succeeds.

Mask R-CNN reads native full-image RLE. Faster R-CNN, RT-DETR, and YOLO26 Detect derive tight boxes from valid annotations; DeepLabV3 derives semantic class maps from native area annotations. MobileNet V3, EfficientNet-B0, and ResNet18 derive one image class only when all shapes on an approved image have the same label; they do not create annotation candidates. YOLO26 Seg converts only single-component, hole-free area shapes and rejects lossy conversions before a worker starts. Each engine writes `checkpoint.pt`, JSONL metrics, task-specific validation/test metrics and an artifact manifest. Ultralytics native `results.csv` is canonicalized into non-duplicated chart fields on read. Detection candidates use rectangles; instance and semantic candidates use masks. All accepted candidates return through revision checks and pending review. The UI may close while a worker continues; persisted run files remain the source of truth after reopening.

## Pipeline module (agent owned workbench/pipeline.py)

`import_sources(paths: list[str|Path]) -> {records:[{path,name?,shapes:[],batch_id?,split?,source?:dict,review_state?:str}],issues:[{level,message,source?}]}`. Must preserve image pairing, mask holes and provenance; avoid auxiliary images in graph packages; reject protected pseudo-label working folders. Source paths read-only. The store consumes records by copying source image bytes.

`validate_project(snapshot:dict,format_key='native',tolerance=0) -> {valid:bool,errors:[],warnings:[],stats:{...},losses:[]}`.

`export_project(snapshot:dict,output_dir:str|Path,format_key='native',version='v1',tolerance=0,acknowledge_loss=False) -> {path,zip_path?,report,format,version}`. Only approved assets exported; pending/rejected counts reported. Never overwrite. Build validated staging then atomic publish. Preserve original image hashes; native lossless project interchange and COCO precise mask primary. Derived YOLO warns/requires acknowledgment for geometry loss. Stable class map, source batch splits avoid overlap.

## Capture / AI module (agent owned workbench/acquisition.py)

`CameraService(storage_root:Path)` with `devices()->list`, `start(index=0,width=1280,height=720,fps=30)->dict`, `stop()->dict`, `status()->dict`, `frame_jpeg()->bytes`, `snapshot()->{path,name,batch_id,source}`, `start_recording(project_id:str)->dict`, `stop_recording()->dict`, `close()`.
`capture_screen(storage_root:Path)->record`; `extract_video(path,storage_root,interval_seconds=1)->list[record]`.
`segment_image(image_path,engine='sam2',points=None,negative_points=None,box=None,label='object',model_dir=None)->{shape,diagnostics}`; true SAM2 using new app's installed deps/local model, informative errors for missing model; explicit GrabCut option. Never silently substitute GrabCut for SAM2. Lazy import Torch.

Root owns store/server/desktop/bootstrap/tests integration. UI agent owns web/. Pipeline agent owns workbench/pipeline.py and owned internal conversion modules and pipeline tests. Acquisition agent owns workbench/acquisition.py, copied segmentation packages and acquisition tests. Do not overwrite other agent files.
