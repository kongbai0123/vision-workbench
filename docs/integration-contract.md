# Vision Workbench internal contract

Independent Windows desktop app: PySide6 WebEngine shell, local Python HTTP service, one durable project store. No imports, launchers, symlinks or runtime dependencies on the three original project directories. Reuse their source as internal modules. No model training scope.

## Shared data

Project: `{id,name,revision,created_at,updated_at,classes:[string],assets:[Asset],stats:{total,pending,approved,rejected},exports:[]}`.
Asset: `{id,name,width,height,sha256,batch_id,split,source,revision,review_state,shape_count,url}`. Full asset detail adds `shapes`. Internal export snapshot adds absolute `image_path` and full shapes. `review_state` is `pending|approved|rejected`; `split` is empty or `train|val|test`. Editing resets approval to pending. Original image bytes are immutable and copied into the new project's image store.
Shape schema is the existing annotation studio native schema: `{id,type,label,hidden,x,y,width,height,points?,counts?,metadata?}`. Types rectangle, obb, polygon, linestrip, point, mask. Mask counts = full-image uncompressed COCO RLE, column-major. Stable IDs. Preserve source metadata. No lossy conversion during editing.

## HTTP API for web UI

GET `/api/projects` => `{projects:[{id,name,...stats}]}`
POST `/api/projects` `{name}` => full project (asset metadata only)
GET `/api/projects/{pid}` => project
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
GET `/api/jobs/{id}` => `{id,kind,state:'queued'|'running'|'succeeded'|'failed',message,progress:0..100|null,result?,error?}`
GET `/api/system` => app/version/capabilities/default export path.
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
POST `/api/dialog` `{kind:'images'|'folder'|'video'|'output'}` => `{paths:[...]}` native chooser (desktop only; browser UI supports path entry fallback). POST `/api/open-folder` `{project_id? ,export_id?}` opens app-owned output/project folder.

All requests with JSON body use Content-Type application/json and X-Workbench: 1. Responses are direct objects, errors `{error,message}` with non-2xx status. UI must show failures, await autosave before asset/project/stage switches, and disable repeated in-flight actions. Saved state remains authoritative. Poll jobs ~500ms, do not refresh canvas during editing. API only binds loopback.

## Pipeline module (agent owned workbench/pipeline.py)

`import_sources(paths: list[str|Path]) -> {records:[{path,name?,shapes:[],batch_id?,split?,source?:dict,review_state?:str}],issues:[{level,message,source?}]}`. Must preserve image pairing, mask holes and provenance; avoid auxiliary images in graph packages; reject protected pseudo-label working folders. Source paths read-only. The store consumes records by copying source image bytes.

`validate_project(snapshot:dict,format_key='native',tolerance=0) -> {valid:bool,errors:[],warnings:[],stats:{...},losses:[]}`.

`export_project(snapshot:dict,output_dir:str|Path,format_key='native',version='v1',tolerance=0,acknowledge_loss=False) -> {path,zip_path?,report,format,version}`. Only approved assets exported; pending/rejected counts reported. Never overwrite. Build validated staging then atomic publish. Preserve original image hashes; native lossless project interchange and COCO precise mask primary. Derived YOLO warns/requires acknowledgment for geometry loss. Stable class map, source batch splits avoid overlap.

## Capture / AI module (agent owned workbench/acquisition.py)

`CameraService(storage_root:Path)` with `devices()->list`, `start(index=0,width=1280,height=720,fps=30)->dict`, `stop()->dict`, `status()->dict`, `frame_jpeg()->bytes`, `snapshot()->{path,name,batch_id,source}`, `start_recording(project_id:str)->dict`, `stop_recording()->dict`, `close()`.
`capture_screen(storage_root:Path)->record`; `extract_video(path,storage_root,interval_seconds=1)->list[record]`.
`segment_image(image_path,engine='sam2',points=None,negative_points=None,box=None,label='object',model_dir=None)->{shape,diagnostics}`; true SAM2 using new app's installed deps/local model, informative errors for missing model; explicit GrabCut option. Never silently substitute GrabCut for SAM2. Lazy import Torch.

Root owns store/server/desktop/bootstrap/tests integration. UI agent owns web/. Pipeline agent owns workbench/pipeline.py and owned internal conversion modules and pipeline tests. Acquisition agent owns workbench/acquisition.py, copied segmentation packages and acquisition tests. Do not overwrite other agent files.
