"""Project mutation dispatch; method and request contracts live in routes/contracts."""
def action_maintenance(self, pid, payload, action):
    from .maintenance import orphan_images
    if type(payload.get('quarantine', False)) is not bool:
        raise ValueError('quarantine 必須是布林值')
    return self.json(orphan_images(self.app.store, pid, quarantine=payload.get('quarantine', False)))


def action_import_preview(self, pid, payload, action):
    return self.json(self.app.jobs.submit('import-preview', lambda progress: self.app.review_workflow.preview(pid, payload.get('paths'), progress=progress)))


def action_import_confirm(self, pid, payload, action):
    return self.json(self.app.jobs.submit('import', lambda progress: self.app.review_workflow.commit_import(pid, payload.get('token'), payload.get('selected'), progress=progress)))


def action_review_trash_list(self, pid, payload, action):
    return self.json(self.app.review_workflow.list_trash(pid))


def action_review_trash(self, pid, payload, action):
    return self.json(self.app.review_workflow.trash(pid, payload.get('asset_ids'), payload.get('revisions'), restore=action == 'review-restore'))


def action_review_quality(self, pid, payload, action):
    return self.json(self.app.jobs.submit('review-quality', lambda progress: self.app.review_workflow.quality(pid)))


def action_import(self, pid, payload, action):
    return self.json(self.app.import_paths(pid,payload.get("paths")))


def action_review(self, pid, payload, action):
    return self.json(self.app.store.review(pid,payload.get("asset_ids"),payload.get("state"),payload.get("revisions"),payload.get('reason', ''),payload.get('note', ''),delta_base=payload.get('delta_base')))


def action_assign(self, pid, payload, action):
    return self.json(self.app.store.assign(pid,payload.get("asset_ids"),batch_id=payload.get("batch_id"),split=payload.get("split"),delta_base=payload.get('delta_base')))


def action_auto_split(self, pid, payload, action):
    return self.json(self.app.store.auto_split(pid,payload.get("ratios") or {"train":70,"val":20,"test":10}))


def action_split_preview(self, pid, payload, action):
    return self.json(self.app.store.preview_split(pid,payload.get("options")))


def action_split_info(self, pid, payload, action):
    return self.json(self.app.store.split_info(pid))


def action_split_apply(self, pid, payload, action):
    return self.json(self.app.store.apply_split(pid,payload.get("options"),payload.get("revision"),payload.get("fingerprint")))


def action_merge(self, pid, payload, action):
    return self.json(self.app.jobs.submit("merge",lambda progress:self.app.store.merge(pid,payload.get("project_ids"))))


def action_validate(self, pid, payload, action):
    return self.json(self.app.pipeline_job(pid,action,payload))


def action_dataset_versions(self, pid, payload, action):
    return self.json(self.app.training.create_dataset_version(pid, payload.get("augmentation")))


def action_training_runs(self, pid, payload, action):
    return self.json(self.app.training.start_run(pid,payload.get("dataset_version_id"),payload.get("config") or {},payload.get('request_id')))


def action_training_compatibility(self, pid, payload, action):
    return self.json(self.app.training.yolo_compatibility(pid,payload.get("dataset_version_id"),payload.get("config") or {}))


def action_review_compatibility(self, pid, payload, action):
    return self.json(self.app.training.review_yolo_compatibility(pid,payload.get("config") or {}))


def action_model_import(self, pid, payload, action):
    upload_token = payload.get('upload_token')
    if upload_token:
        checkpoint, filename = self.app.model_uploads.claim(upload_token)
        def import_uploaded(progress):
            try:
                return self.app.training.import_model(
                    pid, checkpoint, payload.get('name', ''), payload.get('trusted', False), progress,
                    source_filename=filename)
            finally:
                self.app.model_uploads.discard(checkpoint)
        try:
            job = self.app.jobs.submit('model-import', import_uploaded)
        except Exception:
            self.app.model_uploads.discard(checkpoint)
            raise
        return self.json(job)
    return self.json(self.app.jobs.submit('model-import', lambda progress:
        self.app.training.import_model(pid, payload.get('path'), payload.get('name', ''), payload.get('trusted', False), progress)))


def action_model_exports(self, pid, payload, action):
    model_id = payload.get("model_version_id")
    return self.json(self.app.jobs.submit("model-export", lambda progress:
        self.app.training.export_model(pid, model_id, progress)))


def action_predictions(self, pid, payload, action):
    model_id = payload.get("model_version_id")
    return self.json(self.app.jobs.submit("prediction", lambda progress: (
        progress("使用模型產生候選標註", 20),
        self.app.training.create_predictions(pid, model_id, payload.get("asset_ids"), progress)
    )[1]))

def action_model_trials(self,pid,payload,action):
    model_id=payload.get('model_version_id');paths=payload.get('paths')
    return self.json(self.app.jobs.submit('model-trial',lambda progress:self.app.training.create_model_trial(pid,model_id,paths,progress)))

def action_model_comparisons(self,pid,payload,action):
    return self.json(self.app.jobs.submit('model-comparison',lambda progress:self.app.training.create_model_comparison(
        pid,payload.get('model_version_id'),payload.get('split','test'),progress)))


def action_ai(self, pid, payload, action):
    return self.json(self.app.ai_job(pid,payload))


def action_capture(self, pid, payload, action):
    from .acquisition import capture_screen
    self.app.store.get_project(pid, include_assets=False)
    record = self.app.camera.snapshot() if action == "capture" else capture_screen(self.app.incoming)
    if action == "capture" and payload.get("target_shape") is not None:
        self.app.store.require_class(pid,payload["target_shape"].get("label"))
        record["shapes"] = [payload["target_shape"]]
    if payload.get("batch_id"):
        record["batch_id"] = payload["batch_id"]
    result = self.app.store.add_assets(pid,[record])
    if result["asset_ids"]:
        return self.json(self.app.store.get_asset(pid,result["asset_ids"][0]))
    return self.json(dict(result,message="相同原圖已存在，未重複加入"))


def action_video(self, pid, payload, action):
    self.app.store.get_project(pid, include_assets=False)
    def run(progress):
        from .acquisition import extract_video
        progress("從影片擷取影格")
        records = extract_video(payload.get("path"),self.app.incoming,payload.get("interval_seconds",1),
                                progress=lambda percent:progress("從影片擷取影格",round(float(percent)*.8)))
        if payload.get("target_shape") is not None:
            self.app.store.require_class(pid,payload["target_shape"].get("label"))
            for record in records:
                record["shapes"] = [payload["target_shape"]]
        return self.app.store.add_assets(pid,records,progress=lambda n,total:progress(f"保存影格 {n} / {total}",round(n/total*95)))
    return self.json(self.app.jobs.submit("video",run))


PROJECT_ACTIONS = {
    'maintenance': action_maintenance,
    'import-preview': action_import_preview,
    'import-confirm': action_import_confirm,
    'review-trash-list': action_review_trash_list,
    'review-trash': action_review_trash,
    'review-restore': action_review_trash,
    'review-quality': action_review_quality,
    'import': action_import,
    'review': action_review,
    'assign': action_assign,
    'auto-split': action_auto_split,
    'split-preview': action_split_preview,
    'split-info': action_split_info,
    'split-apply': action_split_apply,
    'merge': action_merge,
    'validate': action_validate,
    'export': action_validate,
    'dataset-versions': action_dataset_versions,
    'training-runs': action_training_runs,
    'training-compatibility': action_training_compatibility,
    'review-compatibility': action_review_compatibility,
    'model-import': action_model_import,
    'model-exports': action_model_exports,
    'predictions': action_predictions,
    'model-trials': action_model_trials,
    'model-comparisons': action_model_comparisons,
    'ai': action_ai,
    'capture': action_capture,
    'screen': action_capture,
    'video': action_video,
}


def dispatch_project_action(handler, pid, action, payload):
    callback = PROJECT_ACTIONS.get(action)
    if callback is None:
        raise FileNotFoundError("找不到 API")
    return callback(handler, pid, payload, action)
