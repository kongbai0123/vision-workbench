"""Mutation method contract. Handlers may not silently accept another verb."""
import re

MUTATIONS = [
    (r'/api/projects/[^/]+/maintenance', {'POST'}),
    (r'/api/projects', {'POST'}),
    (r'/api/projects/[^/]+', {'PATCH', 'DELETE'}),
    (r'/api/projects/[^/]+/assets/[^/]+', {'PUT', 'DELETE'}),
    (r'/api/projects/[^/]+/assets', {'DELETE'}),
    (r'/api/projects/[^/]+/assets/[^/]+/restore', {'POST'}),
    (r'/api/projects/[^/]+/training-runs/[^/]+/stop', {'POST'}),
    (r'/api/projects/[^/]+/(classes|import-preview|import-confirm|review-trash-list|review-trash|review-restore|review-quality|import|review|independence-review|assign|auto-split|split-preview|split-info|split-apply|merge|validate|export|dataset-versions|workflow-draft|training-preflight|training-runs|training-compatibility|review-compatibility|model-exports|model-import|predictions|model-trials|model-comparisons|ai|capture|screen|video)', {'POST'}),
    (r'/api/camera/profiles', {'GET', 'POST'}),
    (r'/api/(dialog|open-folder|camera/(start|stop|controls|processing|record/start|record/stop)|cvat/(setup|reboot|reboot/cancel|launch|import))', {'POST'}),
    (r'/api/model-components/[^/]+/install', {'POST'}),
    (r'/api/model-uploads', {'POST'}),
    (r'/api/jobs/[^/]+/(pause|resume|cancel)', {'POST'}),
    (r'/api/predictions/[^/]+/accept', {'POST'}),
]

def validate_method(path, method):
    for pattern, allowed in MUTATIONS:
        if re.fullmatch(pattern, path):
            if method not in allowed:
                raise MethodNotAllowed('此 API 不支援該 HTTP 方法')
            return
    raise FileNotFoundError('找不到 API')

class MethodNotAllowed(ValueError):
    pass
