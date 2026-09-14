# Changelog

## 2.2.0 — 2026-09-14

- Renamed the top-level “設定／更新” entry to “設定”; program and component updates remain organized inside the settings center.
- Added live per-epoch training plots with exact point values, a fixed full-run X range, and stable 0–1 validation axes instead of a sliding viewport.
- Added portable model bundle export with model/checkpoint files, run configuration, metrics, evaluation, lineage, and SHA-256 verification metadata; training images are excluded.
- Added model export history and direct access to the generated local folder from each ModelVersion.

## 2.1.0 — 2026-09-14

- Added a unified settings sidebar with general preferences, model components, compute/storage, editor status, updates, and diagnostics.
- Added a discoverable model catalog that keeps unavailable models visible and explains integration, runtime, annotation, metric, and license requirements without downloading on selection.
- Added bounded on-demand installation and runtime revalidation for the shared TorchVision component.
- Added Faster R-CNN MobileNet V3 FPN, MobileNet V3 320 FPN, and ResNet50 FPN V2 detection adapters with immutable dataset input, box evaluation, checkpoints, isolated inference, and review candidates.
- Added DeepLabV3 MobileNet V3 and ResNet50 semantic-segmentation adapters with native-mask class maps, mIoU/Dice evaluation, checkpoints, isolated inference, and review candidates.
- Added visible roadmap entries for image classification, EfficientAD, PatchCore, RT-DETR, and YOLO26 Seg without claiming unavailable adapters are ready.

## 2.0.0 — 2026-09-14

- Added an in-app Labelme editor and shared revision-safe synchronization across the built-in editor, Labelme, and CVAT.
- Preserved native pixel masks and object bounds across editor round trips, including mask holes.
- Added immutable DatasetVersions sourced directly from approved Workbench project data.
- Added persistent background training Runs, progress, safe stop requests, logs, evaluation, ModelVersions, and artifact manifests.
- Added an isolated TorchVision Mask R-CNN runtime with native RLE loading, CUDA/CPU execution, checkpoints, mask IoU evaluation, and isolated inference.
- Added a lightweight built-in segmentation baseline for environments without PyTorch.
- Added model prediction candidates that return to the same annotation and review workflow with revision conflict protection.
- Moved dataset export to “Backup & Exchange” and expanded the primary workflow through training and model evaluation.
- Added CVAT startup, Windows update/reboot guidance, deterministic class-count splitting, and full API/desktop regression coverage.

## 1.0.0

- Initial Vision Workbench release for acquisition, annotation, review, validation, and dataset export.
