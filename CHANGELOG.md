# Changelog

## 2.6.0 — 2026-09-14

- 新增智慧資料分割管理：來源批次／影片／相同圖片保持群組完整，兼顧類別分布與比例；支援預覽、手動群組、指定集合及保留 Test。
- 審核及訓練頁提供永久入口，跨集合批次警示可直接開啟；以修訂檢查拒絕過期預覽，固定資料版本保留完整分割方案。
- 影片取樣記錄來源影片 SHA-256，避免同一影片換路徑後被當成獨立來源。
- 新增固定、Cosine、線性與暖身 LR 設定／預覽；TorchVision 支援依 Validation 停滯降率。Ultralytics 接原生排程，監控與 Epoch 明細顯示實際 LR。
- TorchVision 保存 optimizer／scheduler 狀態；補強 Windows 並行讀取訓練 JSON 時的短暫檔案鎖重試。
- 增加智慧分割與 LR 單元測試、桌面流程測試，以及四種 TorchVision 引擎的實際 CPU 排程驗證腳本。

## 2.5.0 — 2026-09-14

- Replaced per-metric checkboxes with a single completed-model comparison picker. Selecting model versions controls every chart, summary, Epoch row, and evaluation panel together; recorded metrics render automatically.
- Added model-specific training parameters for image size, batch size, learning rate, weight decay, and optimizer, plus meaningful threshold limits for the deterministic CPU baseline.
- Validated parameters before Run creation and forwarded the effective values to each adapter; changing models or refreshing preserves parameter drafts.
- Honored classification image sizes above 512 and supported singleton DeepLab batches without dropping training images.
- Placed class evaluation and readable execution settings side by side on desktop, with stacked panels on smaller windows.
- Limited the Epoch list to ten visible data rows with a sticky header and scrolling; polling preserves the current scroll position.

## 2.4.0 — 2026-09-14

- Aligned the dataset, training configuration, and execution panels with matching heights and responsive field layouts.
- Expanded training results to full width and moved Run history into a keyboard-accessible sidebar dialog.
- Added two-column metric plots with on-demand metric selection, up to four same-dataset Run overlays, persistent Run colors/line styles, and synchronized show/hide controls.
- Added exact shared-Epoch inspection, missing-value gaps, stable chart domains, per-Epoch tables, class-level evaluation, and classification confusion matrices.
- Prevented overlays of incompatible loss/IoU definitions and clearly identified Test fallback during training evaluation.
- Fixed Ultralytics mAP50 being confused with mAP50–95; absent metrics are omitted instead of fabricated as zero, and final evaluation retains the actual data split.
- Added chart-model regression tests and an isolated desktop workflow covering comparison, live polling, missing values, metric selection, and panel alignment.

## 2.3.0 — 2026-09-14

- Added trainable MobileNet V3, EfficientNet-B0, and ResNet18 image-classification adapters with immutable input validation, per-epoch Accuracy/Macro F1/Macro Recall, evaluation, checkpoints, and model bundles.
- Classification labels are derived only from approved images with exactly one distinct annotation class; classification never creates a full-image bounding box candidate.
- Added RT-DETR ResNet50 and YOLO26n/s Seg adapters with an isolated, on-demand Ultralytics environment, deterministic Workbench-to-YOLO data preparation, per-epoch metrics, evaluation, checkpoints, and review candidates.
- Added strict YOLO Seg geometry checks that block hole-bearing or multi-component masks instead of silently discarding geometry.
- Added per-model runtime selection so TorchVision and Ultralytics workers execute in separate environments; selecting a catalog item still performs no installation or weight download.
- EfficientAD and PatchCore remain visible as Anomalib integration work in progress.

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
