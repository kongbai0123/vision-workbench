# Changelog

## 2.7.6 — 2026-09-15

- 修正 YOLO／RT-DETR 候選推論的 RGB 輸入契約，避免 numpy 影像被當成 BGR 再次翻轉通道。
- YOLO Seg 候選使用原尺寸 Mask，移除將含 letterbox 補邊遮罩直接拉伸到原圖的錯誤；輸出尺寸不符時明確拒絕，避免錯位標註進入審核。
- 補充信心分數、mAP 與原尺寸 Mask 覆蓋品質的區別；模型可正常訓練不代表每張圖片的輪廓已合格。

## 2.7.5 — 2026-09-15

- 智慧分割將 Train 類別覆蓋改為硬限制；不可行方案保留原因與分布預覽，但不能套用。固定版本建立與所有訓練引擎啟動前共用檢查，舊版本也不能繞過。
- YOLO26 Seg 加入預訓練微調／隨機初始化選項，保存初始權重来源與 SHA256；Ultralytics 梯度累積預設 1，可明確設定有效批次。
- 修正 Ultralytics 新版字典格式造成整輪 Loss 漏收，以及最終評估造成 51／50 假 Epoch 的問題。記錄真正成功的權重更新次數與 LR，AMP 跳步不誤計。
- 相同參數組 LR 合併至單張圖，不同曲線保留組別線型；模型勾選、固定座標及 Epoch 清單持續保留。
- 加入可追溯的時間區段流程驗證副本，保存區段間隔排除清單，明示同拍攝批次跨集合的限制，不改寫專案或原始固定版本。

## 2.7.4 — 2026-09-15

- SAM2 候選在原尺寸二值 Mask 產生後，會依單孔、總像素、相對面積及寬高門檻自動填補肉眼難辨的微小孔洞；背景提示點、大孔洞及分離物件保持不變。
- 內建編輯器儲存 Mask 時執行相同的保守檢查，將實際保存結果立即同步回畫布，並記錄修補孔洞、像素與座標供歷史追溯。
- SAM2 獨立分割器預設使用與資料相同的精確輪廓，並在診斷資料記錄自動修補數量，避免預覽平滑但保存 Mask 仍有孔洞的落差。
- 統一審核預檢與 YOLO Seg 實際轉換的孔洞拓樸判定，正確辨識與外部僅對角相連的 1 px 空點；D004 的 `dd802d3480.png / grasp` 已驗證可在 Run 副本修補 1 px 後完整轉換。
- 不會批次改寫既有專案或固定 DatasetVersion；既有微孔洞只在後續編輯存檔或 YOLO Run 相容副本中依規則處理。

## 2.7.3 — 2026-09-15

- 將審核相容檢查改為最多四路並行，並依圖片標註內容快取結果；實際 52 張資料首次掃描由約 2.5 秒降至 0.9 秒，未變更資料的再次顯示約 0.03 秒。
- 孔洞診斷只載入目標附近的小型裁切 JPEG，不再傳送整張高解析原圖或在瀏覽器逐像素重畫 Mask；加入明確的載入及失敗狀態。
- 多孔洞診斷以同一視窗的編號卡片、上一個／下一個與目前進度顯示；切換孔洞即更新精確裁切，不必關閉視窗再從清單進入。
- 「在內建編輯器精確定位」會直接把編輯畫布平移並放大至目前孔洞中心，方便立即使用筆刷或橡皮擦修正。
- 修補後以實際封閉背景區塊及像素 IoU 判定相容性，保留輪廓轉換造成的真實像素差異供後續檢查。

## 2.7.2 — 2026-09-15

- 將 YOLO Seg「完整資料相容檢查」移至資料審核頁，直接掃描目前專案的待審核與已核准標註，不必先建立固定資料版本。
- 進入資料審核，以及核准、退回或調整資料後會自動重新掃描；結果保留放大孔洞位置與跳回編輯器功能。
- 相容報告綁定專案 revision，標註變動後舊結果會立即失效，避免把過期清單當成目前狀態。
- 訓練頁保留背景安全複查：YOLO Seg 啟動時仍驗證固定 DatasetVersion，其他可保留孔洞的模型不受此格式檢查阻擋。

## 2.7.1 — 2026-09-15

- YOLO Seg 相容檢查會明確標示正在掃描的 DatasetVersion，並在固定資料版本早於目前專案時顯示 revision 差異與「建立最新資料版本並重新檢查」入口。
- 建立新 DatasetVersion 後會清除舊版相容檢查結果，避免舊報告繼續停用新版本的開始訓練按鈕。
- 微小孔洞的建議預設調整為單孔 16 px、單一實例合計 16 px、Mask 比例 0.01%、寬高 16 px；修補仍只發生在 Run 副本。
- 大孔洞與具語意的中空區域仍會阻擋 YOLO Seg，以免訓練資料在轉換時遺失幾何內容。

## 2.7.0 — 2026-09-14

- 新增 YOLO Seg 完整資料相容檢查，可在建立 Run 前掃描固定資料版本並列出受影響圖片、標註、孔洞像素數、座標與外框。
- 新增嚴格無損與微小封閉孔洞相容模式；預設只接受單孔 4 px、單一實例合計 16 px、Mask 比例 0.01%、寬高 4 px 以內的孔洞。
- 合格孔洞只修補於 Run 專屬 YOLO 資料副本，原始專案 Mask、審核狀態及既有 DatasetVersion 完全不變；轉換報告保存於 Run、評估及 ModelVersion。
- 超出門檻、細長裂縫或分離區塊仍阻擋訓練，不會靜默遺失幾何內容。
- 相容檢查結果可直接開啟像素化放大預覽，以紅框、十字及精確座標標示肉眼難以察覺的 1 px 孔洞。
- 圖表遇到缺少的 Epoch 指標時保留真實缺口，並直接列出缺值輪次，明示這是指標紀錄缺失而非訓練程序中斷。

## 2.6.2 — 2026-09-14

- 將程式更新完整移入「設定 → 更新與版本」，移除獨立的 Qt 更新視窗。
- 頂部設定按鈕、設定分類與程式更新卡片同步顯示紅點，清楚指出待處理更新的位置。
- 程式更新卡片顯示待套用檔案、阻擋原因及保存／驗證／重新啟動狀態。
- 單一按鈕會先重新檢查，若可更新便直接保存工作內容、驗證程式並自動重新啟動。
- 模型元件維持獨立管理，避免程式更新取代模型環境或中斷訓練。

## 2.6.1 — 2026-09-14

- 將「資料分割」提升為資料審核後、模型訓練前的第 05 個獨立流程階段，後續階段依序調整為 06 與 07。
- 新增分割摘要頁，集中呈現已核准圖片、獨立來源群組、目前比例、智慧分割狀態、阻擋項目與警示。
- 資料審核的下一步直接進入資料分割；未符合訓練條件時停用前往模型訓練，訓練頁可返回分割階段重新調整。
- 保留既有智慧分割預覽、群組鎖定、手動群組與建立新固定資料版本功能。
- 穩定 Windows CI 的訓練程序收尾測試，等待子程序關閉日誌後再清理隔離資料。

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
