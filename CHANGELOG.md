# Changelog

## 2.17.0 — 2026-09-18

- 評估與模型頁左右面板底部對齊；空模型狀態提供訓練與外部匯入提示。
- 新增外部 Ultralytics YOLO 偵測／分割與 RT-DETR `.pt` 匯入，複製權重、驗證類別及 CPU 推論後建立專案模型版本。
- 外部模型保留來源與 SHA-256，可試跑、產生預標註與匯出封裝；清楚標示尚未評估，不虛構訓練紀錄與評估指標。
- 遷移 0005 保留既有模型關聯，允許外部模型不帶本機 Run／資料版本；升級前自動備份資料庫。
- 釐清「備份與外部交換」是圖片／標註資料交換，新增匯入、失敗清理、資料遷移及桌面流程驗證。

## 2.16.0 — 2026-09-18

- 修正 DirectShow 設定 FPS 時將 MJPG 重設為 YUY2 的問題，改為最後套用像素格式，再回報協商結果。
- 擴充右側相機設定：輸出格式與解析度、FPS 連動，支援套用並重新啟動；錄影與自動擷取期間鎖定需要重啟的設定。
- 依 Windows DirectShow 裝置能力顯示曝光、增益、白平衡、對焦及進階畫質，支援自動／手動、滑桿與數值輸入、回讀驗證及還原預設。
- 新增依裝置保存的命名設定檔，運作中保存實際輸出；快照來源紀錄包含輸出格式及最近回讀的硬體參數。
- 預覽頻率可獨立設定為 5／10／15／30 FPS，分開呈現驅動回報、擷取及預覽速率，提供模式不符與低幀率提示。
- 停止相機與擷取畫格固定於面板底部，新增相機使用指南及硬體相容性說明。

## 2.15.0 — 2026-09-17

- 流程 03 明確拆分 SAM2 預訓練互動分割與專案模型自動標註；模型候選先以黃色草稿預覽，僅接受目前圖片後才寫入待審標註，未偵測結果也會明確顯示。
- 流程 07 新增模型試跑：可讀取專案外圖片或完整影片，影片逐幀推論且同一工作只載入模型一次；結果只存於暫存工作階段，關閉程式即清除，不會修改專案。
- 新增 Test／Validation 已標註比對，以 Box IoU 0.50 顯示 TP、FP、FN、Precision 與 Recall；調整信心門檻時同步重算並疊加人工標註與模型預測。
- 耗時工作不再以完成件數或經過時間假裝進度；只有真實進度樣本足夠且持續更新時才估算剩餘時間，否則明示無法估算。
- 優化圖片與 SAM2 工作資料匯入：同一來源只解析一次並重用索引，避免每張圖片重複掃描整個工作階段；保留逐檔格式驗證與可取消處理。
- 補上外部圖片、完整影片逐幀、試跑影格權限、標註比對、未知工期與桌面端全流程回歸測試。

## 2.14.0 — 2026-09-17

- 整頓資料存取、共用標註服務、訓練引擎介面與前端頁面模組；高頻監控改用輕量狀態查詢，審核支援增量回應。
- 新增遷移 0004：歷史標註雜湊索引與無引用資料回收；升級前另存備份，保留既有 0003 備份。0002 遷移保留原本有效的獨立性確認，已遷移專案提供證據檢查與預設試跑的離線修復工具。
- 修正重開後 worker 退出仍卡在 running：核對程序存活及建立時間，重讀最後事件後才判定失敗。
- SAM2 保留模型與 GPU lease，等待者通知或閒置時卸載；GrabCut 改用獨立 CPU 佇列。
- 安裝即時顯示 pip 輸出，取消會清理程序樹；總時間及無輸出逾時可設定，預設不限制安裝總時間。
- 訓練事件依狀態／階段／Epoch 記錄，batch 進度僅覆寫 run.json；序號與終態保護避免過期快照覆蓋結果。
- 補上遷移、重開程序、取消、31,300 次 batch、資料回收與桌面回歸測試。驗證範圍及部署注意事項見 docs/regression-hardening.md。

## 2.13.0 — 2026-09-17

- 訓練監控將重複的摘要與執行設定整併為 4 欄 × 2 列、共 8 格的單一摘要表；Epoch 僅顯示一次，資料版本、引擎、模型與裝置集中呈現。
- 新增 Precision／Recall 曲線診斷：偵測前段劇烈變動與持續落差，分開列出實際 Epoch 證據、可能原因、正常性判讀與處理建議，並納入 Validation 樣本數影響。
- 新增 mAP 數值來源對照，明示 Epoch 曲線使用 Validation 當輪權重、最終表使用的集合與 checkpoint，並說明 mAP50 與 mAP50–95 的 IoU 定義差異及非獨立 Test 限制。

## 2.12.0 — 2026-09-16

- 已審核獨立樣本預設直接建立圖片層級的多類別綜合平衡預覽；介面分開顯示可分配圖片與來源紀錄，不再把單一來源批次誤認成只有一個可分割單位。
- 平衡分割維持固定種子可重現，綜合考量集合比例、類別出現張數、物件數與常見共現配對；Train 類別缺失、空集合及完全相同圖片跨集合仍屬硬性阻擋。
- 資料增強新增每張 Train 原圖擴充份數；每份作為獨立訓練事件重新抽樣增強，DatasetVersion 與 Run 保存原圖、擴充量、每輪事件總數，Validation／Test 不擴充。
- TorchVision 分類、偵測、實例／語意分割與 Ultralytics YOLO／RT-DETR 均依擴充份數增加訓練事件；YOLO 轉換副本優先以 hard link 共用原圖，避免不必要的磁碟複製。

## 2.11.0 — 2026-09-16

- 資料分割新增正式泛化評估、人工確認獨立樣本、寬鬆實驗與全資料最終訓練四種用途；四種多類別平衡目標可在確認獨立後按圖片使用，同批次不再一律阻擋。
- 資料審核保存獨立性確認範圍、資產與標註指紋、時間；圖片、標註、來源或審核狀態變更時自動失效，相同圖片在任何模式仍強制留在同一集合。
- 分割用途、演算法、種子、來源隔離與警告保存到 DatasetVersion／Run；監控頁明確區分正式評估、圖片層級平衡、寬鬆實驗與無獨立評估的最終訓練。
- 標註工具列把快捷鍵整合到英文工具名稱，移除畫布下方重複的快捷提示列。

## 2.10.1 — 2026-09-16

- 資料準備頁先呈現增強配方、再呈現原始來源分割，並明示實際管線只對分割後的 Train 即時增強，避免資料變體洩漏至 Validation／Test。
- 修正增強預覽卡片被表單高度撐開時 SVG 標註層貼齊卡片底部、與圖片分離的問題；圖片與標註改用相同舞台尺寸及翻轉座標。
- 桌面訓練設定恢復訓練資料、參數設定、執行設定同列 1×3 版型；窄視窗仍自動改為雙欄或單欄。

## 2.10.0 — 2026-09-16

- 將「資料分割」擴充為「資料準備」，新增 Train-only 資料增強配方、原圖／同步標註預覽；配方會固定在 DatasetVersion，原圖與 Validation／Test 不受影響。
- YOLO 明確套用顏色、翻轉、幾何及 Mosaic 參數；TorchVision 分類、偵測與分割同步套用顏色與翻轉，框及遮罩會跟著幾何變換。
- 批次大小移至基本設定，執行摘要即時計算每輪 Batch 與有效批次；背景列與監控卡分別顯示 Epoch、Batch 及真正成功的權重更新次數。
- 訓練設定改為基本設定、學習率排程與進階設定分頁；暖身為 0 時明示關閉，啟用函數只顯示模型架構資訊，不再與學習率排程混淆。

## 2.9.1 — 2026-09-16

- 訓練監控改為四格關鍵摘要與效果、損失、學習率、全部圖表分組，保留完整指標並降低長頁面瀏覽負擔。
- 圖表卡片統一高度，新增監控區全螢幕檢視；小螢幕自動改為單欄排列。
- 比較提示改為可收合摘要，Epoch 明細表頭與 Run 欄位固定；全專案介面文字不得低於 12px。

## 2.9.0 — 2026-09-16

- 新增 YOLO26n／YOLO26s Detect 物件偵測訓練、Validation／Test 評估、矩形候選回填與模型封裝。
- 訓練圖表讀取 Ultralytics 原生結果，納入 Train／Validation 各項 loss、Box／Mask Precision、Recall、mAP50 與 mAP50–95；同義欄位正規化，重複學習率群組沿用合併顯示。
- 低於 0.1 的 0–1 指標使用明確標示的放大 Y 軸，避免低 mAP 被誤認為零。
- 實際學習率缺口會說明成功更新次數；新 Run 另記更新嘗試、成功與 AMP 跳過次數。

## 2.8.3 — 2026-09-16

- 新增隨機寬鬆分割：按圖片比例與種子分配，來源不足及缺少類別改為警告；保留圖片配對、去重群組、鎖定與非空集合檢查。
- 寬鬆政策隨明確套用的分割保存至資料版本與訓練警告；舊版與一般模式維持嚴格檢查。
- 像素模型以空預測標記未學習類別，無法校準的類別保留預設門檻；分類模型允許明確實驗模式缺少訓練類別。

## 2.8.2 — 2026-09-16

- 分割管理新增多類別綜合、出現張數、物件數與共現配對平衡，來源隔離改為獨立選項。
- 稀少類別優先搜尋並加入群組交換；維持整張圖片、相同圖片與手動鎖定的完整性。
- 預覽顯示逐類圖片／物件數、來源與分配群組數、比例差距、樣本不足與常見配對分布。

## 2.8.1 — 2026-09-16

- 資料審核改為整張卡片勾選，預覽與編輯使用獨立按鈕；編輯返回保留勾選、篩選、分頁與捲動位置。
- 支援待修正、批次排除原因與備註、可還原垃圾桶；還原後回到待審核，不改動外部來源及既有訓練版本。
- 匯入前提供縮圖勾選與疑似模糊提示；既有圖片可執行模糊檢查，按狀態與原因篩選。
- SAM2 工作資料可匯入為待審核圖片與遮罩；標註畫布新增細十字準線，類別選擇跨圖片保留。

## 2.8.0 — 2026-09-15

- DatasetVersion、Run、Model、候選標註、模型匯出與編輯器同步資料改為保存在所屬專案資料夾；專案改名、備份與刪除會涵蓋完整模型生命週期。
- 啟動時安全遷移舊版全域目錄；既有 DatasetVersion 圖片經 SHA-256 驗證後以 NTFS hard link 共用不可變原圖，後續版本建立時沿用相同機制。
- 專案 SQLite 新增資料版本、訓練、模型、模型匯出與候選標註 catalog，使用外鍵、非空 ID、格式與狀態約束保存 lineage，並提供 integrity 與 filesystem catalog 檢查。

## 2.7.7 — 2026-09-15

- 評估改用資料集整體累積 TP／FP／FN；空白標註與空白預測列為 N/A，不再替漏分割累加 1.0。Mask R-CNN、內建像素模型與 DeepLabV3 同步套用，保存 Macro／Micro IoU、像素數與各類別明細。
- Validation 成為訓練與選模必要條件，並須涵蓋所有已有標註的類別；Test 只做最終評估。缺 Test 時保存 `null` 並顯示尚未執行，不再複製 Validation 或用 Test 調門檻、選 checkpoint、控制學習率。
- Faster R-CNN 評估改為依信心排序的一對一配對，記錄 FP、FN、Precision、Recall、F1 與 AP50；重複框和背景錯誤框會確實扣分。分類指標附上 support，Macro 僅納入該集合有真實樣本的類別。
- 每份新評估保存計算版本、選模集合、權重來源、Test 是否存在及來源獨立性；來源群組跨集合時列出實際重疊的 session。介面顯示 N/A、實際評估集合與歷史重評原因，不再把 Validation 誤標成 Test。
- 支援版本化歷史重評：以 `evaluation.v2.json` 提供修正結果、無效評估原因與來源追溯，並保留原始評估紀錄。

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
