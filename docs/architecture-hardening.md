# 架構整頓：實作與驗收紀錄

> 後續發現的 A–E 回歸與修復驗證，見 [回歸修復紀錄](regression-hardening.md)。下列為第一輪整頓紀錄，不代表當時已覆蓋後續新增案例。

分支：`codex/architecture-hardening`。本次未提高版本號、未建立 Release、未推送 GitHub。
原有未提交功能保留；不可把目前整個工作目錄當成本次單一變更直接提交。

## Phase 0：資料與程序安全

- 圖片永久刪除前同時檢查 assets 與 review_trash；匯入和刪除使用同一個跨程序圖片鎖。
- 先提交資料庫，再清理無引用圖片。交易失敗不刪圖；清理失敗只留下可維護的孤兒檔。
- 缺圖還原會拒絕並保留垃圾桶項目，避免假成功。
- `maintenance.orphan_images` 預設只列出候選；明確要求時移至專案內 orphan-quarantine，可還原、不覆寫。
- 同專案訓練 admission 使用執行緒鎖與 OS 檔案鎖；request_id 相同且內容相同回傳原 Run，內容不同拒絕。
- 子程序取消會終止程序樹；安裝與推論均有 30 分鐘上限，安靜輸出期間仍能取消。
- 程序啟動失敗、非終態退出（含 exit 0）留下明確失敗狀態；等待 GPU 時取消則是 stopped。
- 非預期例外記錄 traceback 並回 500；方法錯誤回 405，認證失敗回 403。
- history(asset_id) 索引；完整 SQLite integrity_check 不在輪詢路徑。

## Phase 1：資料層

- 0000–0003 SQL migrations 統一核心、artifact catalog、垃圾桶、分割、歷史、審核、品質與摘要資料表。
- 僅啟動／建立專案時遷移，不再於一般請求中建表。每個遷移使用交易、版本記錄與跨程序鎖。
- 舊資料庫升級前用 SQLite backup API 保存 WAL 內已提交內容，關閉備份連線後原子發布備份。
- 不再 DROP artifact tables 重建；索引更新比較既有列，僅 upsert 變動內容、移除失效索引列。
- ProjectStore 保留公開 facade；ReviewRepository、HistoryRepository、ArtifactRepository 分別管理垃圾桶、歷史與資料版本摘要。
- 內建、Labelme、CVAT 均走 AnnotationService.commit：共用幾何驗證、微孔修補、版本衝突及核准失效規則。
- 三個編輯器均要求類別已建立；外部編輯器新增類別前需先在類別管理建立。匯入資料仍可建立匯入內容所需的類別。
- 歷史以 annotation_hash 引用 annotation_blobs；相同標註不因審核／分割重複存一份完整幾何。
- annotation_revision 只隨幾何內容變動；僅分割異動不再誤擋持有原標註的編輯器；真正編輯／審核衝突仍受保護。
- review、quality 各存獨立表；source 不再承擔這兩者。API 暫保留 source.review / source.quality 的唯讀相容投影。
- 圖片列表讀取 asset_summaries，不解析所有遮罩；專案列表不再搬移資料夾，獨立性指紋依修訂快取。

## Phase 2：訓練控制與效能

- 啟動前主程式建立初始請求；交接給 worker 後，events.jsonl 是訓練狀態的權威來源，run.json 是相容快照。
- 主程式的停止與異常結束判定寫入 control/，不再讀改寫 worker 的 run.json。舊 stop.requested 仍可讀取。
- worker 每 2 秒更新 heartbeat.json；超過 15 秒標示 stale，不把心跳延遲直接當作失敗或殺掉程序。
- 截斷或格式無效的事件尾端會略過，使用最後完整有效事件。
- SQLite artifact catalog 明確定位成可重建的查詢索引，沒有反向覆寫事件／資料版本的路徑；不再雙方都當主資料。
- 資料版本建立時保存摘要；摘要依 manifest 的時間／大小簽章驗證，並有有界記憶體快取。旧版本第一次讀取後快取。
- 可訓練檢查依專案修訂快取。環境資訊快取過期後背景更新；明確要求重新檢查／安裝仍等待檢查完成。
- 高頻使用 /training/status；不讀資料集、不解碼標註、不做 integrity_check、不啟動環境探測。
- 完整 overview 只在進入、明確更新與終態轉換時取得。
- GPU 使用同一 OS lease，涵蓋訓練、SAM2 與模型推論；CPU 不受 GPU lease 限制。現階段採保守單一 GPU 工作排程。
- 完成時更新衍生 catalog；關閉等待短資料庫交易但不等待整個訓練，防止收尾與資料目錄釋放競態。

## Phase 3：引擎、背景工作與 API

- EngineSpec 統一參數 schema、config／dataset 驗證、train module、predict module；worker 不再按名稱前綴自行選引擎。
- 安裝、互動 AI、一般工作分開 executor；已完成工作定時清理，最多保留 100 筆／1 小時。
- 集中的 method 路由合約；專案操作透過 api_actions 表分派，HTTP request 的共通型別與必要欄位在 contracts 驗證。
- 啟動產生 session token。桌面使用一次入口 URL 換取 HttpOnly、SameSite=Strict cookie，API 同時保留 Host／Origin 驗證。
- 啟動 token 不寫入 HTTP access log；非瀏覽器客戶端可用 X-Workbench-Token。
- immutable 圖片回 ETag 與 private 長效快取，支援 304；縮圖快取有數量與單檔大小上限。
- 高頻審核／指定分割使用增量回應；基底修訂不符回完整專案，前端也會偵測失效 delta 並重新讀取。
- 大型匯入等仍保留完整回應相容性；一般圖片清單已不含完整 shapes。

## Phase 4：前端、清理與開發基礎

- pages/ 拆出 training、settings、review、preparation、export、camera，依賴明確注入，可在無 DOM 環境載入測試。
- state.mjs 集中 session state 與具版本檢查的 delta merge；native-bridge.mjs 集中桌面邊界，既有 Qt callback 名保留相容轉接。
- 審核按鈕／篩選器／樣式與增強預覽容器移回 HTML；不再靠啟動時追加 CSS 或搬動增強面板。
- 匯出頁分割入口改用與資料準備相同的 SplitManager，舊 auto-split API 僅保留客戶端相容性。
- 刪除確定無引用的 sam2_segmentation/verified_export.py；可從 Git 還原。
- SAM2／classical 公開相容工具不是全部無引用：保留其公開 API，改為按需載入，不再於工作台啟動載入整包。
- SAM2 的 RLE 相容函式委派 composer_core.geometry；保留 0/255 像素與原輸入驗證契約。
- src/vision_workbench/contracts 建立新程序邊界契約；workbench 保留相容入口，按原計畫逐步搬移，不做一次性破壞改名。
- pyproject.toml、editable install、Ruff correctness 規則與 CI 驗證；不變更產品版本。
- 原始碼監看重啟僅在 VISION_WORKBENCH_DEVELOPMENT=1 啟用；不把開發中的檔案異動當作產品 Release。
- 原本 12px 最小字體政策保留，響應式訓練頁繼續回歸。

## 有意採用的安全／相容方案

1. 不在啟動時自動永久刪孤兒檔：提供明確、可還原的隔離維護，避免破壞尚待人工確認的圖片。
2. 沒有強迫把 immutable manifests 和 worker events 全部塞入 SQLite。每種記錄只有一個權威來源，SQLite 只作衍生索引；維持訓練程序離線獨立運作與舊資料版本可攜性。
3. 不刪仍有公開匯出／測試使用的 legacy 工具；只移除已確認死碼、取消 eager import、整併重複 RLE。
4. 不將所有寫入 API 強制改成破壞相容的 delta。先在高頻審核／指定操作啟用，revision 不符一律完整同步。
5. 開發包安裝驗證不等同 Windows 獨立安裝程式發行。本次不執行版本發布。

## 驗證

- 完整 Python unittest：251 項，247 通過、4 項 skip（環境／可選引擎條件）。
- JavaScript：52 項全通過。
- Ruff correctness 與 git diff --check 通過。
- 桌面：training 13 個情境、review、multilabel split、settings、camera preview lifecycle 均通過。
- 桌面完整 workflow：專案、匯入、類別管理、標註／AI 候選、Undo／Redo、審核、分割、真正的內建基準訓練、預標註、匯出、重新載入、刪除隔離測試專案均通過。
- contracts 測試涵蓋遷移備份／重開、歷史去重、編輯器共用規則、增量過期、刪除 rollback、孤兒隔離、事件截斷、摘要快取、GPU 鎖與等待中取消。
- API 測試包含錯誤 token、錯誤 method／schema、ETag 304、delta。
- editable install --no-deps --no-build-isolation 驗證通過。

效能實測（tests/benchmark_hardening.py；800 張合成圖片，非使用者資料，單機結果不代表所有實際資料集）：

| 項目 | 結果 |
| --- | ---: |
| 單筆審核＋完整回應，中位數 | 14.386 ms |
| 單筆審核＋增量回應，中位數 | 5.365 ms |
| 完整／增量 JSON 大小 | 517,998 / 1,346 bytes |
| 單 Run status，中位數／最大值 | 0.283 / 0.508 ms |

未執行：使用者原始資料的破壞性遷移演練、真實 GPU 長時間訓練、真正 pip 安裝取消／硬體拔除。取消與程序回收使用隔離子程序驗證，GPU admission 使用 OS lease 驗證；不把這些替代成已通過硬體壓力測試的宣稱。
