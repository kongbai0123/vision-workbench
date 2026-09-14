# Vision Workbench

Vision Workbench 是以 Windows 為主要平台的本地視覺資料與模型工作站，將影像採集、資料匯入、三種標註編輯器、人工審核、固定訓練資料、模型訓練、評估、預標註與外部交換整合在同一套桌面軟體中。

軟體不需要帳號或遠端服務，影像、標註、修訂紀錄與模型皆保存在使用者電腦。專案資料由 SQLite 交易管理，原始影像以內容雜湊保存，避免不同流程間反覆包裝及交換中間檔案。

## 主要流程

```text
建立專案 → 擷取／匯入 → 標註 → 人工核准 → 建立 DatasetVersion → 訓練與評估 → 模型預標註 → 人工修正與再次審核
```

## 功能

- Windows 相機偵測、解析度與 FPS 模式選擇、即時預覽、快照、自動擷取及錄影。
- 圖片、資料夾、影片畫格、螢幕擷取及其他工作台專案整併。
- LabelMe、COCO、YOLO、JSONL 與 Vision Workbench 原生格式匯入。
- Bounding Box、旋轉框、Polygon、折線、關鍵點及像素遮罩標註。
- SAM2 與 GrabCut 候選遮罩；候選需由使用者接受後才會寫入標註。
- 每個標註物件獨立保存 `cls`，專案類別完全由使用者新增及刪除。
- 圖片審核、退回、修訂歷史、衝突檢查及待審狀態追蹤。
- 原生、COCO、YOLO Detection、YOLO Segmentation、LabelMe、圖片分類及 JSONL 匯出。
- 匯出前檢查圖片雜湊、幾何座標、類別、審核狀態、資料分組及格式轉換損失。
- 同一視窗切換內建編輯器、Labelme 7.4.1 與本機 CVAT，共用專案圖片、標註及審核狀態。
- 已核准資料直接建立不可變的 `D001`、`D002` 訓練資料版本，不需要先匯出再匯入另一套訓練程式。
- 訓練由獨立程序執行；離開訓練頁或關閉工作台不會改寫進行中的資料版本，Run 狀態、指標、日誌及產物會持久保存。
- 正式 TorchVision Mask R-CNN 路徑支援原生 RLE 遮罩、CUDA／CPU、checkpoint、mask IoU 評估與隔離推論。
- Faster R-CNN MobileNet V3／320 與 ResNet50 FPN V2 可直接由現有實例遮罩推導緊密框，完成偵測訓練、框評估與候選回填。
- DeepLabV3 MobileNet V3／ResNet50 可由原生面積標註建立語意 class map，完成訓練、mIoU／Dice 評估與遮罩候選回填。
- MobileNet V3、EfficientNet-B0 與 ResNet18 可直接訓練圖片分類；每張圖片需只有一個標註類別，結果提供 Accuracy／Macro F1／Recall，不會被轉成覆蓋整張圖片的框。
- RT-DETR ResNet50 與 YOLO26n/s Seg 使用獨立 Ultralytics 環境；由固定資料版本建立訓練資料、持續寫入 Epoch 指標、保存 checkpoint，並將框或遮罩候選送回同一審核流程。
- YOLO Seg 可先掃描完整固定資料版本；預設只在 Run 專屬副本修補安全門檻內的微小封閉孔洞，並保存圖片、標註、像素、座標、比例與 IoU 稽核。大孔洞、裂縫或多區塊仍會阻擋。
- 設定中心羅列可用及規劃中的模型、所需標註、元件狀態與授權；選取未安裝模型只顯示說明，使用者按下安裝後才準備元件。
- 內建像素原型分割可在未安裝 PyTorch 時立即完成資料到模型的快速基準驗證。
- 模型輸出先保存為候選；接受後才加入圖片並回到待審核，不會直接覆蓋或核准人工標註。
- 訓練頁上方三個設定框等高對齊，下方使用完整寬度顯示摘要及雙欄圖表；執行紀錄移至側欄。
- 訓練頁自動呈現實際記錄的 Loss、IoU／Dice、Accuracy／F1 或 mAP；勾選要比較的已完成模型，所有曲線、摘要與明細一起更新。
- 各模型提供實際支援的影像尺寸、批次大小、學習率、權重衰減及最佳化器；內建像素基準提供門檻搜尋上下限，參數會固定保存於 Run。
- X 軸涵蓋完整 Run，比例指標維持 0–1；隱藏曲線不改變座標，缺值不補零。滑鼠、點選或方向鍵可查看同一 Epoch 各 Run 的數值，並提供數值明細與各類別評估。
- 完成的 ModelVersion 可匯出成獨立 ZIP，包含 checkpoint、設定、評估、指標、來源 lineage 與 SHA-256 manifest，不包含訓練圖片。

## 系統需求

- Windows 10 或 Windows 11（64 位元）
- Python 3.13（64 位元）
- 建議至少 8 GB RAM
- 相機功能需要 Windows 可辨識的 UVC 或相容影像裝置
- SAM2 可使用 CPU；若需要 CUDA 加速，需具備相容的 NVIDIA GPU 與驅動程式

## 安裝

```powershell
git clone https://github.com/kongbai0123/vision-workbench.git
cd vision-workbench
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

完成後雙擊 `vision-workbench.bat`，或執行：

```powershell
.\.venv\Scripts\python.exe .\main.py
```

### 選用 AI 模型

SAM2、PyTorch 與模型權重不包含在 Git 儲存庫中。需要 AI 分割時執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1 -AI
```

首次準備需要網路。模型下載完成後，推論可在本機執行。GrabCut 不需要額外模型權重。

### TorchVision 訓練環境

正式實例分割使用獨立的 `.venv-training`，避免 Torch／CUDA 依賴影響桌面介面。首次安裝執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1 -Training
```

此設定會安裝鎖定的 Torch 2.7.1、TorchVision 0.22.1 與 CUDA 12.8 runtime，供 Mask R-CNN、Faster R-CNN、DeepLabV3 與圖片分類模型共用。也可以從「設定 → 模型與元件」選擇支援模型後按需安裝；工作台會在安裝後重新執行載入與 CUDA 檢查。未安裝時仍可使用內建基準引擎。

RT-DETR 與 YOLO26 Seg 使用 `.venv-models/ultralytics`。請先閱讀模型頁顯示的授權資訊，再於「設定 → 模型與元件」按「安裝必要元件」；選取模型本身不會建立環境或下載權重。Workbench 從套件內建 YAML 架構開始訓練，不會自動套用預訓練 checkpoint。

## 使用方式

1. 建立本地專案。新專案不會自動建立任何物件類別。
2. 在「物件類別」管理中新增專案需要的 `cls`。
3. 從相機、圖片、資料夾、影片、螢幕或其他專案加入素材。
4. 選擇下一個新物件的類別，再使用標註工具建立物件；該選擇在建立完成後自動清除。
5. 既有物件可在物件清單中逐一修改 `cls`，不會影響其他物件。
6. 完成人工檢查並核准影像。修改已核准標註後，影像會回到待審狀態。
7. 在「資料審核」核准可用圖片，下一步進入獨立的「資料分割」，預覽並設定 Train／Validation／Test。
8. 分割符合條件後進入「模型訓練」，建立固定資料版本，選擇實例分割、物件偵測、語意分割或圖片分類模型並啟動 Run；未安裝及規劃中的模型仍可查看用途與準備方式。
9. 在「評估與模型」查看任務對應指標、匯出模型封裝，或產生預標註候選；接受候選後回到人工審核。
10. 模型封裝位於 `data/model-exports/{project_id}/{export_id}`；需要資料集備份或外部交換時，再開啟「備份與外部交換」。

### 智慧資料分割與動態學習率（2.6）

「資料分割」是資料審核後、模型訓練前的獨立流程階段。摘要頁集中顯示已核准圖片、獨立來源群組、目前比例、方案狀態與警示；流程為設定比例／種子 → 預覽群組與類別分布 → 套用，或「套用並建立新資料版本」。只處理目前已核准圖片；分割不會改動標註與核准狀態。訓練頁可返回此階段重新調整。

預設智慧分割以來源批次、同影片、相同圖片 SHA-256 建立不可拆分的群組，接著盡量平衡類別實例數及 Train／Validation／Test 比例。新抽取影片會記錄影片內容雜湊，移動路徑後再次匯入仍可辨識；舊影片紀錄依來源路徑識別，可用手動群組補充關聯。這是來源與標註統計分組，尚未使用影像 embedding 偵測近似圖片。

可指定群組集合、保留目前 Test（含同組圖片）、手動命名獨立情境群組。只有確認圖片彼此獨立時才使用「獨立圖片 · 類別平衡」策略；此策略仍合併完全相同圖片。若群組不足、鎖定衝突或類別無法覆蓋，介面會提示原因；群組隔離優先於精確比例。智慧分割能降低資料洩漏與分布偏差，不能保證 Train／Validation Loss 相同。

預覽後資料或設定改變時必須重新預覽。套用會保存策略、種子、群組、鎖定及分配結果；D001 等既有固定資料版本不變，新建立的 D002 會保存新的分割方案。比較模型時應固定同一資料版本，Test 保留供最終評估。

神經網路模型的「參數設定 → 學習率策略」支援固定、Cosine 衰減、線性衰減，以及暖身輪數、最低學習率與排程預覽。TorchVision 另支援依 Validation 指標停滯降率，可設定耐心輪數及下降倍率；缺少 Validation 時拒絕此策略，不使用 Test 決定降率。Ultralytics 使用原生排程，其預覽為趨勢示意，實際值以訓練記錄為準。

Run 保存實際設定，每輪的實際 LR 會顯示在監控圖表與 Epoch 清單；多個 optimizer 群組會分別記錄。TorchVision 另保存 scheduler／optimizer 狀態；本版未新增斷點續訓入口。舊 Run 不會補造不存在的 LR 資料，內建像素原型也不顯示無作用的梯度參數。

### 訓練監控與模型比較（2.5）

「單次訓練」先顯示評估摘要、固定執行設定，再以雙欄自動呈現模型已實際記錄的指標。沒有紀錄的 Validation Loss、學習率或硬體曲線不會被推測產生。

按「Run 疊圖比較」會列出相同資料版本的已完成模型，以模型版本、Run 編號和引擎名稱辨識。勾選 M001、M002 並取消 M003 後，圖表、摘要、Epoch 明細、詳細評估與設定都只呈現 M001、M002；不用另外切換 Loss 或 IoU 圖表。可自由選擇要比較的模型，取消勾選不會刪除模型。其他執行紀錄仍可從側欄查看。

Epoch 明細以清單呈現，最多顯示 10 列資料，超過時在清單內捲動，欄位標題固定。即時更新保留目前捲動位置。詳細評估與完整執行設定在寬視窗並排，較窄視窗改為上下排列，設定以具名欄位呈現。

神經網路模型可調整輸入影像尺寸、批次大小、學習率、權重衰減與 AdamW／SGD 最佳化器。選用不同模型時保留各自尚未送出的參數，可按「恢復預設值」重設。內建像素原型固定使用 CPU，改以門檻搜尋次數及上下限設定；不顯示不會使用的梯度參數。參數在啟動前檢查，無效值會被拒絕；不會偷偷截斷或忽略使用者輸入。

比較使用相同資料版本與評估分割。不同引擎的 Loss、Mask IoU 與語意 IoU 可能有不同定義，無法直接比較時會停用該指標疊圖並顯示原因，其餘可比較的指標繼續顯示。未設定 Validation 時，訓練期間使用 Test 的曲線會明確標示；詳細評估提供已記錄的各類別 IoU／Dice／Recall 或分類混淆矩陣。

曲線不會左右掃視，隱藏 Run 不會縮放座標。若新數值超出原 Loss 範圍，座標只擴大到能完整呈現資料並顯示提示，不會持續自動縮小。停止的 Run 或缺少某個 Epoch 指標時保留缺口；圖表會列出缺少的 Epoch 並說明這是指標缺值，不代表訓練中斷，數值表以「—」表示缺值。

執行中的桌面程式若偵測到待套用修改，頂部「設定」、設定內的「更新與版本」分類及「Vision Workbench 程式」卡片會同步顯示紅點。進入該頁可查看檔案與阻擋原因；按一次「套用更新並重新啟動」便會保存目前工作、驗證程式並載入 2.7.0，不再開啟另一個更新視窗。本版未新增套件需求；程式與模型元件分開更新。

### YOLO Seg 遮罩相容處理（2.7）

在訓練頁選擇 YOLO26n/s Seg 後，「模型參數 → YOLO Seg 相容處理」提供兩種方式：

- 「修補微小封閉孔洞」預設只接受單一孔洞 ≤ 4 px、單一實例合計 ≤ 16 px、修補面積占原 Mask ≤ 0.01%，且孔洞寬與高皆 ≤ 4 px。
- 「嚴格無損」只要偵測到孔洞便停止建立 Run，適合標註幾何必須逐像素一致的工作。

「檢查全部圖片」會掃描所選的固定 DatasetVersion。結果列出圖片名稱、類別、標註 ID、資料分組、孔洞數量、像素座標及範圍；按「放大位置」會用像素化預覽、紅框與十字顯示位置，因此 1 px 孔洞也能看見。開始訓練時會再次執行同一套後端檢查，不能由前端略過。

通過門檻的孔洞只會填入 `data/runs/{project_id}/{run_id}/dataset/` 內的 YOLO 副本。專案原始 Mask、審核狀態及 D001／D002 等固定資料版本不會被修改。`yolo-compatibility.json` 保存每張圖片的修補像素、外框、占比與轉換 IoU，並寫入評估與 ModelVersion。YOLO 內部 Validation／Test 指標使用該相容副本；原生 Mask 持續作為可追溯來源。超出門檻的孔洞、細長裂縫及多個分離區塊仍阻擋訓練，可改用 Mask R-CNN 保留原生 RLE 幾何。

### 類別管理規則

- 新增類別只增加可用名稱，不會批量修改既有標註。
- 刪除未使用類別會直接移除該名稱。
- 刪除使用中類別時，必須明確選擇將全部相關物件改為另一類，或刪除相關標註物件。
- 每個物件的類別彼此獨立；新物件類別與既有物件類別不會互相連動。
- 匯入資料本身含有標註類別時，會保留來源資料定義的類別。

### 三個編輯器共用專案

在「標註編輯」上方選擇內建編輯器、Labelme 或 CVAT。Labelme 使用原生嵌入介面；CVAT 使用同視窗中的本機網頁介面。兩者會載入目前專案與所選圖片，無須另行匯入或匯出標註。

Labelme／CVAT 上方可直接切換編輯器，或選擇「儲存並前往審核」。切換會等待存檔、讀回專案完成，下一個編輯器才會開啟。新標註、座標修改、類別修改及刪除都會同步；實際變更的影像回到待審核，沒有修改的影像保留原審核狀態。原始圖片內容保持不變。

共用標註支援矩形、旋轉框、多邊形、折線、點與像素遮罩（包含孔洞）。Labelme 圓形會以像素遮罩保存。Labelme 物件的群組、描述與旗標保存在專案 metadata。CVAT 的影片軌跡、整張標籤及尚無共用表示的形狀會阻止同步並保留原編輯器，避免悄悄捨棄資料。

若同一張影像在兩邊同時改動，會提示版本衝突並保留修改。Labelme 復原 JSON 位於 `data/labelme/{project_id}/`；CVAT 推送前的備份位於 `data/cvat/annotation-backups/`。圖片匯入、刪除及專案管理統一由工作台操作。

Labelme 編輯整合需要桌面版；`--serve` 瀏覽器模式仍可使用內建編輯器。Labelme 的選用 AI 功能由其模型設定決定，首次使用可能需要下載模型。

## 支援格式

| 格式 | 匯入 | 匯出 | 內容 |
| --- | :---: | :---: | --- |
| Vision Workbench 原生格式 | ✓ | ✓ | 原圖、完整幾何、遮罩、來源、審核與修訂資訊 |
| COCO | ✓ | ✓ | Bounding Box、Polygon 與 RLE Mask |
| YOLO Detection | ✓ | ✓ | 正規化偵測框與類別定義 |
| YOLO Segmentation | ✓ | ✓ | 多邊形分割與轉換損失報告 |
| LabelMe | ✓ | ✓ | 逐圖 JSON 與向量標註 |
| JSONL | ✓ | ✓ | 每張影像一筆完整原生標註 |
| 圖片分類目錄 | — | ✓ | 依資料分組與唯一圖片類別建立目錄 |

YOLO Polygon 無法精確表達遮罩孔洞，Detection 也不保留物件輪廓。工作台會在匯出前列出轉換影響，不會無提示地捨棄幾何資料。

## 資料保存與安全

```text
data/
├─ projects/   專案資料庫、標註、修訂及原圖副本
├─ incoming/   相機、錄影、影片及螢幕擷取素材
├─ datasets/   不可變的訓練資料版本與 lineage manifest
├─ runs/       訓練狀態、指標、評估、日誌與 artifact manifest
├─ models/     模型版本、checkpoint 與能力資訊
├─ predictions/模型預標註候選及接受狀態
├─ exports/    驗證完成的資料集版本與 ZIP
└─ logs/       本機輪替紀錄

models/        選用的本機 AI 模型
```

- `data/`、`models/`、匯出資料、測試截圖與本機虛擬環境皆已排除於版本控制。
- 軟體只監聽 `127.0.0.1`，寫入 API 會檢查本機 Host、同源與應用程式標頭。
- 原始輸入檔案不會被回寫；工作台在自己的資料目錄保存副本。
- 備份完整工作進度前，請先正常關閉軟體，再備份整個 `data/` 目錄。

## 開發

建立一般開發環境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

以瀏覽器模式啟動：

```powershell
.\.venv\Scripts\python.exe .\main.py --serve --port 8780
```

執行自動測試：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
node --test tests\ui_editor.test.mjs
node --test tests\ui_cvat.test.mjs
node --test tests\ui_training_charts.test.mjs
node --test tests\ui_training_parameters.test.mjs
```

執行隔離的桌面工作流程驗證：

```powershell
.\.venv\Scripts\python.exe tests\desktop_workflow.py
.\.venv\Scripts\python.exe tests\desktop_training.py
```

安裝獨立訓練環境後，可執行一輪真實 CUDA／CPU Mask R-CNN 煙霧測試：

```powershell
.\.venv\Scripts\python.exe tests\maskrcnn_smoke.py
```

已準備本機 CVAT 時，可執行 `.\.venv\Scripts\python.exe tests\desktop_editors.py` 驗證三個編輯器與審核的完整往返；它會建立並清除獨立的 CVAT 測試專案。

測試會使用暫存資料夾，不會操作使用者的正式專案或相機，除非明確執行 `tests\hardware_camera.py`。

## 專案架構

| 路徑 | 職責 |
| --- | --- |
| `web/` | 桌面介面、標註畫布、採集工作區與自動儲存 |
| `workbench/store.py` | 專案資料、SQLite 交易、修訂、審核及衝突檢查 |
| `workbench/server.py` | 本機 API、背景工作與流程協調 |
| `workbench/acquisition.py` | 相機、錄影、畫格擷取、即時處理與 AI 呼叫 |
| `workbench/pipeline.py` | 格式解析、整併、驗證與原子化匯出 |
| `workbench/training.py` | DatasetVersion、Run、ModelVersion、PredictionCandidate 與獨立程序管理 |
| `workbench/training_engine.py` | 內建原生遮罩基準訓練、評估與推論 |
| `workbench/maskrcnn_engine.py` | TorchVision Mask R-CNN 原生遮罩訓練、mask IoU、checkpoint 與推論 |
| `composer_core/` | 共用幾何資料結構與驗證 |
| `classical_segmentation/` | 傳統影像分割與背景處理 |
| `sam2_segmentation/` | SAM2 分割、後處理與資料集轉換 |
| `tests/` | 資料層、API、格式、桌面介面及硬體隔離測試 |
| `docs/` | 整合契約、資料來源及檢測設計文件 |

## 設計原則

- 原圖保持不變，處理結果與標註分開保存。
- 每次修改都有明確修訂版本；過期寫入會被拒絕。
- AI 只產生候選，不直接核准或覆蓋人工標註。
- 資料集只有在驗證完成後才會發佈，既有版本不會被覆寫。
- 自動分割固定相同 SHA-256 圖片；來源批次跨組會在建立訓練資料前顯示資料洩漏提醒。

## 目前範圍

2.0 版已交付影像採集至實例分割模型預標註的本機閉環。正式產線部署、模型服務監控及毫米級量測校正仍需依實際相機、鏡頭、治具、光源與標定流程另外驗證。

2.1 版已交付設定中心、模型目錄、TorchVision 按需準備，以及 Faster R-CNN 與 DeepLabV3 完整訓練／評估／候選流程。影像分類、EfficientAD／PatchCore、RT-DETR、YOLO26 Seg 已顯示於目錄並明確標示整合開發中；安裝入口會在對應 adapter 完成後開放。範圍、介面與驗收清單見 [模型中心與設定中心規劃](docs/model-center-settings-plan.md)。

## 授權

本專案採用 [MIT License](LICENSE)。
