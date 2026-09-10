# Vision Workbench

Vision Workbench 是以 Windows 為主要平台的本地影像資料工作站，將影像採集、資料匯入、AI 輔助標註、人工修正、審核、整併、驗證與資料集匯出整合在同一套桌面軟體中。

軟體不需要帳號或遠端服務，影像、標註、修訂紀錄與模型皆保存在使用者電腦。專案資料由 SQLite 交易管理，原始影像以內容雜湊保存，避免不同流程間反覆包裝及交換中間檔案。

## 主要流程

```text
建立專案 → 擷取／匯入影像 → AI 輔助標註 → 人工修正與核准 → 整併與驗證 → 匯出資料集
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
- 選用的本機 CVAT 整合；內建編輯器仍為預設工作方式。

## 系統需求

- Windows 10 或 Windows 11（64 位元）
- Python 3.13（64 位元）
- 建議至少 8 GB RAM
- 相機功能需要 Windows 可辨識的 UVC 或相容影像裝置
- SAM2 可使用 CPU；若需要 CUDA 加速，需具備相容的 NVIDIA GPU 與驅動程式

## 安裝

```powershell
git clone https://github.com/kongbai0123/graph_catch.git
cd graph_catch
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

## 使用方式

1. 建立本地專案。新專案不會自動建立任何物件類別。
2. 在「物件類別」管理中新增專案需要的 `cls`。
3. 從相機、圖片、資料夾、影片、螢幕或其他專案加入素材。
4. 選擇下一個新物件的類別，再使用標註工具建立物件；該選擇在建立完成後自動清除。
5. 既有物件可在物件清單中逐一修改 `cls`，不會影響其他物件。
6. 完成人工檢查並核准影像。修改已核准標註後，影像會回到待審狀態。
7. 指定拍攝批次與 train／val／test 分組，執行驗證後建立匯出版本。

### 類別管理規則

- 新增類別只增加可用名稱，不會批量修改既有標註。
- 刪除未使用類別會直接移除該名稱。
- 刪除使用中類別時，必須明確選擇將全部相關物件改為另一類，或刪除相關標註物件。
- 每個物件的類別彼此獨立；新物件類別與既有物件類別不會互相連動。
- 匯入資料本身含有標註類別時，會保留來源資料定義的類別。

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
```

執行隔離的桌面工作流程驗證：

```powershell
.\.venv\Scripts\python.exe tests\desktop_workflow.py
```

測試會使用暫存資料夾，不會操作使用者的正式專案或相機，除非明確執行 `tests\hardware_camera.py`。

## 專案架構

| 路徑 | 職責 |
| --- | --- |
| `web/` | 桌面介面、標註畫布、採集工作區與自動儲存 |
| `workbench/store.py` | 專案資料、SQLite 交易、修訂、審核及衝突檢查 |
| `workbench/server.py` | 本機 API、背景工作與流程協調 |
| `workbench/acquisition.py` | 相機、錄影、畫格擷取、即時處理與 AI 呼叫 |
| `workbench/pipeline.py` | 格式解析、整併、驗證與原子化匯出 |
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
- 來源批次保持在同一資料分組，降低相近影格跨組造成的資料洩漏。

## 目前範圍

本專案聚焦影像資料製作、審核與匯出。模型訓練、產線推論部署及毫米級量測校正仍需依實際相機、鏡頭、治具、光源與標定流程另外驗證。

## 授權

本專案採用 [MIT License](LICENSE)。
