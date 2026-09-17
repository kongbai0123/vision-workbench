# 開發指南

[返回 README](../README.md)

以下命令於專案根目錄的 PowerShell 執行。主要開發環境為 Windows 10／11、64 位元 Python 3.13；JavaScript 測試另需支援 `node --test` 的 Node.js。

## 安裝與啟動

建立桌面執行環境：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

使用獨立資料目錄啟動桌面版：

```powershell
.\.venv\Scripts\python.exe .\main.py --data-root .\verification\dev-data
```

瀏覽器開發模式：

```powershell
.\.venv\Scripts\python.exe .\main.py --serve --port 8780 --data-root .\verification\dev-data
```

開啟終端輸出的本機網址；按 `Ctrl+C` 停止服務。未指定 `--data-root` 時使用專案下的 `data/`。瀏覽器模式不提供 Labelme 原生嵌入介面。

## 自動測試

架構整頓的逐項實作、相容性決定與驗證紀錄見 [architecture-hardening.md](architecture-hardening.md)。

使用來源 checkout 時可安裝開發套件（不安裝模型、不更新產品版本）：

```powershell
.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.\.venv\Scripts\python.exe -m pip install 'ruff>=0.11,<1'
.\.venv\Scripts\python.exe -m ruff check workbench src composer_core sam2_segmentation classical_segmentation tests
node --test tests/*.test.mjs
```

新共用契約位於 `src/vision_workbench/contracts`；來源 checkout 的相容入口與 editable install 均可讀取。桌面原始碼監看只在 `VISION_WORKBENCH_DEVELOPMENT=1` 開啟，一般啟動不啟用。

瀏覽器啟動請使用服務輸出的含 session 入口網址。它會設定 HttpOnly Cookie 並導向不含 token 的網址；自動測試／本機 API 客戶端須提供本次程序的 `X-Workbench-Token`。不要把入口 token 貼進問題單或提交版本庫。

Python 測試包含資料層、API、格式轉換與訓練邏輯；JavaScript 測試使用 Node.js 內建測試工具，無須安裝 npm 套件。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
node --test tests\ui_editor.test.mjs
node --test tests\ui_cvat.test.mjs
node --test tests\ui_training_charts.test.mjs
node --test tests\ui_training_parameters.test.mjs
```

部分測試在缺少選配模型環境時會跳過；通過一般測試不代表已驗證真實模型訓練或相機硬體。

## 選配整合測試

桌面流程與圖表測試使用 PySide6／Qt WebEngine，截圖等產物寫入 `qa-output/`：

```powershell
.\.venv\Scripts\python.exe tests\desktop_workflow.py
.\.venv\Scripts\python.exe tests\desktop_training.py
.\.venv\Scripts\python.exe tests\desktop_training_lr.py
```

先建立獨立 TorchVision 環境，再執行真實模型煙霧測試。這些測試會啟動訓練程序並使用 CPU 或可用的 CUDA 裝置：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1 -TrainingOnly
.\.venv\Scripts\python.exe tests\maskrcnn_smoke.py
.\.venv\Scripts\python.exe tests\torchvision_models_smoke.py
.\.venv\Scripts\python.exe tests\classification_models_smoke.py
```

- SAM2 開發環境使用 `bootstrap.ps1 -AI`，會安裝選配依賴並下載模型。
- Ultralytics 環境由「設定 → 模型與元件」安裝，依賴列於 `requirements-ultralytics.txt`。
- `tests/desktop_editors.py` 需要已啟動的本機 CVAT，且含特定本機帳號與位址設定；執行前需配合測試環境調整。測試會建立及清除 CVAT 測試專案。
- `tests/hardware_camera.py --camera-index <索引>` 會操作指定實體相機，須明確指定裝置。

## 歷史模型回測

`tests/backtest_historical_models.py` 是特定資料結構的回測工具，要求指定專案具有 D001／M001、D004／M004、D005／M005。它讀取真實專案與 checkpoint，不屬於一般單元測試；執行環境須能匯入 TorchVision 與 Ultralytics。

必要參數為 `--project-id` 與 `--output`，可用 `--root` 指定儲存庫位置、`--device cpu` 切換裝置，預設使用 CUDA。`--install` 另會寫入版本化的 `evaluation.v2.json`，供工作台讀取重算結果；原始評估檔案仍保留。

## 程式碼配置

| 路徑 | 用途 |
| --- | --- |
| `main.py`、`workbench/desktop.py` | 啟動入口與桌面容器 |
| `web/` | 網頁介面、標註畫布與訓練圖表 |
| `workbench/store.py`、`workbench/server.py` | 專案儲存、本機 API 與工作協調 |
| `workbench/project_storage.py` | 專案容器路徑、舊版資料遷移與 DatasetVersion 去重 |
| `workbench/training.py`、`workbench/*engine*.py` | 資料版本、訓練程序、模型評估與推論 |
| `composer_core/`、`classical_segmentation/`、`sam2_segmentation/` | 幾何結構與分割處理 |
| `tests/`、`docs/` | 自動測試、整合驗證與技術文件 |

開發資料、模型權重及測試產物應放在已忽略的本機目錄。使用自訂資料目錄時，先確認該路徑已列於 `.gitignore`。
