# Contributing

感謝協助改善 Vision Workbench。提交變更前，請先確認修改範圍、資料相容性與本機工作流程。

## 開發環境

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

安裝使用 64 位元 Python 3.13；既有相容的 Python 3.12–3.14 環境可繼續使用。不要依賴全域 `python` 的預設版本。啟動診斷與修復方式見[啟動疑難排解](docs/startup-troubleshooting.md)。

需要測試 SAM2 時，再安裝 `requirements-ai.txt` 並執行 `prepare_models.py`。模型權重不可提交至版本庫。

## 提交要求

- 保持原始影像不可變，衍生資料需保存來源及修訂資訊。
- 所有標註座標使用原始影像像素座標。
- 不得在程式碼、測試或文件中提交使用者資料、憑證、模型權重或本機絕對路徑。
- 修改儲存格式、匯入／匯出或類別行為時，需加入資料層及 API 測試。
- 修改互動流程時，需更新對應桌面工作流程驗證。
- 使用繁體中文撰寫使用者介面文字，並保持錯誤訊息可操作。

## 驗證

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts\check_runtime.py --report qa-output\startup\runtime.json
.\.venv\Scripts\python.exe tests\desktop_startup.py
node --test tests\ui_editor.test.mjs
.\.venv\Scripts\python.exe tests\desktop_workflow.py
```

硬體相機測試只在明確準備測試裝置時執行：

```powershell
.\.venv\Scripts\python.exe tests\hardware_camera.py
```

## Pull Request

Pull Request 應說明問題、完成後的行為、資料或相容性影響，以及實際執行的驗證。請勿附帶正式專案資料或未去識別化的影像。
