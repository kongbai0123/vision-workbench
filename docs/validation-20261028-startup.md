# v2.21.10 啟動修復驗證紀錄

日期：2026-10-28。範圍為 Windows 啟動、環境修復、診斷與本機來源更新；使用 64 位元 Python 3.13.5，啟動器整合測試使用 Windows PowerShell 5.1。

## 問題與修復

原啟動器只確認 `pythonw.exe` 存在，無法辨識虛擬環境建立後尚未安裝套件的狀態；使用隱藏主控台的 Python 啟動時，缺套件錯誤也不會顯示。全域 Python 若被其他工具改成 3.11，則無法安裝目前固定的 NumPy／Labelme 版本。

新版在啟動前核對 Python 版本、64 位元架構、套件版本與核心原生匯入，安裝時優先使用 Python 3.13。需要重建時先保留舊環境，以檔案鎖避免並行安裝。自動修復最多執行一次；錯誤與紀錄位置會保留在主控台，並提供獨立診斷及修復模式。

Windows 下載來源／信譽提示發生於腳本執行之前，與 Python 的 TLS 憑證及網頁 session 不同。文件提供可信來源檔案的個別處理方法；本版未提供 Windows 發行者簽署安裝包。

## 驗證結果

| 檢查 | 指令 | 結果 |
| --- | --- | --- |
| 完整 Python 測試 | `python -m unittest discover -s tests` | 401 項，0 失敗，4 項依環境條件跳過 |
| JavaScript 測試 | `node --test tests/*.test.mjs` | 84 項通過 |
| Python 正確性規則 | `python -m ruff check workbench src composer_core sam2_segmentation classical_segmentation scripts tests` | 通過 |
| 真實桌面入口與本機服務 | `python tests/desktop_startup.py` | 介面就緒、本機 API 驗證、儲存／關閉交接、服務與鎖檔清理通過 |
| 套件一致性 | `python -m pip check` | 無損壞依賴 |
| 發布一致性 | 程式版本、README、CHANGELOG 與 `git diff --check` | 同步為 2.21.10，無空白格式錯誤 |

啟動專項測試另覆蓋：缺少環境時診斷不觸發安裝、修復次數上限、舊環境保留、健康環境重用、連結／越界名稱拒絕、並行安裝鎖、原生 stdout／stderr 紀錄、中文與空白路徑、相對資料路徑、尾端反斜線與引數跳脫。以隔離程序模擬 OpenCV 不可匯入時，桌面模組及核心服務仍可載入與結束。

## 相容性與範圍

- 主程式固定依賴要求對齊為 Python 3.12–3.14；安裝腳本建立環境時使用 3.13。
- 專案資料、模型與資料庫格式不變；OpenCV 延遲載入保留既有功能演算法，在功能實際使用時才回報元件不可用。
- 測試使用隔離的暫存資料；真實相機、完整外部模型與 GPU 訓練不屬於本次驗證。跳過的測試不代表已驗收相應環境。
- GitHub CI 會另外驗證完整 Windows 工作流程；正式版本標籤於發布提交通過 CI 後建立。
- 診斷紀錄、畫面截圖、虛擬環境與備份已排除版本控制。

使用者操作見[啟動疑難排解](startup-troubleshooting.md)，變更摘要見[版本紀錄](../CHANGELOG.md)。
