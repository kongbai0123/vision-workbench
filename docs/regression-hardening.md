# 架構整頓後的回歸修復 A–E

分支：`codex/architecture-hardening`。原始碼快照提交：`98e925b`。
此快照包含先前尚未提交的功能與架構整頓，不宣稱它已修正本文件列出的回歸。
本次沒有提高版本號、建立 Release 或推送遠端。

## 實作

### A：遷移與獨立性確認

- 0002 在同一個交易中先核對舊指紋；僅對原本有效的確認，於 SQL 執行後更新指紋。
- 保留原 `confirmed_at`，記錄 `migration_audit` 的版本、新舊指紋。
- 新增離線修復工具 `python -m workbench.repair_independence --database PATH --backup PATH`。
- 預設 dry-run：只複製 DB 與 WAL 至暫存區，不開啟來源資料庫、不複製／刪除來源 SHM；來源在複製期間變動即拒絕。
- 桌面鎖指向仍存活或無法確認已退出的程序時拒絕操作。套用需先關閉全部寫入程序，再加 `--apply --offline`。
- 修復必須同時符合：同一專案、備份確認原本有效、目前確認未更動、去除舊 metadata 後的 approved 指紋等於目前指紋。
- 套用前另存獨立備份；交易內再次核對，已有效或重複執行不更動。不會將未知／遭竄改的確認洗成有效。

### B：重開後接手的 worker

- Windows 檢查 `GetExitCodeProcess` 與建立時間；記錄 PID identity，辨識 PID 重用。
- 無權限等探測不明確時不當成死亡。舊版沒有建立時間的紀錄仍可檢查存活，但不能追溯補上原始 PID identity。
- 輪詢未由目前實例持有的 worker 時檢查存活；確認退出後重讀最後事件，已有終態則保留，否則寫 `control/terminal.json`。
- 心跳超時但程序仍在，只顯示警告。新訓練的 admission 也讀取更新後的狀態，不再被死 worker 永久占用。

### C：互動分割

- SAM2 模型與 GPU lease 一起保留；不再每次 API 推論完成就卸載。
- 其他 GPU 工作透過 `.gpu-requests` 明確公告等待；SAM2 完成本次推論後或背景檢查發現等待者時，先卸載再釋放 lease。
- 預設閒置 60 秒卸載，關閉服務也釋放。已退出程序的過期等待標記不算有效請求。
- GrabCut 使用獨立 CPU executor，不取得 SAM2 的互斥鎖或 GPU lease。

### D：安裝輸出與取消

- 持續讀取 stdout/stderr；支援 CR 更新、跨區塊 UTF-8 與錯誤位元組，保留最後 16 KiB 診斷內容。
- 取消仍終止程序樹；安靜輸出期間也會檢查取消。
- 安裝預設沒有總時間上限；預設連續 1,800 秒沒有新輸出才中止。
- `VISION_WORKBENCH_INSTALL_TIMEOUT`、`VISION_WORKBENCH_INSTALL_IDLE_TIMEOUT` 以秒設定；0 表示關閉該限制。
- 安裝子程序帶入 `PIP_PROGRESS_BAR=raw`、`PYTHONUNBUFFERED=1`，包括經 PowerShell bootstrap 啟動的 pip。
- 推論等非安裝用途仍可保留自己的總時間上限。

### E：事件與唯一快照

- `events.jsonl` 只在 status、phase、epoch 或 epoch metrics 改變時寫入完整事件。
- 每 batch 覆寫 `run.json`，不新增 `progress.json`；同一把跨程序鎖保護序號與快照寫入。
- 事件決定狀態；只合併序號較新且 status／epoch 相符的快照進度欄位，終態不接受回退。
- 截斷事件尾端會略過，新事件另起一行；不讓壞尾端吞掉下一筆有效事件。
- 不刪除或壓縮既有歷史事件，修正作用於後續寫入。

### 0004 與刪除／還原

- 新增 `history.annotation_hash` 及索引，填入現有 JSON 引用。
- 清除既有孤立附屬列，但保留 assets 或垃圾桶仍持有的紀錄；無 history 引用的 annotation blob 才回收。
- 永久刪除在同一交易中清理附屬列及引用；移到垃圾桶不以 DELETE trigger 連動清理。
- 沒有 shapes 的垃圾桶歷史不能當成標註版本還原，回明確業務錯誤 HTTP 400。
- 0004 升級前建立 `project.before-migration-0004.backup`，不覆寫 0003 的備份。

## 驗證與限制

最終驗證：Python 267 項（263 通過、4 個可選環境案例跳過）、JavaScript 52 項全通過；Ruff correctness 與差異格式檢查通過。
桌面完整 workflow 通過，訓練頁 13 個情境（含 1440／1024／760 寬度與即時監控）通過。

回歸案例集中於 `tests/test_regression_hardening.py` 與 API 測試：

- 舊資料結構、有效／無效確認、尚在 WAL 的內容、歷史去重、遷移重跑、0003／0004 備份並存。
- 修復 dry-run、三項證據檢查、套用前備份、重複執行。
- 真正的訓練 worker 經第一個工作台啟動、關閉、第二個接手後強制結束；可再啟動新 Run 並完成。
- PID 重用、尚存活但心跳逾時、死亡檢查後剛寫出 completed 的競態。
- 7 KB 狀態、31,300 次 batch 更新、100 次 epoch 結束：事件最多 200 筆、檔案小於 2 MB，最新 batch 可讀，舊快照不能蓋掉終態。
- 慢速輸出／CR／無效編碼、無輸出逾時、錯誤尾端、取消後孫程序終止。
- 假 runtime 驗證重複推論只載入一次、真實跨程序 GPU lease 請求觸發卸載、閒置卸載。
- 另一個程序占用 GPU 時，SAM2 工作等待而真正 GrabCut 可獨立完成。
- 永久刪除回滾、共享 blob、垃圾桶保留、HTTP 400。

桌面監控測試原本把已停止 R004 改回 running；已改為新增 R005，沒有為測試放寬終態保護。
完整桌面流程採暫存專案；不開啟使用者資料庫。SAM2 未安裝，因此沒有宣稱測過真實 SAM2／CUDA 推論效能；也沒有實際下載數 GB 套件。

## 部署注意

本輪不重啟正在執行的桌面版，不直接開啟 `data/` 內的資料庫。
兩份 0003 備份與四個 backup WAL／SHM 保留。**真實專案的 0004 遷移尚未在本輪執行**；下次關閉舊程序並啟動新版時會先備份再遷移。
修復工具不會自動對已遷移的專案套用；只有證據顯示確認確實因 0002 失效時才需要明確執行。
