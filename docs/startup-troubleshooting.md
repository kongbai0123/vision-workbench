# 啟動疑難排解

[返回 README](../README.md)

先確認問題出現在 Windows 警告、安裝階段或桌面啟動後。以下 PowerShell 指令均在解壓後的專案根目錄執行。

## Windows 顯示不明發行者或已保護您的電腦

從瀏覽器下載的 ZIP 解壓後，檔案可能保留 `Zone.Identifier` 的 `ZoneId=3` 網際網路標記。啟動腳本未具備受 Windows 信任的發行者簽章，因此可能觸發來源或信譽警告。這與 Python 連線下載套件時的 SSL 憑證錯誤不同。

先確認檔案來自本專案可信的發布來源，再對啟動流程中已檢查的檔案解除封鎖：

```powershell
Unblock-File -LiteralPath .\vision-workbench.bat, .\bootstrap.ps1, .\launch.ps1, .\scripts\runtime.ps1
```

也可在個別檔案的「內容 → 一般」檢查並使用「解除封鎖」。這會移除指定檔案的下載標記，不會替程式取得數位簽章。若仍顯示組織政策、Smart App Control 或其他封鎖，請保留完整提示並由系統管理者檢查政策；這套流程不會變更全機安全設定。

首次下載時的 Windows 提示發生在腳本執行之前，因此程式內的防呆無法保證消除此提示。正式散布可另規劃由可信任發行者簽署的執行檔或安裝程式；仍需符合 Windows 的信譽與管理政策，不能保證簽章後完全沒有警告。

參考：[Microsoft SmartScreen](https://learn.microsoft.com/en-us/windows/security/operating-system-security/virus-and-threat-protection/microsoft-defender-smartscreen/)、[Unblock-File](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/unblock-file)。

## 一次收集啟動診斷

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1 -Diagnose
```

此模式不安裝套件、不啟動 GUI、不連線、不開啟相機，也不執行模型。它檢查實際 `.venv` 的 Python／位元數、固定套件版本、核心 DLL 匯入與啟動腳本的下載標記。環境不完整時會以非零狀態碼結束；即使環境尚未建立也能產生報告。

報告為 `data/logs/startup-check.json`；可加上 `-DataRoot <資料目錄>`，將此次報告寫入該目錄的 `logs/`。報告不收集圖片、專案資料、環境變數或帳號憑證，但可能包含本機使用者路徑，分享前請檢查。

## 修復執行環境

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1 -Repair
```

此模式只修復並驗證環境，不開啟桌面。不相容或損壞的 `.venv` 會先改名保留，再以已安裝的 64 位元 Python 3.13 建立新環境；缺少該 Python 時會先停止，不會搬走舊環境。既有相容的 Python 3.12–3.14 環境會繼續使用。環境若是連結，請先人工確認目標，不會自動替換。

修復前先關閉工作台。同一份專案的安裝流程會使用檔案鎖，避免重複點擊同時修改環境；若另一個安裝正在執行，請等待它完成再重試。

安裝中斷可重跑相同指令。套件安裝成功後仍須通過版本與核心匯入檢查，才會顯示準備完成。預設不安裝 AI／訓練環境，也不下載模型權重。

一般雙擊 `vision-workbench.bat` 時會先執行相同檢查，失敗則自動嘗試修復一次；修復失敗會顯示原因及紀錄位置並停止。若只想嘗試啟動且禁止自動安裝：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1 -NoAutoRepair
```

## 依錯誤定位

| 現象 | 檢查與處理 |
| --- | --- |
| Python 3.11、32 位元或版本不支援 | 執行 `-Repair`；安裝腳本會尋找 Python 3.13 並保留舊環境 |
| `ModuleNotFoundError` 或套件版本不符 | 執行 `-Repair`；僅存在 `pythonw.exe` 不代表安裝完成 |
| `CERTIFICATE_VERIFY_FAILED` | 查看 bootstrap 紀錄，檢查 Windows 日期時間、企業代理及憑證部署；由管理者配置可信憑證，不使用停用 TLS 驗證的方式修復 |
| 下載逾時、無法解析主機或找不到套件 | 確認網路及公司代理允許 PyPI／套件來源；檢查錯誤中的 Python 版本與套件名稱，再重跑安裝 |
| Qt／核心 DLL 匯入失敗或子程序異常退出 | 查看診斷的模組名稱與退出碼，確認 Windows 封鎖提示；重裝對應可信套件。OpenCV／mss 的選配警告不會阻止核心介面啟動 |
| 有程序但沒有畫面 | 查看當次 `startup-*.stderr.log` 與 `workbench.log`；確認是否已有視窗或鎖定提示 |
| 網頁顯示「啟動憑證無效」 | 本機 session 已更換，從新啟動的桌面或本次 `--serve` 輸出的入口進入，不沿用舊書籤；不要分享包含 session 的網址 |

`py --version` 或全域 `python --version` 不一定是工作台真正使用的環境。直接檢查：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

## 紀錄與回報

- `data/logs/startup-check.json`：最近一次環境診斷。
- `data/logs/bootstrap-*.log`：安裝過程、失敗套件及錯誤原因。
- `data/logs/startup-*.stderr.log`／`startup-*.stdout.log`：每次透過啟動器開啟桌面的程序輸出。
- `data/logs/workbench.log`：應用程式例外與執行資訊。

指定 `-DataRoot` 時，啟動紀錄與應用資料會使用該目錄；安裝紀錄仍位於專案的 `data/logs/`。紀錄與舊環境備份不會自動刪除，可在確認修復後自行歸檔。

回報請附上 Windows 提示文字、啟動方式、`startup-check.json` 及當次錯誤末段。先移除個人路徑、代理伺服器帳密、session 網址與私人資料。這份診斷用來驗證啟動環境，不代表相機、AI 模型或訓練已通過驗證。
