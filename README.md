# Vision Workbench

Windows 桌面視覺資料工作台，整合影像採集、標註、審核與模型訓練。專案資料與模型保存在本機。

```text
採集／匯入 → 標註 → 審核 → 資料準備（分割／增強）→ 訓練 → 評估與預標註
```

## 核心功能

- **資料採集**：相機、圖片、影片畫格與螢幕擷取。
- **標註編輯**：同視窗切換內建編輯器、Labelme 與本機 CVAT，支援 SAM2／GrabCut 輔助分割。
- **資料管理**：人工審核、修訂追蹤、依來源分組的資料分割、Train-only 資料增強，以及可隨專案備份的固定資料版本。
- **模型訓練**：支援物件偵測、實例分割、語意分割與圖片分類，提供訓練曲線及模型比較。
- **資料交換**：COCO、YOLO、LabelMe、JSONL 與原生格式匯入／匯出，以及模型封裝匯出。

## 快速開始

需要 Windows 10／11（64 位元）與 Python 3.13。首次安裝需連線下載套件。

```powershell
git clone https://github.com/kongbai0123/vision-workbench.git
cd vision-workbench
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1
.\vision-workbench.bat
```

後續直接開啟 `vision-workbench.bat`。

訓練模型的選配環境可於「設定 → 模型與元件」安裝。SAM2 與 CVAT 的準備方式見[使用指南](docs/usage.md)。

## 文件

- [使用指南](docs/usage.md)：選配元件、標註流程、資料分割、模型評估與備份。
- [開發指南](docs/development.md)：本機開發、程式結構與測試。
- [版本紀錄](CHANGELOG.md)
- [問題回報](https://github.com/kongbai0123/vision-workbench/issues)

## 授權

本專案採用 [MIT License](LICENSE)。第三方元件與模型權重適用各自授權，選用前請查閱元件說明。
