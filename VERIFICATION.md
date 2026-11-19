# 不訓練驗證結果

驗證日期：2026-11-18（Asia/Taipei）。結果：所有已執行項目通過。

## 資料與權重

| 項目 | 結果 |
| --- | --- |
| 官方 CARGO 壓縮檔與解壓檔清單 | 完全一致 |
| 圖片大小、CRC、完整解碼、RGB 格式、檔名及相機標籤 | 全部 102,475 張通過 |
| 訓練集 | 51,451 張，2,500 個身分 |
| Query | 312 張，149 個身分 |
| Gallery | 50,712 張，2,500 個身分 |
| 訓練與測試身分交集 | 0 |
| 訓練與測試之間位元組完全相同的圖片 | 0（以 SHA256 比對） |
| Query 身分是否存在於 Gallery | 全部存在 |
| ViT-base 預訓練權重 SHA256 | 符合官方檔名的雜湊及先前保存的完整雜湊 |

測試圖片僅讀取檔案以驗證完整性，沒有送入模型。雜湊比對只能排除位元組完全相同的重複圖片，不能排除經過重新壓縮或裁切的近似圖片。身分檢查依資料集標籤進行，無法保證人工標註本身沒有錯誤。

## 官方 DataLoader 與模型

- 官方訓練 DataLoader 讀取 3 個 batch，每個尺寸為 `[16, 3, 256, 128]`，成功搬到 GPU。
- 每個 batch 有 4 個身分、每個身分 4 張圖片；影像值有限且落在 0–255。
- 每筆圖片路徑、重新編碼的身分／相機標籤、Aerial／Ground 視角均能對應原始訓練資料。
- 使用其中一個訓練 batch 執行 ViSA 推論，輸出為 `[16, 1536]`，所有數值有限。
- 推論前後，全部模型參數與 buffers 的 SHA256 完全一致；沒有建立梯度或 optimizer。
- 本次沒有執行反向傳播、參數更新、正式訓練或測試集辨識評估。

## 指標驗證與修正

人工案例涵蓋：正確答案第一名、第二名、多個正確答案、同身分同相機排除、沒有正確答案的 query 跳過、多 query 平均，以及所有 query 無效時拒絕計算。Rank-1、AP 與 INP 都與手算結果一致。另通過 cosine／平方 Euclidean 距離測試。

完整 `ReidEvaluator` 也使用人工特徵通過測試；人工設計的完美檢索案例得到 Rank-1／5／10、mAP、mINP 全部 100%。**這些數字是程式正確性測試，不是 CARGO 模型成績。**

發現並修正兩項問題：

1. `fastreid/evaluation/rank.py` 原本含有 `all_cmc[0][0] = 0`，會強制把第一個有效 query 的 Rank-1 設成 0。修正前，單一完美 query 被算成 Rank-1 0%；移除該行後為正確的 100%。
2. `fastreid/evaluation/reid_evaluation.py` 在 Cython 擴充不存在時強制呼叫 `make`，導致目前 Windows 環境初始化失敗。現在缺少 `make` 時使用既有的 Python 評估路徑，並明確記錄提示。

目前驗證的是 Python 評估路徑；未編譯或驗證 Cython 路徑。後續報告應記錄這些修改，避免把不同版本的指標實作混用。

## 重跑方式

在 `C:\Project\CV final project\ViSA` 執行：

```powershell
..\.venv\Scripts\python.exe tools/verify_no_training.py
```

只重跑人工指標案例：

```powershell
..\.venv\Scripts\python.exe tools/verify_no_training.py --checks synthetic_metrics
```

完整機器可讀結果位於 `output/verification_no_training.json`。部分重跑會保留其他檢查結果與其時間戳。

## 尚未驗證的事項

- 正式訓練的 optimizer、排程、完整 epoch、長時間穩定性與 checkpoint 恢復。
- 論文與補充資料的全部訓練設定及評估協定是否已完整對齊。
- CARGO Rank-1／mAP 或論文結果能否重現。
- LAGPeR 尚未取得，因此未驗證。
