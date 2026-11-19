# CARGO 訓練與辨識評估結果

2026-11-18 台灣時間 14:19 完成全部流程。120 epochs、24,000 iterations；四個協定皆已評估。訓練與評估程序已結束，未見 traceback。

| 協定（依官方程式） | Rank-1 (%) | mAP (%) |
| --- | ---: | ---: |
| ALL | 68.91 | 64.59 |
| A→G / CARGO_AG | 69.15 | 68.16 |
| A↔A / CARGO_AA | 77.50 | 66.99 |
| G↔G / CARGO_GG | 85.71 | 82.02 |

數字取自原生 evaluation table，精度為小數點後兩位。完整 Rank-5／10、mINP 及來源精度記錄於 `logs/CARGO/ViSA_official_checkpointing/evaluation_results.json`。

固定採第 120 epoch final checkpoint；沒有依測試集成績挑選其他模型。保留官方程式 λ=0.0001，CARGO_AG 為單向 A→G，相關論文歧義見 `CARGO_SETTINGS_AUDIT.md`。

- 模型：`logs/CARGO/ViSA_official_checkpointing/model_final.pth`，約 2.21 GB，包含訓練狀態。
- 設定快照：同資料夾 `config.yaml`。
- 原始訓練／評估紀錄：`train.stdout.log`；訓練逐步指標：`metrics.json`。
- 原始程式雜湊與修改：`source_manifest.json`、`source_changes.patch`。
- 訓練運算時間：10 小時 57 分 20 秒。
- 含評估、存檔等 hooks 的總時間：12 小時 5 分 9 秒。

這是 CARGO 的重現 baseline。效率量測（參數口徑、GFLOPs、推論延遲）、改進方法、消融與成功／失敗檢索視覺化仍未完成；目前沒有啟動新的訓練。
