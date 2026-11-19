# CARGO 50-step 短訓練結果

日期：2026-11-18。這是獨立測試模型，不是正式 baseline，不作為論文準確率結果。

## 結果

| 項目 | 結果 |
| --- | --- |
| GPU | RTX 5070 Ti，16 GB |
| 資料 | CARGO 訓練集；未執行任何測試集評估 |
| 設定 | batch 256；64 個身分 × 4 張；checkpointing；AMP；官方 SGD、LR 與 warmup |
| DataLoader workers | 8（官方值） |
| 已執行 iteration | 50 |
| 實際權重更新 | 46 次 |
| AMP 溢位而正常跳過 | 第 1、2、9、42 步，共 4 次 |
| Loss | 全部有限，無 NaN／Inf loss |
| 最後連續更新 | 第 43–50 步，共 8 步 |
| 配置顯存峰值（allocated） | 11.447 GiB |
| 平均每步耗時 | 1.552 秒 |
| 每步耗時中位數／P95 | 1.539／1.639 秒 |
| 120 epochs 純訓練粗估 | 10.35 小時，另加最終評估、存檔及其他停頓 |

耗時使用第 11–40、44–50 步共 37 筆，排除起始及恢復後暖機。包含官方 AMPTrainer 的前向、反向、optimizer／GradScaler 更新、metric 收集與 warmup hook；以 CUDA synchronize 包住計時。

估算為 `1.552 × 200 iterations/epoch × 120 epochs`。這只是早期 warmup 的短樣本，不能保證長時間速度不變。未直接量出 checkpointing 比原版增加的比例，因為原版 batch 256 在本機 OOM。

## Checkpoint 儲存與恢復

- 第 40 步儲存 `short_iteration_40.pth`，約 2.21 GB（2.05 GiB）。
- 存檔約 2.60 秒，載入約 1.64 秒；不含新建 trainer／worker 啟動時間。
- 舊 trainer 已釋放，使用全新模型、optimizer、scheduler、AMP scaler 重新載入。
- 模型、SGD momentum、cosine／warmup scheduler、AMP scaler、已保存 RNG 的 state hash 全部一致。
- 從第 41 步繼續到第 50 步，確認可恢復後更新。
- DataLoader sampler／worker 預取佇列沒有還原，因此不宣稱暫停前後的「下一批資料與增強」逐位元連續相同。

原官方 DefaultTrainer 未保存 AMP scaler，本次已將它加入正式 checkpointer 的 checkpointables，避免恢復時丟失縮放狀態。RNGState 則由短測試工具註冊。

## AMP 驗證判準

四次未更新都對應 GradScaler 降低 scale，屬其遇到非有限梯度時的保護行為，沒有無法解釋的權重停滯；所有 loss 有限。最初的測試斷言要求最後 10 步完全不能跳過，但第 42 步遇到正常 AMP 溢位。已保留原斷言報告，並依每步記錄重新驗證「所有跳過都有 scale 降低、最後 5 步連續更新、checkpoint state 完全一致」。沒有為取得通過結果重跑或調整模型超參數。

## 檔案

資料夾：`logs/bringup_20261118_015514_c822bb/`

- `config.yml`：起始設定快照。
- `iterations.jsonl`：全部 50 步的 loss、LR、AMP scale、更新狀態及耗時。
- `report.json`：最終摘要與恢復狀態 hashes。
- `initial_strict_validation_report.json`：原先過於嚴格的斷言結果，保留供追溯。
- `short_iteration_40.pth`：只供短測試恢復使用，不能當作正式 baseline。

工具：`tools/short_train.py`。重跑會建立新的獨立資料夾，不覆蓋先前結果。

## 範圍限制與正式開跑前事項

這次沒有跑完整 epoch、完整 cosine 排程、高 LR 階段或辨識評估，也沒有根據短期 loss 調參。正式 120 epochs 尚未啟動。

此次直接呼叫 Checkpointer.save 測試儲存／恢復；官方 PeriodicCheckpointer 另有 best-only 邏輯，註解掉了固定週期的 model snapshot。正式訓練前仍應明確設定「按週期保存 latest、依固定最終 checkpoint 評估」，不能把本次手動存檔成功當成官方每 5 epochs 都會保存最新狀態的證明。

後續已依使用者要求修正上述定期保存邏輯並通過 5 項測試，詳見 `CHECKPOINT_POLICY.md`；保留本段作為短訓練當時的範圍記錄。
