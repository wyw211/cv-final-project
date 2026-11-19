# Activation checkpointing 驗證

日期：2026-11-18。GPU：RTX 5070 Ti，15.920 GiB；PyTorch 2.7.1+cu128。

後續已依使用者授權完成 50-step 完整短訓練（包含參數更新及 checkpoint 儲存／恢復），結果在 `SHORT_TRAINING.md`。下方記錄仍保留最初「不更新參數」的計算正確性與容量驗證。

## 結論

採用 checkpointing 後，batch 256 已在本機通過短期容量檢查，不必先降低 batch。這不等同完整訓練或長時間穩定性驗證，也尚未量出正式訓練的速度增幅。

## 修改範圍

- 新增 `MODEL.BACKBONE.ACTIVATION_CHECKPOINTING`，預設 False；原官方設定仍預設關閉。
- 使用 `configs/CARGO/visa_official_checkpointing.yml` 才開啟；保留 batch 256、輸入尺寸、專家數、loss 係數及官方資料協定。
- 重算區域：encoder 前 11 個 Transformer block（包含各 block 後的 token subtraction），以及最後的 Transformer／LayerNorm 分支。
- 使用 `use_reentrant=False`、`preserve_rng_state=True`；只在訓練模式且啟用梯度時使用。
- BatchNorm heads 不放入重算區域；不變更模型參數或 checkpoint 的 state_dict keys。
- 沒有修正官方共享 GCN／no_grad 等架構疑點，仍保留其既有行為。

## 計算正確性

腳本：`tools/verify_checkpointing.py`。相同模型初始權重、合成輸入（batch 8）與 seed，分別測 FP32 及 AMP FP16，兩者都通過：

- 全部 loss、backbone 五項輸出一致。
- 489 組有效梯度逐一比較，最大絕對差異 **0**，本次測試為逐位元相同。
- 239 組無梯度參數在開關前後保持相同；包含未使用與未路由到的參數，並非全部參數都應有梯度。
- 所有 buffers（含 BatchNorm 統計）一致。
- CPU／CUDA RNG 最終狀態一致，驗證 stochastic depth 重算沒有改變隨機序列。
- 沒有建立 optimizer 或更新參數；模型參數與初始值一致。

這些結果只對此次測試輸入成立，不代表所有 GPU／輸入都保證逐位元一致。

詳細結果：`output/audit/checkpointing_correctness.json`。

## Batch 256 容量

使用官方訓練 DataLoader 的 CARGO 訓練圖片，連續 3 個不同 batch，開啟 AMP；每批 64 個身分、每個身分 4 張。為本次檢查將 workers 設成 0，不將其耗時當成官方完整流程的性能測量。

| 項目 | 實測 |
| --- | ---: |
| 已完成前向／反向批次 | 3 |
| 預先常駐的 SGD momentum buffers | 1.027 GiB |
| PyTorch 配置顯存峰值（allocated） | 11.424 GiB |
| PyTorch allocator 保留顯存峰值（reserved） | 12.295 GiB |
| 模型參數更新 | 0 次 |
| checkpoint 寫入 | 無 |

momentum 在前向前已預留，因此涵蓋後續訓練迭代的常駐狀態，不只測尚未建立 optimizer 狀態的第一批。allocated／reserved 都不包含其他桌面程式等的全部顯存占用。

官方 GradScaler 初始 scale 在前兩批有溢位，依正常機制降低後，第三批梯度有限。測試沒有呼叫 `scaler.step()` 或 `optimizer.step()`，因此所有批次都沒有更新權重。FP32／AMP 小 batch 的正確性比較則逐一確認了有限梯度。

本次容量檢查沒有執行 optimizer 更新、梯度裁切、排程、checkpoint 存取及最終評估；正式開跑前仍需要隔離的完整短訓練流程驗證。不能把單次或三次冷啟動耗時直接外推成完整訓練時間。

詳細結果：`output/audit/batch_256_checkpointing_cargo_memory.json`。

## PyTorch 2.x 梯度 buffer 相容性修正

官方 FastReID optimizer 使用 contiguous parameters／gradient buffers。新版 `optimizer.zero_grad()` 預設 `set_to_none=True`，會讓 optimizer 的 flat-buffer 梯度變成 None，但模型參數自己的梯度仍存在，兩者連結失效。

已將 `fastreid/engine/train_loop.py` 的 SimpleTrainer、AMPTrainer 改成 `zero_grad(set_to_none=False)`，保留原程式依賴的 buffer 行為，不改 loss、LR 或模型。

獨立 CPU 小模型測試確認：現代預設會造成「模型有梯度、optimizer buffer 沒梯度」；明確 False 後兩者都有梯度。此次 CARGO 容量測試亦確認 buffer pointer 有效、每組 optimizer buffer 都有梯度。沒有為這項測試執行參數更新。

詳細結果：`output/audit/optimizer_buffer_compatibility.json`。

## 重跑

```powershell
# 正確性：不更新參數
..\.venv\Scripts\python.exe tools/verify_checkpointing.py

# CARGO 容量：不更新參數
..\.venv\Scripts\python.exe tools/check_batch_memory.py --batch-size 256 --activation-checkpointing --passes 3 --real-training-data
```

後續若進行速度測試，應先暖機，再用相同 batch 比較開關前後，並將 optimizer 更新等實際訓練工作納入。這次尚未測量原先粗估的 20–50% 額外時間。
