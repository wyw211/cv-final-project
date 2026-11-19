# 正式訓練 checkpoint 設定

設定日期：2026-11-18。未啟動正式 ViSA 訓練。

## 保存規則

- 每完成 **5 個 epoch**，更新輸出資料夾的 `model_latest.pth`。
- 不依測試集成績選檔，不再建立 `model_best.pth`。
- 完成最後一個 epoch 時，更新 latest 並額外保存固定的 `model_final.pth`。
- `last_checkpoint` 指標在訓練中指向 latest，完成後指向 final。
- 正式評估使用事先固定的 final。final 在評估前保存，評估失敗也不會失去已完成的訓練結果。
- 只保留最新狀態與 final，不累積 24 份完整模型。依短測試檔案大小估計，兩份共約 4.4 GB 硬碟空間；原子寫入時需額外一份暫存空間。

120 epochs 的保存時間點為第 5、10、15、…、120 epoch；第 120 epoch 另有 final。若最後 epoch 不是 5 的倍數，也仍會保存 latest 和 final。

## 保存內容與恢復

- 模型參數與 buffers。
- SGD optimizer／momentum。
- Cosine 與 warmup scheduler。
- AMP GradScaler。
- 主程序 Python、NumPy、PyTorch CPU／CUDA RNG 狀態。
- 完成的 epoch 數、iteration 數；內部 epoch 索引仍為零起算。

例如第 5 epoch 保存後，resume 從第 6 epoch 開始，沿用已保存的 optimizer、LR 與 AMP 狀態。使用相同的 checkpointing 設定檔及輸出資料夾，加上 `--resume` 即可走既有恢復流程。

不保證恢復後的下一批資料逐位元相同：DataLoader worker、sampler 與預取佇列不在保存內容中。固定 seed 也不代表不同硬體／CUDA 演算法能逐位元一致。

## 寫入保護

先在同一個資料夾寫入暫存檔，完成 flush／fsync 後以 `os.replace` 替換目標。`last_checkpoint` 指標也採相同方式。

若序列化過程失敗，前一份可用 checkpoint 保留；此保護不等於保證硬碟損壞或所有斷電情境都能恢復。

## 已驗證

工具：`tools/verify_periodic_checkpoints.py`，5 項測試全部通過，使用小型 CPU 模型與不執行模型運算的 epoch loop：

1. 完整 120 epoch 保存時程、不提供 metric 仍保存、不依 metric 高低選檔，以及非整除最後 epoch。
2. 真實 `.pth` 檔案保存、全新物件載入、模型／momentum／scheduler／AMP／RNG 還原，以及第 5 epoch 後正確 resume。
3. 模擬寫入中途失敗，舊 latest 檔案內容與指標保持完整且仍能載入。
4. 模擬最終評估失敗，確認 final 已先保存。
5. 透過真正的 TrainerBase epoch loop 驗證 completed_iterations 沒有 off-by-one。

結果索引：`output/audit/latest_checkpoint_verification.json`。

另有先前真實 ViSA 50-step 短訓練的完整 checkpoint 復原測試，見 `SHORT_TRAINING.md`。本次沒有重跑 ViSA 訓練。
