# 組員導覽：目前成果與重現方式（2026-11-26）

這份文件說明我們在官方 ViSA 程式（[Cat-Zero/ViSA](https://github.com/Cat-Zero/ViSA)）上做了什麼、結果如何，以及每個檔案的用途。
完整的實驗過程、數字與失敗的嘗試，請看 [NEXT_IMPROVEMENTS.md](NEXT_IMPROVEMENTS.md)（依時間順序記錄）。
每個實驗的目的與想回答的問題，請看 [EXPERIMENT_PURPOSES.md](EXPERIMENT_PURPOSES.md)。

## 1. 一句話總結

把 ViT 的預訓練權重從 ImageNet（jx）換成行人專用的自監督權重 **PersonViT（LUPerson）**，每組 expert 從 8 個減為 **4 個**，
測試時再加上不需訓練的 **αQE（alpha query expansion，k=2，α=3）**。
CARGO 測試集 ALL mAP 從 64.59（我們復現的 baseline）提高到 **73.38**，參數量少 31%，計算量和速度不變。

## 2. 最終結果（CARGO 測試集，Rank-1 / mAP，%）

| 方法 | ALL | A↔G | G↔G | A↔A |
|---|---|---|---|---|
| ViSA 論文（Table 1） | 70.51 / 65.46 | 71.28 / 69.00 | 88.39 / 83.90 | 82.50 / 67.78 |
| 我們復現的 baseline（官方程式，120 epoch） | 68.91 / 64.59 | 69.15 / 68.16 | 85.71 / 82.02 | 77.50 / 66.99 |
| 復現 baseline + αQE | 73.40 / 69.02 | 73.40 / 72.15 | 87.50 / 84.54 | 72.50 / 68.52 |
| 最終模型（PersonViT + 4 experts） | 74.04 / 69.90 | 73.40 / 72.24 | 86.61 / 84.13 | 80.00 / 69.70 |
| **最終模型 + αQE（最終方法）** | **77.88 / 73.38** | **74.47 / 75.79** | **91.07 / 89.03** | **80.00 / 73.12** |
| 最終模型 + k-reciprocal re-ranking（參考） | 79.81 / 74.32 | 73.40 / 73.72 | 92.86 / 87.19 | 72.50 / 66.60 |

- A↔G 欄位是官方程式的 `CARGO_AG`（單向 A→G），與論文的定義可能不同，見 [CARGO_SETTINGS_AUDIT.md](CARGO_SETTINGS_AUDIT.md)。
- 測試集只做最後評估：所有設定都先在驗證集決定，測試集每個模型只評估一次。
- A↔A 測試集只有 40 個有效 query（1 個 query = 2.5 點 R1），R1 很粗，看 mAP 比較可靠。

**效率**（RTX 5070 Ti，fp32，輸入 256×128）：

| 模型 | Params | GFLOPs | 延遲（batch 1） |
|---|---:|---:|---:|
| 復現 baseline（8 experts） | 275.6 M | 42.06 | 約 15 ms |
| 最終模型（4 experts） | 190.4 M（−31%） | 42.06 | 約 14.5 ms |

每張圖只走 top-2 experts，所以 expert 減半省的是參數和記憶體，不是計算量。αQE 在 GPU 上處理整個測試 gallery（約 5 萬張）只要 0.41 秒。

## 3. 我們做了什麼（含沒用的嘗試）

| 項目 | 結果 | 說明 |
|---|---|---|
| 復現官方 ViSA | ALL mAP 64.59（論文 65.46） | [CARGO_BASELINE_RESULTS.md](CARGO_BASELINE_RESULTS.md) |
| 從訓練集切出驗證集（250 個 ID） | — | `fastreid/data/datasets/cargo_val.py`；所有設計決策都只看它 |
| 修正 expert 的輸入（v1） | jx 上 +1.4，PersonViT 上無效 | [EXPERT_INPUT_FIX.md](EXPERT_INPUT_FIX.md) |
| **換成 PersonViT 權重**（預訓練的 ViT 用 0.2 倍學習率） | **驗證集 ALL mAP +6.9，最大的進步** | `tools/convert_personvit.py`；另跑「jx + 同樣 0.2 倍學習率」對照，確認進步來自權重 |
| **expert 8 → 4** | 準確度不變，參數 −31% | `EXPERTS: 4` |
| Prototype contrastive loss（PCL） | 在 PersonViT 上無效 | `pilot_val_personvit_pcl.yml` |
| 夜間增強 L1 / 亮度校正 L2 / 光照一致性損失 L3 | 都無效 | 夜間影像缺的是資訊，不是亮度；見 NEXT_IMPROVEMENTS §13 |
| **αQE 後處理** | **驗證集 A↔A mAP +5.0，夜間 +7.0** | `tools/posthoc_val.py`；比 re-ranking 穩定，不傷 A↔A |
| 翻轉 TTA、依光照或攝影機做特徵去中心化 | 幾乎無效 | NEXT_IMPROVEMENTS §14 |

**A↔A 的弱點**：驗證集上夜間 A↔A mAP 76.75，白天 92.64；測試集剩下的 8 個 A↔A R1 失敗全部是夜間 query。這可以寫進報告的限制與分析。

## 4. 檔案導覽

| 路徑 | 內容 |
|---|---|
| `NEXT_IMPROVEMENTS.md` | 全部實驗紀錄（時間順序，含數字） |
| `EXPERIMENT_PURPOSES.md` | 每個實驗的目的、想回答的問題與結論（寫報告用） |
| `results/figures/` | 檢索視覺化：`CARGO_*_final_aqe.png`（成功與失敗範例）、`compare_CARGO_*.png`（baseline 對照我們）。範例用 seed 0 隨機挑選，不是手挑 |
| `results/test/aqe_test_results.json` | 第 2 節測試數字的原始輸出 |
| `results/efficiency/` | Params、GFLOPs、延遲（兩次、順序對調） |
| `results/val/` | 驗證集後處理比較，以及逐 query 的夜間／白天分析 |
| `results/runs/` | 兩個 120 epoch 正式訓練的設定檔與紀錄 |
| `configs/CARGO/visa_personvit_experts4.yml` | **最終方法的訓練設定** |
| `configs/CARGO/pilot_val_*.yml` | 20 epoch 驗證集 pilot（每個嘗試一個檔） |
| `tools/convert_personvit.py` | 把 PersonViT checkpoint 轉成 ViSA 可載入的格式 |
| `tools/eval_test_aqe.py` | 測試集評估（未處理與 αQE 兩種） |
| `tools/measure_efficiency.py`、`tools/aqe_gpu_timing.py` | 效率測量 |
| `tools/vis_retrieval.py` | 產生檢索視覺化 |
| `tools/posthoc_val.py`、`tools/diag_per_query.py` | 驗證集的後處理比較與夜間／白天分析（拒絕讀取測試集） |
| `tools/run_cargo_pilot.sh` | 依序跑 pilot：`bash tools/run_cargo_pilot.sh light_l1 light_l2` |
| `tools/run_full_personvit_experts4.sh` | 最終方法 120 epoch 正式訓練（可中斷續跑） |

程式改動（`fastreid/`）：新增 `SOLVER.PRETRAINED_LR_FACTOR`、`MODEL.BACKBONE.EXPERTS`、PCL、光照相關選項等設定，預設值都等於官方行為，
所以官方設定檔 `configs/CARGO/visa.yml` 的結果不受影響。

## 5. 如何重現

環境請見 [INSTALL.md](INSTALL.md) 和 [LOCAL_SETUP.md](LOCAL_SETUP.md)。資料集放在 `datasets/CARGO/`。`datasets/`、`weights/`、`logs/` 不在 git 裡。
以下指令在 `ViSA/` 資料夾內，用 Git Bash 執行；Python 路徑請換成自己的。

1. **下載並轉換 PersonViT 權重**（ViT-B/16，LUPerson，1.51 GB）：
   從 HuggingFace `lakeAGI/PersonViT` 的 `vitb.lup.256x128.wopt.csk.4-8.ar.375.n8/checkpoint0240.pth` 下載到 `weights/personvit_vitb_lup/`，然後執行：
   ```bash
   python -I tools/convert_personvit.py weights/personvit_vitb_lup/checkpoint0240.pth weights/personvit_vitb_lup_teacher_e240_s12.pth
   ```
   轉換後的 SHA256 記錄在 `results/runs/final_personvit_experts4_120ep/pretrain_sha256.txt`。
2. **訓練最終模型**（120 epoch，RTX 5070 Ti 約 9.5 小時）：
   ```bash
   bash tools/run_full_personvit_experts4.sh
   ```
   或 `python tools/train_net.py --config-file configs/CARGO/visa_personvit_experts4.yml`。輸出在 `logs/CARGO/ViSA_personvit_experts4/`。
3. **測試集評估（含 αQE）**：
   ```bash
   python tools/eval_test_aqe.py logs/CARGO/ViSA_official_checkpointing logs/CARGO/ViSA_personvit_experts4
   ```
4. **效率與視覺化**：
   ```bash
   python tools/measure_efficiency.py logs/CARGO/ViSA_official_checkpointing logs/CARGO/ViSA_personvit_experts4
   python tools/vis_retrieval.py logs/CARGO/ViSA_official_checkpointing logs/CARGO/ViSA_personvit_experts4
   ```

訓練好的模型檔（每個約 1–3 GB）沒有放進 git。需要的話可以另外用雲端硬碟分享。

## 6. 還沒做的

- 期末報告（CVPR 2027 模板，最多 8 頁）與口頭報告投影片（12/16、12/23）。
