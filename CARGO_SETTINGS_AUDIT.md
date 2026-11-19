# CARGO：論文與官方設定核對

設定核對日期：2026-11-18；執行狀態更新：2026-11-19。

**CARGO baseline 已完成：啟用 activation checkpointing，保留 batch 256，完成正式 120 epochs（24,000 iterations）訓練，以及 ALL、A→G、A↔A、G↔G 四個協定的最終評估。** 全部流程於 2026-11-18 台灣時間 14:19 結束，結果見 [CARGO_BASELINE_RESULTS.md](CARGO_BASELINE_RESULTS.md)。本文件同時記錄設定核對與先前容量測試，歷史上的 OOM 不代表目前仍無法執行。

## 採用原則

使用者已確認的決定：

- **MoE λ 沿用官方程式的 0.0001**，不改成論文分析最佳值 0.001。報告及實驗紀錄保留此差異。
- **跨視角評估沿用官方 CARGO_AG 的 Aerial query → Ground gallery（A→G）**，不自行改成雙向平均。報告應註明論文使用 A↔G 名稱，但本實作為單向 A→G。
- **已完成正式 120 epochs 訓練與四協定評估**。未啟用 activation checkpointing 的 batch=256 曾在本機 OOM；之後加入 checkpointing，通過輸出／梯度一致性、3 批容量檢查、50-step 完整短訓練及 checkpoint 儲存／恢復驗證，再完成正式訓練。沒有降低 batch 或修改模型架構。驗證細節見 [CHECKPOINTING.md](CHECKPOINTING.md)、[SHORT_TRAINING.md](SHORT_TRAINING.md)；正式結果見 [CARGO_BASELINE_RESULTS.md](CARGO_BASELINE_RESULTS.md)。

以官方單 GPU README 指令及其實際程式行為建立 baseline。論文和程式不一致時明確記錄，不自行更改程式來追求論文數字。這是「官方程式重現設定」，不宣稱所有細節與論文完全一致。

核對時採用的官方來源 commit 為 `22ff906c28c63144d382c160d097b0de6f4f0510`；本機版本以此為基礎加入 Windows／PyTorch 相容性、評估、checkpointing 與存檔修正，不表示目前本機 HEAD 仍等於該 commit。資料與評估驗證見 [VERIFICATION.md](VERIFICATION.md)，存檔規則見 [CHECKPOINT_POLICY.md](CHECKPOINT_POLICY.md)。

## 來源

- [官方 README](https://github.com/Cat-Zero/ViSA/blob/main/README.md)
- [官方 CARGO YAML](https://github.com/Cat-Zero/ViSA/blob/main/configs/CARGO/visa.yml)
- 主論文：`references/ViSA_CVPR2026.pdf`，§4.1（印刷頁 4388–4389）、Table 1、§4.4。
- 補充資料：`references/ViSA_CVPR2026_supplemental.pdf`，§8、§9、Table 5。
- 完整預設值：`fastreid/config/defaults.py`；最佳化器／排程：`fastreid/solver/build.py`、`fastreid/engine/hooks.py`。

主論文 PDF SHA256：`6cabd0f831146f57b6fa56b4d64fefb311b2850aac618c8612bf10fa23a3824d`。
補充資料 PDF SHA256：`33329898e43e3ed5a6e949f5435a91bd82d09afff1d9c9a7ef0c051596c8cb88`。

## 核心設定對照

| 項目 | 論文／補充資料 | 官方程式實際設定 | 本次採用 |
| --- | --- | --- | --- |
| GPU | 單張 RTX 4090 | 支援單 GPU | 單張 RTX 5070 Ti；記錄硬體差異 |
| Backbone | ImageNet 預訓練 ViT | ViT-base，指定 JX 權重 | 官方權重與模型 |
| 輸入 | 256 × 128 | 訓練／測試同尺寸 | 相同 |
| Batch | 256，64 個身分 × 4 張 | YAML 128；單 GPU README 覆蓋為 256 | **256** |
| Activation checkpointing | 實驗設定未明列 | 原版未啟用 | **本機啟用，保留相同 batch 與模型；已驗證並完成訓練** |
| 取樣 | 每身分 4 張 | NaiveIdentitySampler，NUM_INSTANCE=4 | 相同 |
| Epochs | 120 | 120 | 相同 |
| Optimizer | SGD with momentum | SGD；momentum=0.9，Nesterov=False | 官方程式 |
| 初始 LR | 0.008 | 0.008；warmup 時較低 | 官方程式 |
| 最低 LR | 0.0000016 | 0.0000016 | 相同 |
| 排程 | cosine | warmup + cosine | 官方程式 |
| 專家 | 各型別 8，選 top-2 | shared／aerial／ground 各 8，top-2 | 相同 |
| 負載平衡 λ | §4.4、補充 §8 最佳值 0.001 | **寫死 0.0001** | **0.0001，保留差異** |
| 指標 | Rank-1、mAP | 同時輸出 Rank-1/5/10、mAP、mINP | 主要報告 Rank-1、mAP |

論文 ImageNet 描述沒有在實驗段落中明確指認官方權重檔的完整預訓練變體；以官方指定的 `jx_vit_base_p16_224-80ecf9dd.pth` 為準，不自行替換其他 ViT。

## YAML 未完整列出的有效設定

這些由程式預設值合併而來，不能只閱讀 `visa.yml`：

| 項目 | 有效值／行為 |
| --- | --- |
| 正規化 | mean=std=[127.5,127.5,127.5]，原始像素範圍 0–255 |
| Patch | kernel 16 × 16，stride 12 × 12 |
| DropPath | 0.1；一般 dropout、attention dropout 為 0 |
| 局部 token／圖鄰域 | PROMPT_LEN=16，GRAPH_TOP_K=64 |
| PROMPT_DEPTH | YAML 為 2；目前只存入屬性，未用於控制 forward 深度，不能將修改此值視為有效消融 |
| 訓練增強 | 水平翻轉 p=0.5；padding=10 後裁切；random erasing p=0.5 |
| 其他增強 | 獨立 RandomResizedCrop、color jitter、rotation、AutoAugment 等未開啟 |
| Weight decay | 一般權重 0.0001；normalization 參數及 bias 為 **0.0005** |
| Warmup | linear，2,000 iterations，初始 factor=0.01（LR=0.00008） |
| 迭代數 | 51,451 // 256 = 200 iterations/epoch；120 epochs 共 24,000 iterations |
| Cosine | 排程建構時 T_max=110 epochs（120 減去 ceil(2000/200)）；實際切換由 hook 執行 |
| AMP | 開啟；官方 AMPTrainer 使用預設 GradScaler，與先前 smoke test 的初始 scale=128 不同 |
| 梯度裁切 | 開啟，L2 norm，clip value=5；實作按參數／contiguous buffer 裁切，非整個模型單一 global norm |
| Workers | 訓練 8；官方測試 loader 預設 4；先前本機 bring-up 的 0 不屬於官方設定 |
| 測試 batch | 384 |
| 測試距離 | cosine；無 test-time flip、AQE、reranking |
| Seed | `default_setup` 寫死 `seed_all_rng(0)`；先前 smoke test 用 42，不是正式 baseline seed |
| cuDNN benchmark | True；固定 seed 並不保證逐位元重現 |
| 身分分類器 | 從訓練集自動設定為 2,500 類 |
| Checkpoint／評估 | 每 5 epochs 儲存；第 120 epoch 評估。最終 checkpoint 事先固定，禁止依測試分數挑選 |

`WEIGHT_DECAY_BIAS` 在 defaults 定義時已取值 0.0005；之後 YAML 覆蓋一般 weight decay 並不會讓它連動變成 0.0001。

## CARGO 評估協定核對

下表是以標籤、數量與跨相機正樣本建立的協定核對表，本身不包含檢索成績。正式訓練後已完成四個指定協定的最終評估，實測 Rank-1／mAP 另列於 [CARGO_BASELINE_RESULTS.md](CARGO_BASELINE_RESULTS.md)。

| 論文欄位／程式類別 | Query 張數 | Gallery 張數 | 有跨相機正樣本的 query |
| --- | ---: | ---: | ---: |
| ALL / CARGO | 312 | 50,712 | 312 |
| A↔G / CARGO_AG | 134 | 32,268 | 94 |
| A↔A / CARGO_AA | 134 | 18,444 | 40 |
| G↔G / CARGO_GG | 178 | 32,268 | 112 |
| 額外的 G→A / CARGO_GA | 178 | 18,444 | 66 |

重要：論文使用 A↔G 符號，但官方 `CARGO_AG` 實作為 **Aerial query → Ground gallery**，並非將兩個方向平均。補充 Figure 6 圖說也使用 A→G。這不能單獨證明 Table 1 的成績必定來自同一方向；官方 issue [#6](https://github.com/Cat-Zero/ViSA/issues/6) 也詢問此差異，核對時尚無回覆。本次按官方程式記錄此方向，不自行加入 G→A 來改變定義，亦不宣稱已消除協定歧義。

`CARGO_GA` 雖有實作，但不在官方 `DATASETS.TESTS` 內，故不加入官方 baseline 主表。沒有跨相機正樣本的 query 按既有評估程式跳過，不能改分母或補造匹配。

正式訓練只用 `DATASETS.NAMES=("CARGO",)` 的 51,451 張圖。AA／GG dataset 類別內也會建構其 view-filtered train 清單，但它們在官方設定中僅用於測試，其 train 清單不參與此次模型訓練。

## 官方來源中的差異與未決點

1. **Batch 已釐清**：論文和單 GPU README 都是 256。128 是 YAML 預設，512 是四 GPU 範例的 global batch，不能混用。
2. **λ 未完全對齊**：論文／補充分析為 0.001；`baseline_multi_view_ViSA.py` 第 161 行固定 0.0001。採用官方程式值；不能宣稱與論文最佳超參數完全相同。另 `VIEW_LAMBDA=1.0` 是不同 loss 的係數，不能拿來設定 MoE λ。
3. **資料總數差異**：論文描述 108,563 張；官方下載的 train/query/gallery 共 102,475 張（51,451 + 51,024）。沿用官方已發布 split，不自行補資料。
4. **效率表標題歧義**：補充 Table 5 標題寫 ALL，但其 ViSA mAP=69.00，對應主表的跨視角欄，ALL 主表為 65.46。效率數字不可直接當成無歧義的 ALL 對照；自身測量另列輸入尺寸、batch、精度、硬體與參數計數口徑。
5. **程式實作疑點，暫不修改**：DLFM 的兩個分支都呼叫 `share_gcn`，雖然建立了 `view_gcn`；圖節點特徵組合也位於 `no_grad` 中。這是忠實重現官方程式時必須保存的行為，不在設定核對時擅自改成另一種架構。
6. **原版 batch=256 曾 OOM，已透過 activation checkpointing 解決**：未啟用 checkpointing 的獨立對照測試使用官方模型、FP16 AMP、官方 SGD contiguous buffers 及合成圖片，在訓練模式前向即 OOM；當次 allocated 峰值為 15.716 GiB，詳見 `output/audit/batch_256_memory.json`。後續保留實際 batch 256 並啟用 checkpointing，50-step 完整短訓練的 allocated 峰值約 11.447 GiB，並已成功完成正式 120 epochs。此處不是將小 batch 累積成 256；普通梯度累積也不能直接宣稱等價，因為 batch-hard triplet 和 BatchNorm 依賴實際 batch。
7. **參數分析不等於統一訓練設定**：補充 §8 在 E 掃描時固定 k=1，且 A↔A 的最佳 E 為 7；官方統一設定仍是 E=8、k=2。本次保留統一設定，不依測試協定另挑專家數。

論文 Table 1 的參考數字另列如下，屬已發表結果，不是本機結果，也不是保證可重現的門檻：

| 欄位 | Rank-1 (%) | mAP (%) |
| --- | ---: | ---: |
| ALL | 70.51 | 65.46 |
| A↔G | 71.28 | 69.00 |
| G↔G | 88.39 | 83.90 |
| A↔A | 82.50 | 67.78 |

## 本機實作與完成結果

正式訓練使用 [configs/CARGO/visa_official_checkpointing.yml](configs/CARGO/visa_official_checkpointing.yml)，繼承 `visa_official_single_gpu.yml` 並啟用 activation checkpointing。後者繼承官方 `visa.yml`，將 batch 設為官方單 GPU README 的 256，並調整本機權重及輸出路徑。舊的 `visa_windows.yml` 是 batch-16 bring-up 設定，沒有用於正式 baseline。

正式執行的完整設定保存在 `logs/CARGO/ViSA_official_checkpointing/config.yaml`，已由 trainer 填入 2,500 個訓練身分類別。設定核對階段另外保存的 `output/audit/visa_official_single_gpu_resolved.yml` 只是原版配方的合併快照，不是最後啟用 checkpointing 的正式執行設定；來源種子與寫死 loss 係數的核對記錄仍在 `output/audit/cargo_settings_audit.json`。`logs/` 與 `output/` 依 `.gitignore` 留在本機，未隨程式碼上傳 GitHub。

已完成的工作包括：輸出與梯度一致性驗證、batch 256 容量驗證、50-step 真實權重更新與恢復測試、每 5 epochs 保存 latest／結束保存 final 的機制，以及正式 120 epochs 訓練與四協定評估。最終模型為 `logs/CARGO/ViSA_official_checkpointing/model_final.pth`，事先固定採最後一個 epoch，沒有依測試集分數挑選。訓練運算耗時 10 小時 57 分 20 秒，含評估與其他 hooks 總耗時 12 小時 5 分 9 秒。結果表見 [CARGO_BASELINE_RESULTS.md](CARGO_BASELINE_RESULTS.md)。

後續仍未完成的專題項目是正式效率量測、改進方法、消融實驗與成功／失敗檢索視覺化；這些與「baseline 訓練是否完成」是不同進度。

## 原文與程式位置索引

- λ=0.001：主論文 PDF 第 8 頁（印刷頁 4390），§4.4 Parameter Analysis，左欄最後一段、Figure 4(c)；補充 PDF 第 2 頁，§8 延續段落、底部右欄，並對應第 3 頁 Figure 8(g)–(i)。這些是最佳值分析，不足以證明所有已發表表格都用了該值。
- λ=0.0001：`fastreid/modeling/meta_arch/baseline_multi_view_ViSA.py:161`，直接乘上 `load_balance_loss_s`。
- A↔G：主論文 PDF 第 6 頁（印刷頁 4388），Table 1 欄位及圖說。
- A→G 實作：`fastreid/data/datasets/cargo.py:146` 使用 `view='Aerial'` 建立 query，147 行使用 `view='Ground'` 建立 gallery；164–165 行過濾其他視角。
- 反方向 `CARGO_GA` 不在 `configs/CARGO/visa.yml:101` 的 `DATASETS.TESTS`，因此官方預設沒有計算兩方向平均。

最初未啟用 checkpointing 的記憶體對照測試，以 allocator fraction=1.0 將 PyTorch 配置量限制在 GPU 實體容量內，避免將 Windows 共享系統記憶體誤認為專用顯存。該次合成輸入為 `[256,3,256,128]`，只用於確認原版 OOM。後續 activation checkpointing、真實 CARGO 短訓練及完整訓練的驗證，分別記錄於 [CHECKPOINTING.md](CHECKPOINTING.md)、[SHORT_TRAINING.md](SHORT_TRAINING.md) 與 [CARGO_BASELINE_RESULTS.md](CARGO_BASELINE_RESULTS.md)，不應將最初對照測試的範圍當成目前執行狀態。
