# CARGO：論文與官方設定核對

核對日期：2026-11-18。僅核對 CARGO，未啟動訓練、反向傳播或測試集模型評估。

## 採用原則

使用者已確認的決定：

- **MoE λ 沿用官方程式的 0.0001**，不改成論文分析最佳值 0.001。報告及實驗紀錄保留此差異。
- **跨視角評估沿用官方 CARGO_AG 的 Aerial query → Ground gallery（A→G）**，不自行改成雙向平均。報告應註明論文使用 A↔G 名稱，但本實作為單向 A→G。
- **暫不開始正式訓練**。官方直接執行的 batch=256 已 OOM；使用者後續同意先驗證 activation checkpointing，目前已通過正確性及 3 批 CARGO batch=256 前後向容量檢查，詳見 `CHECKPOINTING.md`。沒有降低 batch 或修改模型架構，尚未做完整短訓練與速度測試。

以官方單 GPU README 指令及其實際程式行為建立 baseline。論文和程式不一致時明確記錄，不自行更改程式來追求論文數字。這是「官方程式重現設定」，不宣稱所有細節與論文完全一致。

官方 GitHub HEAD 與本機來源均為 `22ff906c28c63144d382c160d097b0de6f4f0510`。先前 Windows 路徑與評估修正仍保留，詳見 `VERIFICATION.md`。

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

以下只檢查標籤、數量與跨相機正樣本是否存在，沒有計算實際檢索成績。

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
6. **batch=256 已實測無法直接容納**：後續依使用者要求，在獨立程序使用官方模型、FP16 AMP、官方 SGD 的 contiguous parameter/gradient buffers 及合成圖片測試。訓練模式前向即 OOM，未進入反向；PyTorch 峰值 allocated=15.716 GiB，reserved=15.871 GiB，GPU 容量=15.920 GiB，失敗配置額外需要 240 MiB。未執行 optimizer.step() 或寫入 checkpoint。不能以梯度累積直接宣稱等價，因為 batch-hard triplet 和 BatchNorm 依賴實際 batch。詳見 `output/audit/batch_256_memory.json`。
7. **參數分析不等於統一訓練設定**：補充 §8 在 E 掃描時固定 k=1，且 A↔A 的最佳 E 為 7；官方統一設定仍是 E=8、k=2。本次保留統一設定，不依測試協定另挑專家數。

論文 Table 1 的參考數字另列如下，屬已發表結果，不是本機結果，也不是保證可重現的門檻：

| 欄位 | Rank-1 (%) | mAP (%) |
| --- | ---: | ---: |
| ALL | 70.51 | 65.46 |
| A↔G | 71.28 | 69.00 |
| G↔G | 88.39 | 83.90 |
| A↔A | 82.50 | 67.78 |

## 已準備但未執行

新增 `configs/CARGO/visa_official_single_gpu.yml`：繼承官方 YAML，只將 batch 設為官方單 GPU README 的 256，並調整本機權重及輸出路徑；其餘演算法設定不改。舊的 `visa_windows.yml` 仍是 batch-16 bring-up 設定，不作為正式官方 baseline。

完整合併設定已保存到 `output/audit/visa_official_single_gpu_resolved.yml`。此檔為 config 合併快照，尚未套用 trainer 自動填入的 2,500 類；來源種子與寫死 loss 係數另記錄於 `output/audit/cargo_settings_audit.json`。

設定核對本身未啟動訓練。後續使用者授權的記憶體測試只嘗試前向／反向容量檢查，實際在前向即 OOM，沒有參數更新。正式開跑前仍須處理 batch=256 的記憶體限制，以及驗證完整 optimizer／AMP／checkpoint 流程。

## 原文與程式位置索引

- λ=0.001：主論文 PDF 第 8 頁（印刷頁 4390），§4.4 Parameter Analysis，左欄最後一段、Figure 4(c)；補充 PDF 第 2 頁，§8 延續段落、底部右欄，並對應第 3 頁 Figure 8(g)–(i)。這些是最佳值分析，不足以證明所有已發表表格都用了該值。
- λ=0.0001：`fastreid/modeling/meta_arch/baseline_multi_view_ViSA.py:161`，直接乘上 `load_balance_loss_s`。
- A↔G：主論文 PDF 第 6 頁（印刷頁 4388），Table 1 欄位及圖說。
- A→G 實作：`fastreid/data/datasets/cargo.py:146` 使用 `view='Aerial'` 建立 query，147 行使用 `view='Ground'` 建立 gallery；164–165 行過濾其他視角。
- 反方向 `CARGO_GA` 不在 `configs/CARGO/visa.yml:101` 的 `DATASETS.TESTS`，因此官方預設沒有計算兩方向平均。

記憶體測試以 allocator fraction=1.0 將 PyTorch 配置量限制在 GPU 實體容量內，避免 Windows 共享系統記憶體被誤認為能放進 16 GB 專用顯存。輸入為 `[256,3,256,128]`；64 個合成身分、每人 4 張，空拍／地面各半；不是 CARGO 效能測試，也不是正式訓練。只驗證本次官方直接執行路徑，未測試 activation checkpointing 等節省記憶體的改寫。
