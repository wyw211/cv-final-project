# 改進方法：修正 ETGM 專家輸入（目標：準確度）

建立日期：2026-11-24；更新：2026-11-25。狀態：程式完成並通過驗證，驗證集上的小規模篩選已完成（見「小規模篩選」）。**尚未進行正式訓練**，沒有任何測試集成績。

## 動機：官方程式與論文描述不一致

論文 §3.3 說專家 token 經過 cross-attention「整合輸入特徵 Z 的資訊」，§3.4 說每個 query 各自選 top-k patch，且兩個分支各有專屬 GCN。檢查官方程式（`fastreid/modeling/backbones/vision_transformer_multiview_ViSA.py`）後發現三處不一致：

| # | 論文描述 | 官方程式實際行為 | 影響 |
| --- | --- | --- | --- |
| 1 | 專家 cross-attend 輸入特徵 | key/value 只有 1 個 token（`inv_features` 或 `view_features`，形狀 `[B,1,768]`）。softmax 只有一個 key 時權重恆為 1，16 個 query 得到**完全相同**的輸出 | 專家完全看不到 patch；cross-attention 退化成「同一向量加到所有 query」 |
| 2 | 每個 query 選自己的 top-k patch（Eq. 8） | 用全域 token 選一次 64 個 patch，16 個 query 共用 | 所有 query 的局部圖完全相同 |
| 3 | 兩分支各有 GCN | 兩分支都呼叫 `share_gcn`；`view_gcn`（0.59M 參數）定義了但從未使用 | 視角分支沒有專屬圖推理 |

第 1 點已用數值確認：隨機權重下，官方設定的 cross-attention 輸出在 16 個 query 間的最大差異為 **0**。

## 修改內容

三個設定開關，預設值等於官方程式；**不新增任何參數**（`view_gcn` 原本就存在）。

| 設定 | 官方（預設） | 改進 |
| --- | --- | --- |
| `MODEL.BACKBONE.EXPERT_KV` | `global` | `patch`：key/value 改為 `[routing token, b1 輸出的 210 個 patch token]` |
| `MODEL.BACKBONE.GRAPH_PER_QUERY` | `False` | `True`：每個專家 query 依 cosine 相似度各自選 top-64 |
| `MODEL.BACKBONE.SEPARATE_VIEW_GCN` | `False` | `True`：視角分支使用 `view_gcn` |

- Router 仍只看 routing token，所以路由行為不受 `EXPERT_KV` 影響。
- 專家的 key 用 b1（最後一層 + LayerNorm）之後的 patch，與 routing token 同尺度；DLFM 圖節點仍沿用官方的 b1 之前的 patch，只改變選取規則。
- 其他所有設定（batch 256、λ=0.0001、120 epochs、AMP、activation checkpointing、資料增強）與重現 baseline 完全相同。

設定檔：`configs/CARGO/visa_expert_input_fix.yml`（繼承 `visa_official_checkpointing.yml`）。

## 驗證（未使用測試集）

1. **官方等價性**：三個開關都關閉時，以相同隨機權重、相同輸入（GPU、fp32、eval 模式），修改後程式與原始 commit 的輸出**逐位元相同**（max|diff| = 0）。
2. **各開關可用**：每個開關單獨打開及三者全開，都能完成 forward / backward，loss 有限；`SEPARATE_VIEW_GCN` 打開後 `view_gcn` 才有梯度。
3. **顯存（batch 256、checkpointing、CARGO 訓練資料、3 次 forward/backward、不呼叫 step）**：

| 設定 | 配置峰值 allocated | reserved |
| --- | ---: | ---: |
| 官方 baseline | 11.42 GiB | 12.29 GiB |
| 三者全開（記憶體修正前） | 13.11 GiB | 13.57 GiB |

   這是下方兩項記憶體修正之前的數字；修正後的完整短訓練峰值見「短訓練結果」。紀錄：`output/audit/batch_256_checkpointing_cargo_expert_kv_patch_graph_per_query_true_separate_view_gcn_true_memory.json`。

4. **速度（50-step 短訓練）**：見下方「短訓練結果」。

## 短訓練結果

### 第一次（已中止）：顯存溢出

最初版本在 batch 256 下 reserved 達 14.5 GiB，加上桌面程式佔用約 2–3 GB，超過 16 GB。Windows 把顯存溢出到系統記憶體，每步約 13 秒（baseline 1.55 秒），於第 20 步左右中止。紀錄保留在 `logs/bringup_20261124_221142_241860_STOPPED_vram_spill/`。

原因與修正：

1. per-query `gather` 原本從展開的 `[B,16,210,768]` 張量取值，反向會產生同尺寸梯度；改為攤平索引、從 `[B,210,768]` 取值（數值完全相同）。
2. 專家 cross-attend 211 個 token 後中間結果變大；`EXPERT_KV=patch` 且啟用 activation checkpointing 時，專家也改為反向重算（專家內沒有 dropout 或 BN）。官方路徑不受影響。

合成資料、不含 optimizer 狀態的 forward + backward 對照（batch 256）：

| 設定 | 每步秒數 | 配置峰值 |
| --- | ---: | ---: |
| 官方 | 1.68 | 9.25 GiB |
| 修正版（未加專家 checkpointing） | 1.68 | 10.82 GiB |
| 修正版（最終） | 1.78 | 9.56 GiB |

batch 128 下四種組合速度相近（0.84–0.94 秒），確認計算量本身增加很少。

### 第二次：50-step 完整短訓練

`logs/bringup_20261124_222020_a0a827/`，CARGO 訓練集，未執行任何測試集評估。

| 項目 | 修正版 | 官方 baseline（`SHORT_TRAINING.md`） |
| --- | ---: | ---: |
| 完成 iteration | 50 | 50 |
| Loss | 全部有限 | 全部有限 |
| AMP 溢位而跳過的更新 | 第 1、2、14、42、48 步，共 5 次 | 第 1、2、9、42 步，共 4 次 |
| Checkpoint 儲存／恢復 | 所有狀態雜湊一致 | 一致 |
| 配置峰值 allocated | 11.79 GiB | 11.45 GiB |
| 峰值 reserved | 13.34 GiB | — |
| 平均每步（同樣排除暖機） | 2.10 秒 | 1.55 秒 |
| 120 epochs 純訓練粗估 | 約 14.0 小時 | 10.35 小時（實際 10.95 小時） |

腳本最後以 `Final updates are not stable` 結束，原因是第 48 步屬於 AMP 正常的 loss scale 下調（4096 → 2048），落在檢查的最後 5 步內；這是 warmup 初期的正常現象，baseline 也有相同類型的跳過，不是程式錯誤。

實際資料下比合成資料測試慢較多（+36%），可能與 DataLoader 的 CPU 負載以及專家重算有關；正式訓練請關閉其他使用 GPU 的程式（遊戲、瀏覽器等）以免再次溢出。

## 小規模篩選（驗證集，20 epochs）

日期：2026-11-25。兩組只差三個開關，其餘設定相同。

- 資料：CARGO 訓練集切分（`fastreid/data/datasets/cargo_val.py`，seed 0）。2,250 人訓練，另 250 人做驗證：每人 1 張空中、1 張地面當 query，其餘當 gallery。**未使用任何測試集影像。**
- 排程：官方排程等比例縮短為 20 epochs（3,600 iterations）；warmup 300 步，維持 8.3%。
- 設定檔：`configs/CARGO/pilot_val_baseline.yml`、`configs/CARGO/pilot_val_expert_input_fix.yml`；啟動腳本：`tools/run_cargo_pilot.sh`；輸出：`logs/CARGO_pilot/`。
- 驗證集 gallery 只有 250 人，比測試集的 2,500 人容易，**分數不能和論文或測試集比較**，只能比較兩組之間的差異。

### 結果（Rank-1 / mAP，%）

| 協定 | ep5 Baseline | ep5 修正版 | ep15 Baseline | ep15 修正版 | **ep20 Baseline** | **ep20 修正版** | **ep20 差（R1 / mAP）** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 73.00 / 53.25 | 76.60 / 59.37 | 89.80 / 79.23 | 91.20 / 81.45 | 90.80 / 80.55 | 92.00 / 82.42 | +1.20 / **+1.87** |
| A→G | 61.60 / 49.35 | 66.00 / 56.41 | 83.60 / 76.83 | 86.00 / 79.87 | 85.20 / 78.55 | 87.20 / 81.06 | +2.00 / **+2.51** |
| A↔A | 59.04 / 46.87 | 61.85 / 52.58 | 80.32 / 73.25 | 84.74 / 76.58 | 81.53 / 74.78 | 85.14 / 77.57 | +3.61 / **+2.79** |
| G↔G | 84.00 / 70.50 | 86.80 / 76.21 | 95.20 / 88.77 | 95.20 / 89.72 | 95.20 / 89.41 | 95.60 / 90.38 | +0.40 / **+0.97** |

ep10 只有 ALL：Baseline 86.40 / 74.67，修正版 88.60 / 77.80。

修正版在每個時間點、每個協定都不低於 baseline；跨視角 A→G 的 mAP 進步 +2.5。**限制：每組只訓練一次（單一 seed），尚未量測 seed 之間的波動。**

### 執行中斷與處理

兩組都在第 10 epoch 的評估時中斷過一次，都從該 epoch 的 checkpoint（含 optimizer、LR、AMP、RNG）接續；DataLoader 的狀態不在存檔內，接續後的資料順序與不中斷時不同。各自的原始 log 收在 `attempt1/`。

1. Baseline：Windows 記憶體認可上限耗盡（錯誤碼 1455）。8 個訓練 worker 加上 4 個評估 worker，每個約佔 6 GB 認可額度。處理：新增 `TEST.NUM_WORKERS`（預設維持 4），試驗設為 0。
2. 修正版：評估 batch 384 加上訓練佔用的顯存，溢出到系統記憶體（每批 130 秒）。處理：試驗的 `TEST.IMS_PER_BATCH` 改為 128。eval 模式下 batch 大小不影響特徵；baseline 試驗的評估用的是 384。

### 專家行為診斷（`tools/analyze_experts.py`，1,024 張驗證影像）

| 指標 | Baseline | 修正版 |
| --- | ---: | ---: |
| cross-attention 注意力落在 routing token 的比例 | 1.000（只有這一個 key） | 0.0047（≈ 1/211） |
| 注意力分布的正規化熵（1 = 完全平均） | — | **1.000** |
| 16 個 query 的 cross-attention 輸出彼此的 cos | 1.000 | 1.000 |
| 16 個 query（專家輸出）彼此的 cos：共享／視角 | 0.928 / 0.999 | 0.991 / 0.991 |
| 不同 query 選到的 patch 重疊率：共享／視角 | 1.000 / 1.000 | 0.936 / 0.932 |
| Router 使用率熵：共享／空中／地面（1 = 8 個平均使用） | 0.69 / **0.33** / **0.33** | 0.78 / 0.39 / **0.75** |
| 專家 prompt token 的長度（初始化約 0.55） | 0.547 | 0.547 |

解讀：

- **修正版的改善不是來自「16 個 query 看不同部位」。** 注意力對 211 個 token 幾乎完全平均，因此 cross-attention 等於對所有 patch 做平均池化。16 個 query 拿到的仍是同一個向量，只是內容從「1 個全域 token」變成「全部 patch 的平均」。
- 目前較合理的解釋：專家改為接收影像局部內容的彙總，路由也明顯更平均。官方版的空中與地面 router 熵 0.333 = log 2 / log 8，代表永遠只用固定 2 個專家；修正版的地面 router 升到 0.75。哪個因素貢獻多少，要靠消融確認。
- 注意力平均的可能原因：prompt token 幾乎沒有離開初始值（長度維持約 0.55，而 LayerNorm 後的特徵長度約 27），所以 query 的投影幾乎由 bias 決定，各 query 的 attention logits 都接近 0。
- 診斷工具已修正兩個精度問題：fp16 的相似度比對，以及 `cdist` 的矩陣乘法近似。修正後，官方版的 patch 重疊率正確顯示為 1.0。

## v2：讓專家注意力不再平均（結果：比 v1 差，不採用）

設定：`configs/CARGO/pilot_val_expert_attn_v2.yml`，在 v1 上加兩個開關。

- `EXPERT_PROMPT_STD: 1.0`：prompt 初始化放大，|prompt| 約 27，和 key 同一量級（官方 0.02）。
- `EXPERT_QK_NORM: True`：cosine 注意力，加上可學習溫度（初始 10）。

選擇依據是**未訓練時**的診斷（`tools/analyze_experts.py --weights init`，512 張驗證影像）：

| 方案 | 注意力熵 | query 間注意力 L1 差 | 專家輸出 cos | patch 重疊率 |
| --- | ---: | ---: | ---: | ---: |
| v1 | 1.000 | 0.003 | 0.99 | 0.94 |
| 只放大 prompt | 0.997 | 0.14 | 0.07 | 0.33 |
| 只加 QK-norm | 0.927 | 0.32 | 0.98 | 0.91 |
| 兩者（v2） | 0.927 | 0.66 | 0.07 | 0.33 |

驗證集 ep20（Rank-1 / mAP）：

| 協定 | Baseline | v1 | v2 |
| --- | --- | --- | --- |
| ALL | 90.80 / 80.55 | **92.00 / 82.42** | 89.60 / 79.33 |
| A→G | 85.20 / 78.55 | **87.20 / 81.06** | 84.40 / 77.56 |
| A↔A | 81.53 / 74.78 | **85.14 / 77.57** | 80.32 / 73.60 |
| G↔G | 95.20 / 89.41 | **95.60 / 90.38** | 94.80 / 88.83 |

ALL mAP 在 ep5／10／15／20 分別為 54.09／74.12／78.59／79.33，每個時間點都落後 v1。

訓練後的診斷顯示 v2 的機制確實生效：16 個 query 的輸出 cos 降到 0.07，query 間注意力 L1 差 0.46，patch 重疊率 0.33。溫度維持在約 9.7，prompt 長度約 24。但準確度下降。**結論：在這個設定與訓練長度下，讓 16 個 query 分工看不同 patch 沒有幫助。** v1 的改善較可能來自「專家改看全部 patch 的平均內容」。

v2 的 router 熵為共享 0.79、空中 0.33、地面 0.35，也就是空中與地面又塌縮回只用 2 個專家。

## Router 塌縮驗證（v3 的依據）

工具：`tools/probe_router_inputs.py`，1,024 張驗證影像（250 空中、774 地面）。

| 指標 | view token（官方 router 輸入） | b1 patch 平均 |
| --- | ---: | ---: |
| 影像間平均 pairwise cos | 0.98 | 0.39–0.57 |
| router logit 在影像間的標準差 | 0.06 | 0.23–0.29 |
| Baseline：出現的不同 top-2 組合數（空中／地面） | **1 / 1** | 14 / 21（以未針對此輸入訓練的 router 計算） |
| v1：出現的不同 top-2 組合數（空中／地面） | 6 / 7 | 14 / 8 |

官方模型把**所有空中影像送到同樣 2 個專家，地面影像也一樣**；原因是 view token 從每個視角共用的同一個向量出發，到 b1 之後仍幾乎相同。同一份輸出中，top-2 與第 3 名的 logit 差距約 0.15–0.44，作為 router 雜訊強度的依據。

v3 設定：`configs/CARGO/pilot_val_router_v3.yml`，以 v1 為底座（依驗證集選出）。

| 開關 | 官方 | v3 |
| --- | --- | --- |
| `ROUTER_INPUT` | `view` | `content`：空中／地面 router 改看 b1 patch 平均 |
| `MOE_BALANCE_LOSS` | `official` | `switch`：E·Σ f_j·P_j |
| `MOE_BALANCE_LAMBDA` | 0.0001（原寫死在 meta-arch） | 0.01（Switch Transformer 常用值） |
| `ROUTER_NOISE_STD` | 0 | 0.3（僅訓練時，用於 top-k 選擇） |

v3 預設全部等於官方行為。等價性測試通過：開關全關時逐位元相同，v1 輸出不變，v3 能正常 forward／backward。

另新增 `SEED` 設定（預設 0，即 upstream 寫死的值），以及 `configs/CARGO/pilot_val_baseline_seed1.yml`，用來量測同一設定下的 seed 波動。

## 五組試驗總結（驗證集，ep20）

完成時間：2026-11-25 08:43。全部使用相同的訓練／驗證切分、20 epochs 排程與評估流程；只有 baseline 有第二個 seed。

| 協定（mAP） | Baseline s0 | Baseline s1 | **Baseline 平均** | v1 | v2 | v3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 80.55 | 81.55 | 81.05 | **82.42** | 79.33 | 81.66 |
| A→G | 78.55 | 79.91 | 79.23 | **81.06** | 77.56 | 79.84 |
| A↔A | 74.78 | 76.16 | 75.47 | **77.57** | 73.60 | 75.94 |
| G↔G | 89.41 | 89.84 | 89.63 | **90.38** | 88.83 | 89.99 |

| 協定（Rank-1） | Baseline s0 | Baseline s1 | v1 | v2 | v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ALL | 90.80 | 90.80 | **92.00** | 89.60 | 91.40 |
| A→G | 85.20 | 86.40 | **87.20** | 84.40 | 86.00 |
| A↔A | 81.53 | 82.73 | **85.14** | 80.32 | 83.13 |
| G↔G | 95.20 | **95.60** | **95.60** | 94.80 | 94.80 |

**Seed 波動**：同一個 baseline 設定只換 seed，mAP 差距在 ALL 為 1.00、A→G 1.36、A↔A 1.38、G↔G 0.43。在 ep5／10／15 的 ALL mAP 差距分別為 2.07／1.48／1.27。

| 和 baseline 兩個 seed 平均比較（mAP） | ALL | A→G | A↔A | G↔G |
| --- | ---: | ---: | ---: | ---: |
| v1 | **+1.37** | **+1.83** | **+2.10** | +0.76 |
| v2 | −1.72 | −1.67 | −1.87 | −0.80 |
| v3 | +0.61 | +0.61 | +0.47 | +0.37 |

### v3 診斷

| Router 使用率熵 | Baseline | v1 | v2 | v3 |
| --- | ---: | ---: | ---: | ---: |
| 共享 | 0.69 | 0.78 | 0.79 | 0.59 |
| 空中 | 0.33 | 0.39 | 0.33 | **0.97** |
| 地面 | 0.33 | 0.75 | 0.35 | **0.99** |

1,024 張驗證影像中，各專家被選入 top-2 的次數：

- Baseline 空中 `[0,0,250,0,0,0,250,0]`、地面 `[0,0,0,774,0,0,774,0]`
- v3 空中 `[80,104,41,72,47,53,36,67]`、地面 `[303,170,188,199,234,154,148,152]`
- v3 共享 `[28,114,7,669,166,1,1024,39]`：專家 6 每張影像都被選中

v3 的專家注意力仍完全平均（熵 1.000），與 v1 相同。

### 結論

1. **v1（修正專家輸入）是唯一穩定的改善。** 四個協定都高於 baseline 的兩個 seed，跨視角的 A→G 和 A↔A 比 baseline 平均高 1.8–2.1 mAP，約為 seed 波動的 1.3–1.5 倍。但 v1 只有一個 seed，證據屬於「一致且偏正向」，還不到確定。
2. **v2（讓 query 分工）機制有效、準確度變差。** 四個協定都低於 baseline 的兩個 seed。
3. **v3（修正 router）確實解決了視角 router 塌縮**（熵 0.33 → 0.97／0.99），但準確度和 baseline 差不多，且低於 v1。在這個設定下，專家分配不平均不是準確度的瓶頸。v3 沒有改共享 router 的輸入，而它的輸入（inv_features）在影像間的 cos 也有 0.99，v3 後反而更集中。
4. 對報告而言，v2、v3 都是有診斷支持的負面結果，可放在分析或消融段落。

## 消融規劃

| 實驗 | EXPERT_KV | GRAPH_PER_QUERY | SEPARATE_VIEW_GCN |
| --- | --- | --- | --- |
| Baseline（已完成） | global | False | False |
| 完整改進 | patch | True | True |
| 只改專家輸入 | patch | False | False |
| 去掉 per-query | patch | False | True |
| 去掉獨立 GCN | patch | True | False |

設計選擇若需要比較，應以訓練集切出的驗證集決定，測試集只做最終評估。
