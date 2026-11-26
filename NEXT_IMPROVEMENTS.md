# 下一輪改進方案（v4 起，不限於 ETGM）

所有數字都是驗證集（CARGO_VAL，從訓練集切出 250 個 ID）上第 20 個 epoch 的結果，測試集完全沒有使用。

## 1. 已經實測：不用重新訓練的方法

評估的是現有的 pilot checkpoint（`logs/CARGO_pilot/tta/`）。格式為 mAP（R1）。

| 模型 | 測試方式 | ALL | A→G | A↔A | G↔G |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | 一般 | 80.55 (90.80) | 78.56 (85.20) | 74.78 (81.53) | 89.41 (95.20) |
| baseline | 翻轉 TTA | 79.15 (90.40) | 77.65 (83.60) | 74.93 (81.93) | 89.35 (95.20) |
| baseline | k-reciprocal 重排序 | **89.89 (91.80)** | **86.15 (86.00)** | **81.96 (85.14)** | **93.03** (94.80) |
| v1 | 一般 | 82.42 (92.00) | 81.06 (87.20) | 77.57 (85.14) | 90.38 (95.60) |
| v1 | 翻轉 TTA | 78.23 (90.80) | 80.21 (84.40) | 77.42 (84.34) | 90.35 (95.60) |
| v1 | k-reciprocal 重排序 | **91.19 (92.60)** | **87.92 (88.40)** | **83.47 (86.75)** | **93.75** (94.00) |

重排序使用預設參數 k1=20、k2=6、λ=0.3。

- **重排序**：mAP 提升 7～9，而且對 baseline 和 v1 的增益差不多，可以和模型改進疊加。
  - 它屬於後處理，報告中要和模型本身的改進分開列。
  - 參數只能在驗證集上調。
  - G↔G 的 R1 會掉 0.4～1.6。
  - 測試集的 gallery 有 5 萬張圖，N×N 矩陣要先估好記憶體用量（約 10 GB），可能要改成 fp16 或分塊計算。
- **翻轉 TTA**：反而變差，可以當負面結果。
- 修正：Windows 的 faiss-cpu 沒有 GPU index，`fastreid/utils/compute_dist.py` 改成在這種情況下自動改用 CPU 搜尋。

## 2. 支撐新方向的觀察

- **閒置的專家**：baseline 的 24 個專家中有 14 個從來沒被選到。
  - 空中組和地面組各只用到 2 個，共享組有 2 個從沒被選到。
  - 每個專家約 7.1M 參數，所以大約 99M（佔全模型 279M 的 35%）是閒置參數。
- **v3 的教訓**：把路由修平均之後，準確度沒有提升。可見瓶頸不在 MoE 的分配，應該往特徵本身（預訓練）和跨視角的監督訊號去找。
- **取樣器不是問題**：每個 ID 隨機抽 4 張時，有 88% 的機率同時包含空中和地面的照片，所以不打算改取樣器。
- **最新文獻**：
  - ProAlign（arXiv 2611.04980，NeurIPS 2026）：只用 RGB，靠「視角感知的原型對比學習」和「漸進式位置編碼」，在 CARGO 上報告的 ALL mAP 為 62.47、A↔G mAP 為 58.44。
  - VR3D（arXiv 2608.02598）：需要 SAM3D 產生的 3D 資訊，成本太高，不採用。

## 3. 候選方案（依推薦順序）

| 代號 | 方法 | 改動位置 | 預期效果 | 成本 |
| --- | --- | --- | --- | --- |
| P1 | 跨視角原型對比損失（參考 ProAlign 的 PCL） | 損失函數（只在訓練時用） | A→G、A↔A 提升 | 約 100 行程式，1 次 pilot |
| P2 | ReID 專用的預訓練權重（LUPerson DINO ViT-B） | backbone 初始化 | 全面提升，預期最大 | 需要下載權重，1 次 pilot |
| P3 | 專家精簡（8 個減到 2～4 個） | ETGM | 準確度持平，參數少約 35% | 只改設定，1 次 pilot |
| P4 | 條件式位置編碼（depthwise conv，以 zero-init gate 加入） | backbone 淺層 | 小幅提升 | 約 40 行程式，1 次 pilot |
| R | 重排序參數調整 | 推論階段 | 已實測 +7～9 mAP | 只要幾分鐘 |

各方案的細節：

- **P1：跨視角原型對比損失**
  - 每個 ID 維護兩個 momentum 原型，分別是空中原型和地面原型。
  - 用 InfoNCE 拉近每張圖和它「另一個視角」的原型，同時推離其他 ID 的原型。
  - 這直接處理目前最弱的跨視角檢索，而且推論時不增加任何成本。
- **P2：ReID 專用預訓練**
  - 把 ImageNet 的權重換成在 LUPerson 上用 DINO 預訓練的 ViT-B（TransReID-SSL 或 PersonViT 提供）。架構相同，只需要對應權重名稱。
  - 根據文獻，ReID 專用的預訓練通常比 ImageNet 好上數個 mAP。
  - 對照組（baseline）要用同一份權重重跑，比較才公平。
- **P3：專家精簡**
  - 作業要求報告 Params、GFLOPs 和延遲。這個方案能提供「準確度幾乎一樣、參數大幅減少」的效率型改進。
  - 由於每次仍只選 top-2，GFLOPs 幾乎不變，主要省的是參數量和顯示卡記憶體。

建議的組合：把 P1、P2、P3 分別疊在 v1 上各跑一次 pilot，選出有效的再一起組合，最後才跑正式訓練。重排序另外當成推論端的改進，baseline 和改進版本都要套用，報告時兩種數字都列出來。

## 4. 執行狀態（2026-11-25）

- **P1 已實作**：
  - 程式：`ViewPrototypeMemory`，在 `fastreid/modeling/meta_arch/baseline_multi_view_ViSA.py`。
  - 設定：`MODEL.LOSSES.PCL.*`，預設關閉。
  - Pilot 設定檔：`configs/CARGO/pilot_val_pcl_p1.yml`。
  - 50 步小規模訓練正常，峰值記憶體 13.4 GB。
- **P3**：
  - Pilot 設定檔：`configs/CARGO/pilot_val_experts4_p3.yml`。
  - 參數量從 275.23M 降到 190.01M（專家部分從 170.42M 降到 85.21M）。
  - 50 步小規模訓練正常，峰值記憶體 11.1 GB。
- 10:41 起依序執行 P1 → P3 的 pilot，每組約 2.2 小時。
- **P2 已準備好**：
  - 已下載 PersonViT ViT-B/16 的 `checkpoint0240.pth`（1.51 GB），SHA256 為 `162c5a4d…`，和 HuggingFace 上 LFS 公布的值一致。
  - 載入時使用 `weights_only=True`，只放行 numpy scalar、dtype 和 argparse.Namespace 這三種型別，所以不會執行檔案中夾帶的程式碼。
  - `tools/convert_personvit.py` 會取出 teacher backbone，把 pos_embed 從 16×8 的網格用雙線性內插成 21×10（對應 stride 12），CLS 的位置編碼同時給兩個前導 token 使用。輸出檔為 `weights/personvit_vitb_lup_teacher_e240_s12.pth`。
  - 驗證結果：150 個 tensor 全部對上（最大差值 0），缺少的只有 `view_token_sky` 和 `view_token_ground`，和官方載入 jx 權重時相同。
  - 學習率：PersonViT 官方的微調是 LR 4e-4、batch 64，依線性縮放換算到 batch 256 是 1.6e-3。因此預訓練的 ViT 部分（`backbone.base`、`backbone.b1`，共 92.9M 參數）使用 0.2 × BASE_LR。新加入的模組維持官方的 0.008。對應的新設定是 `SOLVER.PRETRAINED_LR_FACTOR`，預設 1.0。
  - 設定檔：`pilot_val_personvit_v1.yml` 和 `pilot_val_personvit_baseline.yml`。已排在 P3 之後自動執行（`logs/CARGO_pilot/queue_p2.sh`）。
  - 混淆因素：這兩組同時換了權重和學習率倍率。如果 P2 有效，還需要補一組「jx 權重 + 0.2 倍率」，才能把兩個因素的效果拆開。
- **預計完成時間**：P1 約 12:40、P3 約 15:00、PersonViT-v1 約 17:15、PersonViT-baseline 約 19:30。

## 5. P1 結果（驗證集，ep20，mAP（R1））

| 執行 | ALL | A→G | A↔A | G↔G |
| --- | ---: | ---: | ---: | ---: |
| baseline s0 / s1 平均 | 81.05 | 79.23 | 75.47 | 89.63 |
| v1 | 82.42 (92.00) | 81.06 (87.20) | 77.57 (85.14) | 90.38 (95.60) |
| **P1 = v1 + PCL** | **83.93 (92.60)** | **82.77 (87.60)** | **78.17** (84.74) | **91.05** (95.60) |

- **P1 對 v1：** mAP 分別 +1.51 / +1.71 / +0.60 / +0.67。
- **P1 對 baseline 平均：** mAP 分別 +2.88 / +3.54 / +2.70 / +1.42。四個協定的 mAP 都是目前最高。
- **A↔A：** R1 比 v1 低 0.40，mAP 則高 0.60。
- **跑完時間：** 12:44，訓練中沒有出現錯誤。

P3 的 ep5 中途結果：ALL mAP 61.32。同樣在 ep5 時，v1 是 59.37、baseline 是 53.25。

## 6. P3 結果（驗證集，ep20，mAP（R1））

| 執行 | 參數量 | ALL | A→G | A↔A | G↔G |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1（8 個專家） | 275.23M | 82.42 (92.00) | 81.06 (87.20) | 77.57 (85.14) | 90.38 (95.60) |
| **P3 = v1、4 個專家** | **190.01M** | 82.87 (92.00) | 81.92 (88.80) | 77.30 (83.13) | 90.48 (95.60) |

- **和 v1 比：** mAP 分別 +0.45 / +0.86 / −0.27 / +0.10，都在 seed 波動範圍內（0.4～1.4），可以視為持平。
- **參數量：** 少了 31%，從 275.23M 降到 190.01M。
- **效率型改進成立：** 用少 85M 的參數，得到相同的準確度。
- **待確認：** A↔A 的 R1 掉了 2.0，要留意這是否只是單次試驗的波動。
- **已跑完：** 14:18 結束，exit 0。

**PersonViT-v1 的 ep3 初步觀察：** 訓練損失比 v1 在同一位置低（cls 6.36 對 6.78，total 15.74 對 16.28），沒有發散。

## 7. P2（PersonViT）結果（驗證集，ep20，mAP（R1））

| 執行 | 權重 | ALL | A→G | A↔A | G↔G |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline s0 / s1 平均 | jx (ImageNet) | 81.05 | 79.23 | 75.47 | 89.63 |
| v1 | jx | 82.42 (92.00) | 81.06 (87.20) | 77.57 (85.14) | 90.38 (95.60) |
| P1 = v1 + PCL | jx | 83.93 (92.60) | 82.77 (87.60) | 78.17 (84.74) | 91.05 (95.60) |
| P3 = v1、4 專家 | jx | 82.87 (92.00) | 81.92 (88.80) | 77.30 (83.13) | 90.48 (95.60) |
| **PersonViT-baseline** | PersonViT | **87.91** (94.40) | **86.05** (89.20) | **84.47** (90.36) | 92.89 (96.80) |
| **PersonViT-v1** | PersonViT | **88.02** (94.20) | 86.00 (89.20) | 84.46 (89.96) | **93.09** (96.80) |

- **換權重的效果最大：** PersonViT-baseline 比 jx baseline 平均高 +6.86 / +6.82 / +9.00 / +3.26。A↔A 提升最多。
- **v1（ETGM 修正）在 PersonViT 上幾乎沒有增益：** 對 PersonViT-baseline 只有 +0.11 / −0.05 / −0.01 / +0.20，在 seed 波動內。jx 上的 +1.4 是因為特徵較弱時才有幫助。
- **跑完時間：** v1 15:55、baseline 17:28，都 exit 0。
- **仍有混淆：** 兩組都同時換了權重和 `PRETRAINED_LR_FACTOR 0.2`，需要「jx + 0.2」對照組來拆開。

## 8. 第二批 pilot（2026-11-25 22:21 開始）

| 執行 | 設定檔 | 目的 |
| --- | --- | --- |
| PersonViT-baseline + 4 專家（P2+P3） | `pilot_val_personvit_experts4.yml` | 在強權重上是否仍能減參數而不掉準確度 |
| PersonViT-baseline + PCL（P2+P1） | `pilot_val_personvit_pcl.yml` | PCL 在 PersonViT 上是否仍有效 |
| jx 權重 + 0.2 倍學習率 | `pilot_val_jx_lr02.yml` | 拆開「權重」和「學習率倍率」兩個因素 |

- 小規模測試：三組都能正常訓練，峰值記憶體 10.6～11.7 GB。
- 啟動方式：`tools/run_pilot_queue2.sh`。
- 預計完成時間約為 23:55、01:30、03:05。

## 9. 待辦：Depth Anything（等第二批 pilot 跑完再做）

- 先用 Depth Anything V2-Small（Apache-2.0）在 20～30 張空中和地面的**訓練圖**上估深度，看能不能看出人形。如果看不出來就停在這一步。
- 如果可行：離線算出訓練集的深度圖，加一個只在訓練時使用的輔助深度預測頭，推論時拿掉，所以推論成本不變。以 PersonViT-baseline 為對照跑 1 組 pilot。

## 10. 第二批結果（驗證集，ep20，mAP（R1））

| 執行 | ALL | A→G | A↔A | G↔G |
| --- | ---: | ---: | ---: | ---: |
| PersonViT-baseline | 87.91 (94.40) | 86.05 (89.20) | 84.47 (90.36) | 92.89 (96.80) |
| PersonViT + 4 專家 | 87.88 (94.00) | 86.00 (89.60) | 84.60 (91.16) | 92.98 (96.80) |
| PersonViT + PCL | 87.99 (94.20) | 86.12 (89.20) | 84.34 (90.76) | 92.86 (96.40) |

- 在 PersonViT 權重上，PCL 和 4 個專家的差距都在 ±0.15 以內，等於持平。
- 4 個專家仍然可以當作效率型改進：準確度不變、參數少約 31%。
- jx + 0.2 倍學習率的對照組在 01:26 開始，預計 03:30 左右完成。

## 11. 對照組結果與正式訓練建議（2026-11-26 03:01）

| 執行 | ALL | A→G | A↔A | G↔G |
| --- | ---: | ---: | ---: | ---: |
| jx baseline（LR ×1.0，兩個 seed 平均） | 81.05 | 79.23 | 75.47 | 89.63 |
| jx + LR ×0.2 | 61.17 (77.20) | 57.93 (69.20) | 54.27 (64.26) | 77.21 (86.40) |
| PersonViT-baseline（LR ×0.2） | 87.91 | 86.05 | 84.47 | 92.89 |

- 降低學習率讓 jx 掉了約 20 mAP，所以 P2 的進步確實來自 PersonViT 權重本身。
- 參數量（含分類頭）：PersonViT-baseline 為 271.78M；4 個專家的版本為 186.56M，少 31%。
- 正式訓練建議（120 epoch，全部 2,500 個 ID，每組約 10 小時）：
  1. **重現的 baseline**：官方 ViSA，jx 權重。
  2. **最終方法**：PersonViT 權重 + 4 個專家。
  3. 選配：PersonViT + 8 個專家，用來在測試集上拆開「權重」和「專家數」兩個因素。
- 所有組都只在最後評估一次測試集，並另外列出套用重排序後的數字。

## 12. 正式訓練結果：測試集，120 epoch，只評估一次，mAP（R1）

| 執行 | 參數量 | ALL | A→G | A↔A | G↔G |
| --- | ---: | ---: | ---: | ---: | ---: |
| 重現的 baseline（11/18，jx，8 專家） | 271.78M | 64.59 (68.91) | 68.16 (69.15) | 66.99 (77.50) | 82.02 (85.71) |
| **PersonViT + 4 專家（11/26）** | **186.56M** | **69.90 (74.04)** | **72.24 (73.40)** | **69.70 (80.00)** | **84.13 (86.61)** |
| 差值 | −31% | +5.31 (+5.13) | +4.08 (+4.25) | +2.71 (+2.50) | +2.11 (+0.90) |

- 訓練時間：03:18 到 12:48，exit 0。
- 重排序評估（`logs/CARGO/rerank/`）在 12:5x 開始，兩個模型都使用預設參數。

### 12.1 加上重排序（k1=20、k2=6、λ=0.3，測試集，mAP（R1））

| 執行 | ALL | A→G | A↔A | G↔G |
| --- | ---: | ---: | ---: | ---: |
| baseline | 64.59 (68.91) | 68.16 (69.15) | 66.99 (77.50) | 82.02 (85.71) |
| baseline + 重排序 | 66.81 (70.83) | 73.71 (72.34) | 57.02 (60.00) | 82.81 (85.71) |
| 最終方法 | 69.90 (74.04) | 72.24 (73.40) | 69.70 (80.00) | 84.13 (86.61) |
| 最終方法 + 重排序 | **74.32 (79.81)** | **73.72** (73.40) | 66.60 (72.50) | **87.19 (92.86)** |

- 在測試集上，重排序的增益比驗證集小：ALL 只有 +2.2～+4.4，驗證集上是 +7～9。
- A↔A 反而下降，baseline 掉 10 mAP、最終方法掉 3。A↔A 的 query 很少，R1 每 2.5% 就是一張，所以波動很大。
- 兩種情況下，最終方法都贏過 baseline：不加重排序 ALL mAP +5.31，加重排序 ALL mAP +7.51。
- 重排序跑完的時間：baseline 13:13，最終方法 13:32。

### 9.1 Depth Anything 可行性檢查結果（2026-11-26）

- 方法：Depth Anything V2-Small（24.8M 參數，safetensors，SHA256 `3152477c…`，和 HuggingFace 公布的值一致）。從**訓練集**隨機抽 6 個 ID，每個 ID 取空中和地面各 2 張，共 24 張（seed 0）。
- 結果圖：`logs/depth_check/depth_contact_sheet.png`。套件裝在 `../extra_pkgs/depth`，沒有改動專案的 .venv。
- **地面照**：12 張中有 11 張的人形輪廓清楚。不過旁邊的路人和目標深度相同，無法用深度把目標分出來。
- **空中照**：約 6 張可用，主要是白天、解析度較高的；約 4 張失敗，只剩一團色塊或地面的深度漸層，主要是夜間和寬度不到 40 像素的小圖。
- 判斷：深度圖帶來的資訊，主要等於一張人形遮罩。無人機從高處往下看，人和地面的深度差很小，提供不了多少 3D 結構。而最需要幫助的夜間和小尺寸空中圖，正是它失敗的地方。預期增益有限，優先度排在 GFLOPs、延遲和視覺化之後。

## 13. A↔A 診斷（只用驗證集，模型為 PersonViT + 4 專家的 pilot）

工具：`tools/diag_per_query.py`，只接受 `_VAL` 協定。輸出放在 `logs/aa_diag/`。

| 查詢圖分組 | 數量 | mAP | R1 |
| --- | ---: | ---: | ---: |
| 全部 A↔A | 249 | 84.60 | 91.16 |
| 白天 | 123 | **92.64** | **99.19** |
| 夜間 | 126 | **76.75** | **83.33** |
| 橫向圖（w>h） | 149 | 85.72 | 91.95 |
| 直向圖（h/w≥1.5） | 63 | 87.47 | 93.65 |
| 短邊 < 48 px | 49 | 85.69 | 89.80 |

- **瓶頸是夜間，不是圖片形狀或解析度。** 22 個 R1 失敗中有 21 個是夜間查詢。
- 最暗的三台夜間攝影機是 Cam1、Cam2、Cam5，平均亮度約 44，白天約 125。它們的 mAP 最低：Cam1 72.3、Cam5 71.5、Cam2 80.0。Cam3、Cam4 的夜間亮度約 66～73，mAP 是 82～85。
- 夜間查詢的正樣本中，有 27% 是白天拍的，所以模型也需要跨光照比對。
- A→G 有相同的現象：夜間 79.07，白天 93.15。
- 失敗案例圖：`logs/aa_diag/val_aa_failures.png`。
- 64% 的空中圖是橫向的，壓成 256×128 時會被拉長約 2.6 倍，但它們的 AP 並沒有比較低，所以不優先處理。
- 測試集的 A↔A 只有約 40 個有效查詢，每個查詢就佔 R1 的 2.5%，波動很大。改進要以驗證集的夜間子集來判斷。

候選方案：
- L1：低光照資料增強，只在訓練時使用。
- L2：以影像本身亮度做自動 gamma 校正，訓練和測試都套用。
- L3：L1 加上光照一致性損失。

### 13.1 光照改進的實作與 pilot（2026-11-26 14:10 開始）

- **L1**：`RandomIllumination`，位於 `fastreid/data/transforms/transforms.py`，對應設定 `INPUT.LOWLIGHT.*`。以 0.5 的機率處理每張圖：
  - 亮圖調暗：gamma 2～4、對比 0.3～0.7、飽和度 0.3～0.8，再加上雜訊。調暗後的白天圖平均亮度是 0.26，真實夜間圖是 0.18。
  - 暗圖調亮：gamma 0.4～0.75。
- **L2**：`AutoGamma`，設定 `INPUT.AUTOGAMMA.*`。訓練和測試都套用。平均亮度 m 低於 0.45 的圖，使用 gamma = log 0.45 / log m，下限 0.4。
- **L3**：L1 再加上 `MODEL.LOSSES.ILLUM`。每個 batch 取前 64 張圖，在 GPU 上用同樣方式重新打光，要求它們的不變特徵和原圖特徵的 cosine 相似度接近 1（原圖特徵 detach），權重 1.0。速度會慢約 24%。
- 小規模測試：三組都正常，峰值記憶體約 10.6 GB。L3 的 loss_illum 初始值約 0.2。
- 範例圖：`logs/aa_diag/light_aug_examples_v2.png`。
- 啟動方式：`tools/run_pilot_queue3.sh`。預計完成時間約為 L1 15:45、L2 17:20、L3 19:15。
- 判斷標準：驗證集 A↔A 夜間子集的 mAP 要比 PersonViT + 4 專家 pilot 的 76.75 高 2 分以上，而且 ALL 和 G↔G 不能退步。

### 13.2 L1 結果（驗證集，ep20，mAP（R1））

| 執行 | ALL | A→G | A↔A | G↔G | A↔A 夜間 | A↔A 白天 | A→G 夜間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 對照：PersonViT + 4 專家 | 87.88 (94.00) | 86.00 | 84.60 (91.16) | 92.98 | 76.75 | 92.64 | 79.07 |
| L1 低光照增強 | 85.27 (92.00) | 83.92 | 81.90 (86.75) | 91.27 | **73.46** | 90.54 | 77.06 |

- **L1 無效，而且全面退步**：ALL −2.6、A↔A −2.7，連目標的夜間子集也 −3.3。
- 可能原因：
  - 人工調暗會破壞顏色和紋理線索，模型學到的特徵反而變弱。
  - 加了強增強後收斂比較慢，20 個 epoch 不夠。不過到 ep15、ep20 仍落後 2.6，看不出會追上。
- 可以當作負面結果：簡單的低光照模擬無法取代真實的夜間資料。
- 15:36 跑完。
- **L3 已修改**（使用者同意）：拿掉 L1 的資料增強，只保留一致性損失。主要的訓練圖不調暗，每個 batch 另外在 GPU 上重新打光 64 張，只用在一致性分支。

### 13.3 L2 (AutoGamma, train + test) result — 2026-11-26 17:06, ep20 val

| Run | ALL mAP | A→G | A↔A | G↔G | Night A↔A mAP (R1) | Day A↔A mAP (R1) |
|---|---:|---:|---:|---:|---:|---:|
| Control (PersonViT + 4 experts) | 87.88 | 86.00 | 84.60 | 92.98 | 76.75 (83.33) | 92.64 (99.19) |
| L2 AutoGamma | 87.62 | 85.53 | 84.22 | 92.88 | 76.73 (79.37) | 91.89 (97.56) |

Conclusion: no gain on night A↔A (−0.02 mAP, −4.0 R1) and slightly lower elsewhere. Brightening the input does not
add information that the network did not already use. L2 is rejected. L3 (consistency loss only) started 17:06.

### 13.4 L3 (illumination-consistency loss only) result — 2026-11-26 18:58, ep20 val

| Run | ALL mAP | A→G | A↔A | G↔G | Night A↔A mAP (R1) | Day A↔A mAP (R1) |
|---|---:|---:|---:|---:|---:|---:|
| Control (PersonViT + 4 experts) | 87.88 | 86.00 | 84.60 | 92.98 | 76.75 (83.33) | 92.64 (99.19) |
| L3 consistency loss | 87.52 | 85.67 | 84.50 | 92.72 | 76.22 (81.75) | 92.97 (100.00) |

Conclusion: no gain (night A↔A −0.53 mAP). loss_illum was already ~0.003 early in training, i.e. the PersonViT
features are already nearly invariant to global relighting; the night gap is information loss (noise, blur,
missing colour), not a brightness shift that a photometric transform can undo.

### 13.5 Summary of the low-light attempts

None of L1 (augmentation), L2 (AutoGamma) or L3 (consistency loss) beats the control, so the final method stays
PersonViT + 4 experts. The A↔A weakness is reported as analysis/limitation in the report, using the night/day split
(night mAP 76.75 vs day 92.64 on val A↔A; 21 of 22 R1 failures are night queries) and the three negative results.

## 14. Training-free post-processing for night A↔A (val only) — 2026-11-26

Script: `tools/posthoc_val.py` (+ `tools/posthoc_combo_val.py`), features of the PersonViT + 4 experts pilot
(`logs/posthoc/*.npz`, plain + flipped features, luminance from pixels). Validation protocols only. mAP (R1); night = A↔A night queries.

| Variant | ALL | A→G | A↔A | G↔G | Night A↔A |
|---|---:|---:|---:|---:|---:|
| base | 87.88 (94.00) | 86.00 (89.60) | 84.60 (91.16) | 92.98 (96.80) | 76.75 (83.33) |
| flip TTA | 88.46 (94.00) | 86.69 (90.00) | 85.36 (90.36) | 93.50 (97.20) | 77.48 (80.95) |
| illum-center (luminance groups) | ≈ base | ≈ base | ≤ base | ≈ base | ≤ base |
| cam-center a=1.0 | 87.89 (94.60) | 85.80 (90.00) | 84.67 (91.97) | 92.90 (96.40) | 77.66 (85.71) |
| αQE k=2 α=3 | 92.36 (93.60) | 89.92 (88.80) | 89.60 (91.57) | 95.46 (96.40) | 83.71 (84.92) |
| flip + cam-center 0.5 + αQE k=2 α=3 | 93.14 (94.20) | 91.08 (90.40) | 89.97 (91.97) | 96.68 (97.60) | 83.90 (84.92) |

Findings: αQE (query expansion + database-side augmentation, features only, no labels) gives +5.0 A↔A mAP and
+7.0 night A↔A mAP, and helps every protocol. Gains are mainly in mAP; R1 is roughly flat. Illumination-group
centering does nothing (consistent with L1–L3: brightness is not the problem). Night queries gain most because
their few confident gallery matches pull in the other positives. Like k-reciprocal re-ranking, this is transductive
(uses gallery features at test time); all settings are chosen on val only.

### 14.1 Test set, αQE locked on val (k=2, α=3) — single evaluation, 2026-11-26 20:19

`tools/eval_test_aqe.py`, results in `logs/CARGO/aqe_test/results.json` (exit code 1 came from a `tee` to a not-yet-existing
folder; the script itself finished and wrote the json). Plain numbers reproduce §12 exactly. mAP (R1).

| Run | Post-proc | ALL | A→G | A↔A | G↔G |
|---|---|---:|---:|---:|---:|
| Reproduced baseline | none | 64.59 (68.91) | 68.16 (69.15) | 66.99 (77.50) | 82.02 (85.71) |
| Reproduced baseline | αQE | 69.02 (73.40) | 72.15 (73.40) | 68.52 (72.50) | 84.54 (87.50) |
| Final (PersonViT + 4 experts) | none | 69.90 (74.04) | 72.24 (73.40) | 69.70 (80.00) | 84.13 (86.61) |
| Final (PersonViT + 4 experts) | αQE | **73.38 (77.88)** | **75.79 (74.47)** | **73.12 (80.00)** | **89.03 (91.07)** |
| Final | k-reciprocal re-rank (§12.1) | 74.32 (79.81) | 73.72 (73.40) | 66.60 (72.50) | 87.19 (92.86) |

A↔A: final + αQE is +3.42 mAP over final plain and +6.13 over the reproduced baseline; unlike k-reciprocal
re-ranking it does not hurt A↔A. Only 40 valid A↔A test queries (1 query = 2.5 R1), so A↔A R1 is coarse.

## 15. Efficiency (Params / GFLOPs / latency) — 2026-11-26

`tools/measure_efficiency.py` (random inputs, no dataset images), RTX 5070 Ti, torch 2.7.1+cu128, fp32, eval, no grad,
256×128 input. FLOPs with `torch.utils.flop_counter` at batch 1 (executed ops; the MoE dispatch is sparse, so this is
the active top-2 compute). Latency = median of 200 runs after 50 warm-up. Two runs in opposite order
(`logs/efficiency/results_run1.json`, `results_run2_reversed.json`).

| Model | Params (M, incl. heads) | GFLOPs (GMACs) | Latency bs=1 (ms), run1 / run2 | Throughput bs=64 (img/s), run1 / run2 |
|---|---:|---:|---:|---:|
| Reproduced baseline (8 experts) | 275.62 | 42.06 (21.03) | 15.2 / 14.9 | 156.8 / 136.1 |
| Final (PersonViT + 4 experts) | 190.40 | 42.06 (21.03) | 14.1 / 14.5 | 138.9 / 158.3 |

- Params −85.2 M (−31%). FLOPs identical: each token goes through top-2 experts in both models, so halving the number
  of experts saves parameters/memory, not compute. Latency/throughput equal within noise (throughput depends on which
  model is measured first, ±10%). Aerial and ground inputs cost the same.
- (Earlier §11 count 271.78 / 186.56 M excluded some heads; difference between the models is the same 85.2 M.)
- αQE cost at CARGO test size (312 queries, 50,712 gallery, d=1536): 91 s with the numpy implementation (CPU, full
  argsort), **0.41 s** with an equivalent GPU torch implementation (`tools/aqe_gpu_timing.py`, max abs difference
  6e-8). The gallery-side expansion is a one-off offline cost; per query it is one extra top-k over the gallery.

## 16. Retrieval visualisation (test, final evaluation output only) — 2026-11-26

`tools/vis_retrieval.py logs/CARGO/ViSA_official_checkpointing logs/CARGO/ViSA_personvit_experts4`; figures in
`logs/CARGO/vis/` (random examples, seed 0, not hand-picked; cached test features in `logs/CARGO/vis_feats/`).
- `CARGO_{AG,AA,GG}_final_aqe.png`: final + αQE, 3 R1 successes + 3 R1 failures, top-10 (green same ID / red other).
- `compare_CARGO_{AG,AA,GG}.png`: baseline (plain) vs final + αQE on the same queries.

R1 failure statistics of final + αQE (test):

| Protocol | R1 failures | of which night | fixed vs baseline | both wrong | broken vs baseline |
|---|---:|---:|---:|---:|---:|
| A→G | 24 / 94 | 14 | 13 | 16 | 8 |
| A↔A | 8 / 40 | **8** | 3 | 6 | 2 |
| G↔G | 10 / 112 | 6 | 6 | 10 | 0 |

All remaining A↔A failures on test are night queries (same pattern as val). Typical failures: very dark, tiny
aerial crops where the top-10 are other dark silhouettes; A→G failures often retrieve people with similar clothing.
