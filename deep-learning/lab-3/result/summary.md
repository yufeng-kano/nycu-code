# DL Lab 3 — Experiment Summary

**目前最佳：016-erasing-0.5，val 0.9325**（supervised 0.9042；對應 `result/016-erasing-0.5.csv`、`checkpoints/016-erasing-0.5.pt`）

搬回 notebook 需要改的地方：
1. Building blocks cell：`ModelConfig.drop_path_rate` 0.1 → 0.2
2. Building blocks cell：`ModelConfig.widths` (64, 128, 256, 384) → (64, 128, 256, 512)（參數 7.80M → 11.25M）
3. Supervised 訓練迴圈 cell：`max_epochs` 100 → 200
4. Self-training 迴圈 cell：`n_epochs` 100 → 200、`N` 10 → 5
5. Transforms cell：`transforms.RandomErasing(p=0.25)` → `p=0.5`

> 📉 平台期：017–040 共 24 組單一改動（不含刻意壓力測試的 TA 關閉、LS 0.0）都落在 0.9075–0.9300，016 可能是偏高的一次抽樣。實驗在 040 告一段落（使用者指示）。

> ⚠️ 注意：notebook 與腳本的隨機數使用順序不同，數字不會完全一樣（001 腳本重現 0.8958 vs notebook 0.9025）。把設定搬回 notebook 重跑後，notebook 的 val 和 result.csv 會跟 `result/` 裡的 csv 不同。助教看的是 notebook 自己的輸出，決定交哪個 csv 前請先注意這點。
> 1 張 val 圖 = 0.083%；差距小於約 0.5%（6 張）的「破紀錄」都在 val 雜訊範圍內，會在備註標明。

---

## Experiments

固定 seed 2026；val acc 為 self-training 階段的最佳 val acc（括號內為 supervised 階段最佳）。

| 編號 | 改動 | val acc | 是否破紀錄 | 備註 |
|---|---|---|---|---|
| 001-init | notebook baseline（腳本重現） | 0.8958 (0.8842) | 基準 | 原 notebook 0.9025 (0.8908)；差 0.67%，在 ±1% 內，重現成功 |
| 002-sup200 | supervised epochs 100 → 200 | 0.9025 (0.8883) | 否（與最佳同分） | 比 001 腳本重現 +0.67%；sup 後段 train 0.935 > val 0.88，開始 overfit。同分時以 002 作為後續實驗基礎 |
| 003-img144 | IMG_SIZE 128 → 144（基於 002） | 0.9008 (0.8925) | 否 | sup 略升但 SSL 沒有提升；耗時 +25%。維持 128 |
| 004-ssl-lr2e-4-warmup | SSL lr 5e-4 → 2e-4 + 5 epoch warmup（基於 002） | 0.8950 (0.8883) | 否 | 開頭掉分未改善；pseudo 只收 2537（002 為 2803）。維持 5e-4、不 warmup |
| 005-droppath-0.2 | drop_path_rate 0.1 → 0.2（基於 002） | **0.9100** (0.8992) | **是** | 加強正則化，sup +1.1%、SSL +0.75% |
| 006-droppath-0.3 | drop_path_rate 0.2 → 0.3（基於 005） | 0.9067 (0.8942) | 否 | 正則化過強，維持 0.2 |
| 007-act-relu | activation GELU → ReLU（基於 005） | 0.9025 (0.8942) | 否 | activation 比較用 |
| 008-act-leaky-relu | activation GELU → LeakyReLU(0.1)（基於 005） | 0.9075 (0.8983) | 否 | activation 比較用 |
| 009-thr-0.85 | pseudo threshold 0.9 → 0.85（基於 005） | 0.9075 (0.8992) | 否 | pseudo 3221 張（005 為 2761），多收的雜訊抵銷數量優勢；維持 0.9 |
| 010-wd-0.1 | weight decay 0.05 → 0.1（兩階段，基於 005） | 0.9033 (0.8983) | 否 | 維持 0.05 |
| 011-crop-0.4 | RandomResizedCrop scale (0.6,1) → (0.4,1)（基於 005） | 0.9092 (0.8958) | 否 | 接近最佳但未超越；維持 0.6 |
| 012-bs-32 | batch size 64 → 32（基於 005） | 0.9092 (0.8933) | 否 | 只差 1 張 val 圖；維持 64 |
| 013-width-512 | widths 最後一層 384 → 512（基於 005） | **0.9117** (0.8967) | **是** | 只比 005 多對 2 張，在 val 雜訊內；參數 11.25M、pseudo 2716 張 |
| 014-ssl200 | self-training epochs 100 → 200（基於 013） | **0.9175** (0.8967) | **是** | 比 013 多對 7 張，剛超過 0.5% 雜訊門檻；pseudo 3046 張（013 為 2716）；最佳在 SSL ep160，val loss 0.623 |
| 015-pl-interval-5 | pseudo-label 間隔 N 10 → 5（基於 014） | **0.9183** (0.8967) | **是** | 只比 014 多對 1 張，在 val 雜訊內；pseudo 3160 張（014 為 3046）；最佳在 SSL ep152，val loss 0.623 |
| 016-erasing-0.5 | RandomErasing p 0.25 → 0.5（基於 015） | **0.9325** (0.9042) | **是** | +17 張（+1.42%），明顯超出雜訊；sup 也 +0.75%、train/val 差距縮小；pseudo 3146 張；最佳在 SSL ep142，val loss 0.599 |
| 017-erasing-0.75 | RandomErasing p 0.5 → 0.75（基於 016） | 0.9242 (0.9033) | 否 | 遮擋過多，val loss 0.616 > 016 的 0.599；維持 p=0.5 |
| 018-img144 | IMG_SIZE 128 → 144（基於 016） | 0.9225 (0.9075) | 否 | sup 較好（+0.33%）但 SSL 較差；耗時 32 min；維持 128 |
| 019-depth-2232 | depths (2,2,2,2) → (2,2,3,2)（基於 016） | 0.9217 (0.9000) | 否 | 參數 12.44M；加深沒有幫助，維持 (2,2,2,2) |
| 020-act-relu | activation GELU → ReLU（基於 016） | 0.9258 (0.9050) | 否 | activation 比較用（新基礎） |
| 021-act-leaky-relu | activation GELU → LeakyReLU(0.1)（基於 016） | 0.9075 (0.8958) | 否 | activation 比較用（新基礎） |
| 022-head-drop-0.4 | head_dropout 0.2 → 0.4（基於 016） | 0.9192 (0.9025) | 否 | 維持 0.2 |
| 023-crop-0.4 | RandomResizedCrop scale (0.6,1) → (0.4,1)（基於 016） | 0.9175 (0.9000) | 否 | 搭配 erasing 0.5 後裁切過強；維持 (0.6,1) |
| 024-ls-0.0 | label smoothing 0.1 → 0.0（兩階段，基於 016） | 0.9092 (0.8917) | 否 | 模型過度自信，pseudo 收 3570/3600 幾乎全收，含大量錯誤標籤；LS 0.1 同時在幫 threshold 過濾，維持 0.1 |
| 025-laurel | use_laurel False → True（基於 016） | 0.9258 (0.9083) | 否 | sup 較好（+0.42%）但 SSL 較差；維持 False |
| 026-se-8 | se_reduction 16 → 8（基於 016） | 0.9117 (0.8908) | 否 | 維持 16 |
| 027-sup-lr-2e-3 | supervised lr 1e-3 → 2e-3（基於 016） | 0.9200 (0.9092) | 否 | sup 較好（+0.5%）但 SSL 較差；維持 1e-3 |
| 028-ssl-lr-1e-3 | self-training lr 5e-4 → 1e-3（基於 016） | 0.9217 (0.9042) | 否 | 004 調低也變差，維持 5e-4 |
| 029-no-trivial-aug | TrivialAugmentWide 開 → 關（基於 016） | 0.8975 (0.8850) | 否 | -3.5%，augmentation 是關鍵；維持開 |
| 030-ssl300 | self-training epochs 200 → 300（基於 016） | 0.9242 (0.9042) | 否 | 耗時 36 min；維持 200 |
| 031-wd-0.02 | weight decay 0.05 → 0.02（兩階段，基於 016） | 0.9300 (0.9017) | 否 | 只差 3 張，在雜訊內；維持 0.05 |
| 032-droppath-0.15 | drop_path_rate 0.2 → 0.15（基於 016） | 0.9233 (0.9000) | 否 | 維持 0.2 |
| 033-thr-0.95 | pseudo threshold 0.9 → 0.95（基於 016） | 0.9158 (0.9042) | 否 | pseudo 只收 1299 張（LS 0.1 讓信心上限約 0.92），資料不足；維持 0.9 |
| 034-erasing-0.6 | RandomErasing p 0.5 → 0.6（基於 016） | 0.9158 (0.9050) | 否 | 0.6 / 0.75 都較差，維持 0.5 |
| 035-ssl-warmup-5 | self-training warmup 0 → 5 epochs，lr 不變（基於 016） | 0.9250 (0.9042) | 否 | 維持不 warmup |
| 036-thr-0.85 | pseudo threshold 0.9 → 0.85（基於 016） | 0.9225 (0.9042) | 否 | pseudo 3466 張（016 為 3146），多收的雜訊抵銷數量；新基礎下重測，結論同 009；維持 0.9 |
| 037-erasing-0.4 | RandomErasing p 0.5 → 0.4（基於 016） | 0.9217 (0.9075) | 否 | erasing 掃描：0.4 / 0.5 / 0.6 / 0.75 → 0.9217 / **0.9325** / 0.9158 / 0.9242；維持 0.5 |
| 038-bs-128 | batch size 64 → 128（基於 016） | 0.9133 (0.8842) | 否 | 步數減半、sup -2%；（012 的 bs 32 是在 005 基礎上測）維持 64 |
| 039-sup-warmup-10 | supervised warmup 5 → 10 epochs（基於 016） | 0.9233 (0.9000) | 否 | 維持 5 |
| 040-head-drop-0.1 | head_dropout 0.2 → 0.1（基於 016） | 0.9250 (0.9000) | 否 | 0.1 / 0.4 都較差，維持 0.2 |

## Activation comparison

### 目前最佳設定（016：widths 512、sup 200 / SSL 200 ep、N=5、drop_path 0.2、RandomErasing p=0.5），只換 activation

| Activation | 編號 | sup val | self-training val |
|---|---|---|---|
| GELU | 016-erasing-0.5 | 0.9042 | **0.9325** |
| ReLU | 020-act-relu | 0.9050 | 0.9258 |
| LeakyReLU (0.1) | 021-act-leaky-relu | 0.8958 | 0.9075 |

結論（report 用這組）：self-training 後 GELU > ReLU > LeakyReLU。GELU 比 ReLU 高 0.67%（8 張），在雜訊邊緣；比 LeakyReLU 高 2.5%（30 張），差距明顯。supervised 階段三者差距 ≤ 0.92%，差異主要在 self-training 階段放大；三者 pseudo 數量相近（3146 / 3171 / 3153），推測是 pseudo-label 的正確率不同、錯誤會在後續輪次累積（val 無法直接驗證 pseudo 正確率）。

### 舊設定參考（005：widths 384、sup 200 / SSL 100 ep、N=10、RandomErasing p=0.25）

| Activation | 編號 | sup val | self-training val |
|---|---|---|---|
| GELU | 001-init | 0.8842 | 0.8958 |
| GELU | 005-droppath-0.2 | 0.8992 | 0.9100 |
| ReLU | 007-act-relu | 0.8942 | 0.9025 |
| LeakyReLU (0.1) | 008-act-leaky-relu | 0.8983 | 0.9075 |

舊設定下的排名是 GELU ≥ LeakyReLU > ReLU，但差距都在雜訊內。兩組設定都是 GELU 最好；ReLU 和 LeakyReLU 的相對順序不穩定，表示兩者差異在 seed/設定雜訊範圍內。
