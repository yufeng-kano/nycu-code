# DL Lab 4 — Experiment Summary

**目前最佳（單模型，最終只能交一個模型一次訓練）：最終 run（使用者 2026-10-30 在 Colab G4 跑 `HW4_315834011_final.ipynb`：011 配方、150 epoch），val mIoU 0.627（EMA 0.628），hflip TTA 0.633，推論鏈（五尺度 512–768、hflip、小類別抑制 50）0.6535，Kaggle public 0.658；歷史最佳 ensemble 003+008 加推論鏈 val 0.636、Kaggle public 0.640，已不採用**

評分指標：逐張 mIoU 再平均（val 1,000 張），與 Kaggle 計分方式一致。表中 val 為最佳 epoch 的 val mIoU；括號內為 hflip TTA 後的數字。

## Experiments

| 編號 | 架構 / 改動 | val mIoU | 是否破紀錄 | 備註 |
|---|---|---|---|---|
| U-Net（HW4 notebook） | 作業要求的原始 U-Net 31.0M，batch 16，AdamW 1e-3 OneCycle，20 epoch，bf16，以 val mIoU 選 checkpoint | 0.520（epoch 18；最後 epoch 0.519） | — | 繳交的 `HW4_315834011_final.ipynb` 內的 run：使用者 2026-10-30 在 Colab G4 與 mini-SAM 同一次執行，51 分鐘（每 epoch 2.55 分）；報告引用這個數字與曲線。較早使用者在本機 3090 Ti 以 U-Net 版 notebook 跑過一次：0.522（epoch 18；最後 0.520），2026-10-29 17:05 完成，每 epoch 約 6 分鐘，`output.csv` 0.87 GB、3,961 列，保留為備案，Kaggle 未上傳。`scripts/tools/predict.py` 的逐張 mIoU 與 notebook 模板的 mIoU 函式在同一 checkpoint 上算出一樣的數字，所以腳本/Colab checkpoint 的 val 可直接和 notebook 數字比較 |
| 001-mini-sam | Hiera 式階層 transformer 編碼器 + FPN + SAM 式 two-way mask decoder，從零訓練，40 epoch | 0.403（EMA，停在 epoch 7） | — | 17.1M 參數；3090 Ti 共用時每 epoch 約 4.5 分鐘；G4 就緒後手動停止，改跑 002 |
| 002-mini-sam-g4 | 001 同架構，100 epoch，warmup 5，Colab G4（session lab4） | 0.575（0.581） | 是 | Kaggle public 0.583；115 分鐘；最佳在 epoch 100（raw，EMA 0.573），曲線尾端仍微升；弱項 glove 0.25、scarf 0.29、socks 0.35、skirt 0.36、鞋 0.36、sunglasses 0.47；每 epoch 1.1 分鐘 |
| 003-mini-sam-wide | 002 加寬：stem 48、dims 96/192/384/768、depths 2/2/8/2、heads 3/6/12/24、drop path 0.2、lr 6e-4，Colab G4（session lab4b） | 0.588（0.596） | 是 | 143 分鐘；最佳在 epoch 100（raw，EMA 0.587），尾端仍微升；比 002 +0.013，主要來自 coat、上衣、手臂；小部件仍弱：glove 0.29、skirt 0.30、scarf 0.31、socks 0.33、鞋 0.37 |
| 004-mini-sam-hr | 002 加 stride-2 mask head（logits 160×160，接 conv stem 的 stride-2 特徵）、邊緣輔助分支（BCE，權重 0.3）、逐張 Dice，Colab G4（session lab4） | 0.556（0.569） | 否 | 比 002 低 0.019；三項改動一起上（stride-2 head、邊緣分支、逐張 Dice）結果變差，整包捨棄；TTA 增益較大（+0.013）；每 epoch 1.2 分鐘 |
| 005-mini-sam-cctt | 002 的前兩段改 ConvNeXt block（C-C-T-T），其餘同 002，Colab G4（session lab4b） | 0.574（0.581） | 否 | 與 002 相同（0.575 / 0.581），前兩段改卷積沒有差別；每 epoch 1.1 分鐘 |
| 008-mini-sam-wide-deep | 003 的第三段加深：depths 2/2/8/2 → 2/2/12/2，其餘同 003，Colab G4（session lab4） | 0.593（0.599） | 是 | 比 003 +0.005；159 分鐘，每 epoch 1.55 分鐘 |
| 009-mini-sam-wide-150ep | 003 同設定、150 epoch（同樣的 cosine 拉長），Colab G4（session lab4b） | 0.599（0.606） | 是 | 211 分鐘；最佳在 epoch 150（raw 0.5977、EMA 0.5990，尾端仍微升）；比 003 +0.010、比 008 +0.007：拉長訓練是目前最便宜的增益，最終配方採 150 epoch |
| 011-mini-sam-448 | 008 同設定，但影像與 mask 放大到 448 訓練與推論（logits 縮回 320 計分），Colab G4（session lab4） | 0.615（0.621） | 是 | 289 分鐘（每 epoch 2.8 分，008 的 1.8 倍）；最佳在 epoch 100（raw 0.6154、EMA 0.6143，尾端仍微升）；比 008 +0.022、比 009（150 epoch）+0.015：原生 448 是目前最大的單一增益；小部件仍最弱：glove 0.31、scarf 0.31、socks 0.40、skirt 0.42、鞋 0.42 |
| 014-mini-sam-lovasz | 008 同設定，loss 的 batch Dice 換成逐張 Lovász-softmax（只算該張出現的類別，即排行榜指標的可微替代；B×C 條誤差向量一次排序），Colab G4（session lab4b） | 0.572（0.583） | 否 | 188 分鐘（每 epoch 1.8 分，排序多花 20%）；前 10 個 epoch 領先 008 約 0.02，epoch 30 起落後，最終比 008 低 0.021；所有小部件也都更差（socks 0.32、鞋 0.33）：Lovász 不採用，最終配方維持 CE + Dice |
| 最終 run（`HW4_315834011_final.ipynb`） | 011 同設定（008 模型、原生 448、CE + batch Dice）、150 epoch；使用者本人在 Colab G4 以最終 notebook 一次跑完（同一 notebook 前段先跑模板 U-Net 20 epoch） | 0.627（0.633） | 是 | 420 分鐘（每 epoch 2.80 分）；最佳 raw 0.6270 在 epoch 140、EMA 0.6282 在 epoch 146（存檔的 checkpoint `checkpoints/minisam-final-best.pt`）；比 011 +0.012；推論鏈 0.6535（見下表）；小部件：scarf 0.33、glove 0.35、socks 0.37、鞋 0.42/0.43、skirt 0.45、sunglasses 0.49；notebook 內的 U-Net 在 G4 上 51 分鐘、最佳 0.520（epoch 18）；CSV `result/minisam-final.csv`（845 MB、3,961 列已驗證）2026-10-30 14:08 UTC 上傳 Kaggle（submission 57807731），public 0.65835，比舊 ensemble 的 0.640 高 0.018 |

## 推論端（不需重訓，val 上以 003 量測，`scripts/tools/ensemble.py`）

| 設定 | val mIoU |
|---|---|
| 320，hflip TTA（訓練解析度） | 0.596 |
| 448 單一尺度 | 0.611 |
| 320/384/448/512 四尺度平均 | 0.620 |
| 四尺度 + 抑制面積小於 50 像素的預測類別 | 0.629 |
| 009 單模型加完整推論鏈（四尺度、hflip、min-area 50） | 0.6355 |
| 011（448 訓練）448 單尺度 + hflip | 0.6211 |
| 011 448 單尺度 + hflip + min-area 50 | 0.6275 |
| 011 384/448/512 + hflip + min-area 50 | 0.6320 |
| 011 448/512/576/640 + hflip + min-area 50 | 0.6428 |
| 011 512/576/640/704/768 + hflip + min-area 50（最終採用） | 0.6445 |
| 011 512/576/640/704/768 + hflip，min-area 30 / 80 | 0.6448 / 0.6423 |
| 011 576…832 或 512…896 七尺度 | 0.6427 / 0.6432 |
| 最終 run 448 單尺度 + hflip | 0.6337 |
| 最終 run 512/576/640/704/768 + hflip | 0.6501 |
| 最終 run 512/576/640/704/768 + hflip + min-area 50（上傳設定，notebook 內量測） | 0.6535 |

多尺度靠把 decoder 的位置編碼做 bicubic 縮放，只有 64 的倍數可用（window 切分）。448 訓練的模型在高於訓練解析度的尺度上繼續受益，到 768 為止（832、896 反而下降）。小類別抑制有效是因為指標逐張對「標註或預測出現的類別」平均，一個誤報類別就多一個 0 分項。最終 CSV 用五尺度（512–768）+ hflip + 抑制 50。

同一條推論鏈套在 008：0.630；003+008 softmax 平均：0.636（加 002 變 0.634；008 權重 1.5 無差）。`result/ens-003-008-ms-ma50.csv` 已送 Kaggle，public 0.640（2026-10-29 23:30）。
