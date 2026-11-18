# 色彩量化與 Dithering 還原

請繳交一個 Jupyter Notebook（`.ipynb`）。Notebook 必須包含程式碼、執行結果（圖與表）以及文字回答。確切的內容清單請見最後的「Notebook 必須包含的內容」。

本作業分成兩部分：

- **Part A** 要你自學 clustering（分群），方法是閱讀 scikit-learn 的官方文件。課堂上沒有教 clustering。
- **Part B** 用監督式學習（回歸）把 dithering 還原。

> **必須取樣。** 為了縮短執行時間，作業中標明要取樣的地方，都必須用隨機取樣的像素來 fit。照規定取樣的話，整份 notebook 在 2 核 CPU（例如免費版 Google Colab）上大約 3 分鐘可以跑完。如果你的 notebook 跑得比這久很多，請先檢查取樣。

---

## Part A — 用 Clustering 做色彩量化（自學）

全彩影像最多可以有約 1670 萬種顏色。舊式螢幕、GIF 檔和許多嵌入式裝置，只能顯示少數幾種顏色組成的調色盤，例如 256 色。色彩量化就是選出一組好的調色盤，再把每個像素換成調色盤中最接近的顏色。

把每個像素看成 3D 色彩空間（R, G, B）中的一個點，選出 256 色的調色盤就變成分群問題：把像素分成 256 群，每群用一個代表色。

### A0. 閱讀

寫程式前請先閱讀：

- scikit-learn User Guide 的 Clustering 章節：最上方的總覽表，以及 K-Means、Mini Batch K-Means、Hierarchical（Agglomerative）clustering、BIRCH、DBSCAN 各節。
- `KMeans`、`MiniBatchKMeans`、`BisectingKMeans`、`AgglomerativeClustering`、`Birch` 的 API 頁面。
- User Guide 的 Gaussian Mixture Models（`GaussianMixture`）。
- scikit-learn 範例庫的 "Color Quantization using K-Means"。可以參考這個範例起步，但 Part A 的要求遠多於這個範例。

用自己的話回答以下問題，簡短即可：

1. K-Means 最小化的是什麼？什麼是「群中心」（cluster center）？
2. 總覽表中哪些方法可以直接指定群數？為什麼不能直接要求 DBSCAN 或 MeanShift 分出剛好 256 群？
3. 有些方法（例如 agglomerative clustering）沒有 `predict`，要怎麼把每個像素對應到調色盤顏色？
4. Agglomerative clustering 的時間與記憶體成本和樣本數是什麼關係？能直接對一張 640×427 影像的全部像素執行嗎？
5. 用像素的樣本來 fit 和用全部像素 fit 有什麼差別？各自得到和失去什麼？

### A1. 用 K-Means 做 256 色影像

- 載入 `sklearn.datasets.load_sample_image("china.jpg")`，像素值縮放到 [0, 1]。
- 用 `KMeans` 量化成 256 色。
- 原圖與量化結果並排顯示，並附上局部放大圖。
- 把調色盤畫成 16×16 的色塊格子。

### A2. 比較分群方法

比較至少 5 種方法，必須包含：

- `KMeans`
- `MiniBatchKMeans`
- `AgglomerativeClustering`，至少兩種不同的 linkage
- `BisectingKMeans`、`Birch`、`GaussianMixture` 中至少兩種

另外加入兩個不需學習的 baseline：

- 均勻調色盤，例如 R × G × B 分成 8 × 8 × 4 階
- 隨機調色盤：隨機挑選 256 個像素的顏色

**取樣規定：** 每種方法都用最多 10,000 個隨機取樣的像素來 fit（`AgglomerativeClustering` 最多 5,000 個），再把全部像素指派到最近的調色盤顏色。`KMeans` 和 `MiniBatchKMeans` 可以另外用全部像素跑一次作為對照。

每種方法都要報告：

- 執行時間（fit 加上指派）
- 重建誤差：MSE 和 PSNR，像素值在 [0, 1] 時 $\mathrm{PSNR} = 10\log_{10}(1/\mathrm{MSE})$
- 實際用到的調色盤顏色數

整組比較跑兩次：K = 256 和 K = 16。

> **提示：** K = 256 時，好的方法產生的影像幾乎和原圖一樣。請用局部放大、差異圖（|原圖 − 量化結果|）以及 K = 16 來讓差異看得出來。

### A3. 視覺化分群的效果

- 選兩種結果差異明顯的方法，各畫一張 3D 散佈圖：隨機取約 2,000 個像素放在 RGB 空間中，以其調色盤顏色上色。說明兩者的差別。
- 至少對兩種方法，畫出誤差對 K（K = 2, 4, 8, …, 256）的曲線。

### A4. 討論

用一段話回答：你會推薦哪種方法做色彩量化？為什麼？請考慮品質、速度與記憶體。至少說明一種表現不好的方法，並解釋它為什麼不適合這個任務。

---

## Part B — 用回歸還原 Dithering

Dithering 只用黑白兩色來呈現灰階影像。它把四捨五入的誤差分散到鄰近像素，因此白點的局部密度就代表灰階值。你的任務是訓練一個回歸模型，從 dithering 後的黑白影像還原出灰階影像。

### B1. 實作 Floyd–Steinberg dithering

寫一個函式 `fs_dither(img)`，輸入值域 [0, 1] 的灰階影像。由左到右、由上到下掃描每個像素：

1. 原值 ≥ 0.5 時新值設為 1，否則設為 0。
2. 計算誤差：誤差 = 原值 − 新值。
3. 把誤差加到還沒處理過的鄰居，權重為 7/16（右）、3/16（左下）、5/16（下）、1/16（右下）。

展示幾張影像與其 dithering 結果。

### B2. 資料集

使用 `skimage.data`（scikit-image）內建的影像。每張都轉成值域 [0, 1] 的灰階（`skimage.color.rgb2gray`），並縮放成 256×256（`skimage.transform.resize`）。

- **訓練影像：** coins、moon、chelsea、coffee、rocket、brick、grass、gravel、page。
- **測試影像：** camera、astronaut。

每張影像都做 dithering。絕對不要在訓練影像的像素上測試，並說明為什麼這樣會得到誤導性的分數。

### B3. 特徵與目標

每個像素的輸入是 dithering 影像中、以該像素為中心的 k × k 黑白區塊；目標是該像素原本的灰階值。

- 至少使用兩種區塊大小，例如 k = 7 和 k = 11。
- **提示：** 用 `numpy.lib.stride_tricks.sliding_window_view` 搭配 `np.pad`，不用 Python 迴圈就能建出所有區塊。
- **取樣規定：** 每個模型都用 60,000 個隨機取樣的訓練像素來訓練（9 張訓練影像共約 59 萬個像素）。所有模型使用同一組樣本，比較才公平。在兩張測試影像的全部像素上評估。

### B4. Baseline

對 dithering 影像做高斯模糊，是最簡單的「還原」方法。在訓練影像上選出最佳的 sigma，再報告它在測試影像上的 PSNR。你訓練的每個模型都要和這個 baseline 比較。

### B5. 模型

至少訓練並比較四種回歸模型，例如：

- `Ridge`
- `KNeighborsRegressor`
- `RandomForestRegressor`
- `HistGradientBoostingRegressor`
- `MLPRegressor`（注意：較新版的 scikit-learn 中，`hidden_layer_sizes=` 要用關鍵字參數傳入）

每個模型都要報告訓練時間、預測時間和測試 PSNR。並把還原結果和原圖、dithering 影像、模糊 baseline 並排顯示。

**必做：** 把學到的 Ridge 係數畫成 k × k 的影像。作用在區塊上的線性模型就是一個卷積濾波器。你學到的濾波器長什麼樣子？和高斯濾波器有什麼不同？

### B6. 泛化能力

- 只用一類影像訓練，例如只用紋理（brick、grass、gravel）或只用 page（文字），再於 camera 和 astronaut 上測試。結果變差多少？
- 討論模型學到的是什麼：訓練影像的內容，還是 dithering 這個過程本身？

### B7. 討論

哪個模型表現最好？比模糊 baseline 好多少？哪個模型最差？為什麼？

> **提示：** 想想高維二值向量之間的距離。

---

## Notebook 必須包含的內容

請依照下列順序，每一項用一個 Markdown 標題分隔。繳交前請執行所有 cell，確保每個輸出都看得到。

### Part A

- **A0** — 5 個閱讀問題的文字回答。
- **A1** — 圖：原圖與 256 色 K-Means 結果並排，附局部放大。圖：16×16 調色盤。
- **A2** — K = 256 與 K = 16 各一張比較表格，每列一種方法（至少 5 種方法加 2 個 baseline），欄位為：方法、使用的樣本數、執行時間（秒）、MSE、PSNR（dB）、實際使用顏色數。圖：兩種 K 下所有方法的局部放大並排比較。圖：至少兩種方法的差異圖。
- **A3** — 圖：兩種方法的 3D RGB 散佈圖。圖：至少兩種方法的誤差對 K 曲線（x 軸用對數刻度）。
- **A4** — 一段討論。

### Part B

- **B1** — `fs_dither` 函式。圖：至少 3 張影像與其 dithering 結果。
- **B2** — 載入、轉灰階、縮放、dithering 訓練與測試影像的程式碼。一句話說明為什麼不能在訓練像素上測試。
- **B3** — 建立區塊特徵的程式碼。寫明區塊大小與訓練樣本數。
- **B4** — 選出的 sigma，以及模糊 baseline 在每張測試影像上的 PSNR。
- **B5** — 結果表格，每列一組（模型、區塊大小），欄位為：模型、區塊大小、訓練時間（秒）、預測時間（秒）、camera 的 PSNR、astronaut 的 PSNR。表格中必須包含模糊 baseline。圖：每張測試影像的原圖、dithering、模糊、你最好的兩個模型，使用同一個局部放大區域。圖：Ridge 係數的 k × k 影像，並附上你的解釋。
- **B6** — 比較「全部訓練影像」與「只用一類影像」的表格，並簡短說明。
- **B7** — 一段討論。

### 一般要求

- 每一節都要清楚標示項目編號（A0、A1、A2、…、B7），方便找到各部分。
- 每張圖都要有標題，有意義的座標軸要有標籤。表格要標明單位。