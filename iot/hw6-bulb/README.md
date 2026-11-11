# IoT HW6：Cyber Bulb（JavaScript DA）

把課程附件 `Bulb_origin` 的燈泡網頁動畫接上 IoTtalk，變成 ODF 裝置 Bulb。顏色和亮度由 IoTtalk 送進來，網頁標題顯示註冊後的 Device Name。

```
Remote_control ── Color-I1 ── Join 1 ──▶ Bulb Color-O      (R, G, B: 0–255)
               └─ Slider1  ── Join 2 ──▶ Bulb Luminance-O  (0–100 %)
```

## 檔案

| 檔案 | 用途 |
|---|---|
| `index.html` | `Bulb_origin/index.html` 加上 DA 模板的 script，標題文字加 `class="Device_name"` |
| `css/style.css` | `Bulb_origin` 原檔，未修改 |
| `js/sa.js` | 本作業的修改：伺服器設定、`Color_O` / `Luminance_O` 兩個 ODF、`sa_init` 寫入 Device Name |
| `js/` 其餘檔案 | [Dummy_Device_IoTtalk_v1_js](https://github.com/IoTtalk/Dummy_Device_IoTtalk_v1_js) 原檔（`dai.js`、`dan.js`、`csmapi.js`、`httponly.js` 與函式庫） |

## 1. IoTtalk GUI 設定（https://classgui.iottalk.tw）

課程伺服器已有 `Bulb`（Color-O、Luminance-O）與 `Remote_control` 兩個 Device Model，直接沿用。以下已在 GUI 建好：

1. Project：`315834011_HW6`
2. Model：`Remote_control` 選 Color-I × 1、Slider × 1；`Bulb` 勾 Color-O、Luminance-O
3. Join 1：`Color-I1` → `Color-O`；Join 2：`Slider1` → `Luminance-O`。兩條都用預設 function
4. 確認上方專案開關是開啟（綠燈）

## 2. 執行

在本資料夾起一個靜態伺服器：

```bash
uv run --no-project python -m http.server 8606 --bind 127.0.0.1
```

1. 開 http://127.0.0.1:8606/ ，標題從 `Registering` 變成裝置名稱（例如 `f45c.Bulb`）
2. 到 GUI 點 Bulb 方塊的名稱，綁到網頁標題顯示的那個裝置
3. 點 Remote_control 方塊的齒輪，開 `Remote_control Link`（https://class.iottalk.tw/RemoteControl/315834011_HW6 ）
4. 在遙控器頁面點色塊改顏色，拉 Slider1 改亮度

每次重新整理 Bulb 網頁都會註冊成新的裝置名稱，要重新綁定。

## 3. 預期結果

| Color-I1 | Slider1 | 燈泡顏色 |
|---|---|---|
| 藍 | 100 | `rgb(0, 0, 255)` |
| 洋紅 | 40 | `rgb(102, 0, 102)` |
| 任何顏色 | 0 | 黑（全暗） |

## 設計說明

- **Device Name**：`dan.js` 註冊成功後把伺服器回傳的名稱寫回 `profile.d_name`，`sa_init` 用 `$('.Device_name')[0].innerText` 顯示
- **ODF 命名**：`dai.js` 會把函式名結尾的 `_O` 轉成 `-O`，所以 `Color_O` 對應 DF `Color-O`
- **亮度 0–100**：沿用 `Bulb_origin/js/color.js` 的算法，每個色頻乘上 `lum / 100`。Slider 的 0–100 直接對應，不需要 Join function
- **輸入值夾限**：顏色夾在 0–255、亮度夾在 0–100，非數字當成 0，避免 Join function 送出超界的值讓 CSS 顏色失效
- **MQTT**：`mqtt_url` 等設定取自課程伺服器 `https://class.iottalk.tw/static/da_mqtt_config.js`。連不上時在 console 執行 `http.only()` 改走 HTTP，並在 GUI 重啟專案
- **`color.js` 併入 `sa.js`**：原本的 `draw()` 需要讀寫 r、g、b、lum，和 ODF 函式放在同一個 closure 最直接
