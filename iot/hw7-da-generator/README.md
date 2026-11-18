# IoT HW7：用 DA Generator 串接微氣象站與 LineBot

用 IoTtalk 的 DA Generator（DeviceTalk）產生兩個 DA：M2 微氣象站（輸入）與 LineBot（輸出）。氣象站的 CO2、濕度、溫度經 IoTtalk 送到 LineBot，推播到 LINE。

```
farm.iottalk.tw ─ sensors.py ─▶ M2_315834011 DA ── CO2 ─────────┐ IDF function 315834011_hw7_label
                                                └─ Humidity ────┤ （用 idf_df_name_ 標上來源）
                                                └─ Temperature ─┤
                                                                ▼ Join function 315834011_hw7
LINE ◀─ LineBot.pushLineMsg ◀─ LineBot_315834011 DA ◀── Msg-O [來源, 整則訊息]
```

## 檔案

| 檔案 | 用途 |
|---|---|
| `M2_315834011/` | DeviceTalk 產生的 M2 DA，另外放入 `sensors.py` |
| `LineBot_315834011/` | DeviceTalk 產生的 LineBot DA，另外放入 `LineBot.py`、`config.py`、`userIdFile` |
| `*/SA.py`、`*/libraries/*/safuncs.py` | DeviceTalk 依網頁上填的設定與函式產生，未手動修改 |
| `*/DAI.py`、`DAN.py`、`csmapi.py`、`Agent.py` | DeviceTalk 附帶的 IoTtalk DA 程式庫 |
| `gui_functions.py` | GUI 上 IDF function 與 Join function 的副本 |
| `LineBot_315834011/config.example.py` | 複製成 `config.py` 填入 LINE channel 資訊（`config.py`、`userIdFile` 已 git-ignore） |

`sensors.py` 來自課程附件 `M2_library.zip`，只改了 `on_disconnect` 的參數，配合 paho-mqtt 2.x 的 callback 格式（同 HW3、HW4）。`LineBot.py` 來自 [LineBot-basic](https://github.com/IoTtalk/LineBot-basic)，未修改。

## 1. IoTtalk GUI 設定（https://classgui.iottalk.tw）

沿用 HW3、HW4 建好的 Device Model `M2_315834011`（CO2、Humidity、Temperature）與 `LineBot_315834011`（Msg-I、Msg-O）。以下已在 GUI 建好：

1. Project：`315834011_HW7`，放入 `M2_315834011`（三個 IDF 全勾）與 `LineBot_315834011`（只勾 `Msg-O_315834011`）
2. Join 2：CO2、Humidity、Temperature → `Msg-O_315834011`
   - 三個 IDF 的 Function 都選 `315834011_hw7_label`
   - Join Function：`315834011_hw7`
   - ODF：y1=`315834011_hw4_y1`、y2=`315834011_hw4_y2`（HW4 建的，取 join 結果的第 1、2 個元素）
3. 確認上方專案開關是開啟

## 2. 用 DA Generator 產生 DA

在 GUI 點方塊的齒輪 → `DA`，開啟 DeviceTalk。

1. Device Name 選 `Add new device`，名稱必須以英文字母或底線開頭（`M2_315834011`、`LineBot_315834011`）
2. Global Variables 填共用的 import 與物件；每個 DF 選 `Add new function`，在 `def` 與 `return` 之間寫內容
   - M2：`get_co2`、`get_humidity`、`get_temperature`，分別回傳 `sensors.py` 的讀數，還沒有資料時回傳 `None`
   - LineBot：`push_message`，收到 `[來源, 訊息]` 後呼叫 `LineBot.pushLineMsg`
3. DA 分頁確認 IoTtalk Server 為 `https://class.iottalk.tw`、Push Interval 1 秒，按 `Download SA Code`
4. 解壓縮，把作業要求的檔案放進資料夾，否則 import 會失敗

## 3. 執行

需要先裝 [uv](https://docs.astral.sh/uv/)。開兩個 terminal，分別在兩個 DA 資料夾執行（`userIdFile` 用相對路徑讀取，所以要在 LineBot 資料夾內執行）：

```bash
# Terminal 1
cd M2_315834011 && uv run --project .. python DAI.py

# Terminal 2
cd LineBot_315834011 && uv run --project .. python DAI.py
```

`SA.py` 已經寫好 `device_name`，註冊後會出現 `Device name: M2_315834011`。到 GUI 點兩個方塊的名稱綁定裝置（名稱變藍）。

## 4. 預期結果

LINE 收到：

```
M2 weather station
CO2: 415.4 ppm
Humidity: 62.6 %
Temperature: 27.4 °C
```

LineBot 啟動後收到第一筆就推播，之後最多每 10 分鐘一則。

## 設計說明

- **`idf_df_name_` 只在 IDF function 有定義**：放在 Join function 會 `NameError`，專案開關被伺服器關掉。所以由三個 IDF 共用一個 `315834011_hw7_label`，各自把讀數標上來源名稱，Join function 再合成一則訊息
- **為什麼要 Join function**：Msg-O 只有 2 維，三個 IDF 的結果直接送過去只留得下前兩個。合成一則後 y1 放來源、y2 放訊息
- **推播節流**：Join 每秒觸發一次，但 LINE 免費方案每月只能 push 200 則。`push_message` 每 10 分鐘最多推一次（`PUSH_INTERVAL`）
- **Global Variables 重複**：DeviceTalk 把 Global Variables 複製到每個函式前面，`safuncs.py` 裡的 `import sensors` 與物件建立出現三次。重複建立不影響結果，因為讀數都存在 `sensors.py` 的模組變數
- **LineBot 只做輸出**：只 import `LineBot` 用 `pushLineMsg`，不啟動 Flask webhook。推播對象是 `userIdFile` 裡的使用者（HW4 傳過訊息給 bot 時記錄的）
- **MQTT broker**：DeviceTalk 產生的 `SA.py` 用 `iot.iottalk.tw`，和 HW4 手寫的 `class.iottalk.tw` 不同，沿用產生的設定
