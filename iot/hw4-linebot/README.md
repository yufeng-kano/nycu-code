# IoT HW4：LineBot 查詢溫溼度

在 LINE 問溫度或濕度，Bot 回傳農場氣象站目前的數值。其他問題一律回覆 `Only temperature and humidity inquiries are supported.`

```
farm.iottalk.tw ─ sensors.py ─▶ SA_M2.py ── Temperature_315834011 ─┐
                                         └─ Humidity_315834011 ────┤
LINE ─ ngrok ─▶ SA_LineBot.py (Flask) ── Msg-I ────────────────────┤ Multi-join
                     ▲                                              │ join_function.py
                     └──────────────── Msg-O [問題, 回覆] ◀─────────┘
```

## 檔案

| 檔案 | 用途 |
|---|---|
| `LineBot-basic/` | submodule，課程 LineBot 範本（Flask webhook、push message） |
| `Dummy_Device_IoTtalk_v1_py/` | submodule，IoTtalk 官方 `DAI.py` / `DAN.py` / `csmapi.py` |
| `SA_LineBot.py` | LineBot 裝置：IDF `Msg-I_315834011`、ODF `Msg-O_315834011`，同時啟動 Flask |
| `SA_M2.py`、`sensors.py` | 氣象站，沿用 HW1 的 `M2_315834011` |
| `join_function.py` | 貼到 IoTtalk GUI 的 join function |
| `config.example.py` | 複製成 `config.py` 填入 LINE channel 資訊（已 git-ignore） |

## 1. LINE 與 ngrok

照 `LineBot-basic/CreateYourLineBot.pdf` 操作：

1. 建立 Messaging API channel，記下 Channel secret 與 Channel access token
2. `cp config.example.py config.py`，填入上面兩個值
3. `ngrok http 32768`，把 HTTPS 網址填到 LINE 後台的 Webhook URL
4. 開啟 Use webhook，關閉 Auto-reply

## 2. IoTtalk GUI 設定（https://classgui.iottalk.tw）

伺服器上的 DF 全班共用，所以名稱都加學號（作業原文的 `Msg-I` / `Msg-O` 對應下面兩個）。以下已在 GUI 建好：

1. Device Feature
   - IDF `Msg-I_315834011`：1 維，string
   - ODF `Msg-O_315834011`：2 維，string、string（y1 問題原文、y2 回覆）
2. Device Model：`LineBot_315834011`，含上面兩個 DF
3. Project：`315834011_HW4`，放入 `M2_315834011`（只勾溫度、濕度）與 `LineBot_315834011`
4. Join 1：`Temperature_315834011`、`Humidity_315834011`、`Msg-I_315834011` → `Msg-O_315834011`
   - IDF (Line)：面板依序是濕度 1、溫度 2、Msg-I 3，所以設 z1=2、z2=1、z3=3，讓 `run()` 收到溫度、濕度、Msg-I
   - Join Function：`315834011_hw4`，內容就是 `join_function.py`
   - ODF：y1=`315834011_hw4_y1`、y2=`315834011_hw4_y2`，分別取 join 結果的第 1、2 個元素
5. 確認上方專案開關是開啟（綠燈）。ODF function 出錯時伺服器會把整個 project 停掉，開關會變紅

## 3. 執行

需要先裝 [uv](https://docs.astral.sh/uv/)。clone 後先 `git submodule update --init`。開兩個 terminal，都在本資料夾執行：

```bash
# Terminal 1: weather station
PYTHONPATH=. uv run python Dummy_Device_IoTtalk_v1_py/DAI.py SA_M2

# Terminal 2: LineBot
PYTHONPATH=.:LineBot-basic uv run python Dummy_Device_IoTtalk_v1_py/DAI.py SA_LineBot
```

`PYTHONPATH` 讓 `DAI.py` 找得到本資料夾的 SA 與 `config.py`。註冊成功後，到 GUI 把兩個方塊綁到 terminal 印出的 `Device name`，等出現 `Device state: RESUME.`

## 4. 預期結果

| LINE 訊息 | Bot 回覆 |
|---|---|
| `溫度`、`temperature` | `Current temperature: 25.3 °C` |
| `濕度`、`humidity` | `Current humidity: 61.2 %` |
| 其他 | `Only temperature and humidity inquiries are supported.` |

## 設計說明

- **為什麼 Msg-O_315834011 有 2 維**：溫溼度每秒更新，每次更新都會重新觸發 join。只回覆字串的話，LINE 會每秒收到一則。所以 join 把問題原文一起送回，`SA_LineBot.py` 只在問題對得上「還在等回覆的那則」時推送一次
- **為什麼不用內建的 x1 / x2**：伺服器把 join 的結果整包當成一個參數交給 ODF function，內建 `x2`（`return args[1]`）會 `IndexError`，整個 project 因此停止。自訂的 y1 / y2 改從 `args[0]` 取值
- **回覆用 push 而非 reply**：沿用課程 `LineBot.pushLineMsg`，會推給所有傳過訊息的使用者。LINE 免費方案的 push 每月有則數上限
- **剛啟動時沒有溫溼度**：`sensors.py` 約 20 秒後才有第一筆，這段期間會回覆 `... is not available yet.`
- **不要同時跑 HW1 的 `SA_M2.py`**：兩份用同一個 `device_id`，會互相搶註冊

## HW2

用 ChatGPT 完成和 HW1 相同的作業，結果對錯不計，把對話的分享連結（ChatGPT 右上角 Share → Copy link）交到 E3。
