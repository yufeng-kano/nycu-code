# IoT HW5：Cyber Timer

在 IoTtalk GUI 把 Timer 的 DF 改名成時間區間（例如 `18:00~18:05`），區間內 Timer 送 1，區間外送 0。Join 到 Bulb 後，燈只在區間內亮。

```
SA_Timer.py ── Time1 (0/1) ── Join 1: ×100 ──▶ Bulb Luminance-O (0–100 %)
     ▲
     └── DAN.get_alias('Time1') → '18:00~18:05'
```

## 檔案

| 檔案 | 用途 |
|---|---|
| `Dummy_Device_IoTtalk_v1_py/` | submodule，IoTtalk 官方 `DAI.py` / `DAN.py` / `csmapi.py` |
| `SA_Timer.py` | Timer 裝置：IDF `Time1`、`Time2`、`Time3`，每秒讀 alias 判斷是否在區間內 |

## 1. IoTtalk GUI 設定（https://classgui.iottalk.tw）

課程伺服器已有 `Timer`（Time1–Time9）與 `Bulb`（Luminance-O）兩個 Device Model，直接沿用。alias 存在專案裡的裝置上，不會和其他同學衝突。以下已在 GUI 建好：

1. Project：`315834011_HW5`
2. Model：`Timer` 勾 Time1–Time3；`Bulb` 勾 Luminance-O
3. Join 1：`Time1` → `Luminance-O`
   - Time1 的 IDF function：`315834011_hw5_x100`（`return args[0] * 100`）。Bulb 網頁的亮度是 0–100 %，直接送 1 只有 1 % 亮度，看起來是暗的
4. 點 Timer 方塊上的 `Time1`，右側表頭的鉛筆改名成時間區間，按 Save
5. 確認上方專案開關是開啟（綠燈）

## 2. 執行

需要先裝 [uv](https://docs.astral.sh/uv/)。clone 後先 `git submodule update --init`，在本資料夾執行：

```bash
PYTHONPATH=. uv run python Dummy_Device_IoTtalk_v1_py/DAI.py SA_Timer
```

註冊後到 GUI 把 Timer 方塊綁到 terminal 印出的 `Device name`。Bulb 開 https://class.iottalk.tw/da/Bulb/ ，把 Bulb 方塊綁到網頁標題的裝置名稱。

## 3. 預期結果

| Time1 名稱 | 目前時間 | Bulb |
|---|---|---|
| `18:00~18:05` | 18:02 | 亮 |
| `18:00~18:05` | 18:06 | 暗 |
| `23:00~01:00` | 00:30 | 亮（跨午夜） |
| `Time1`（未設定） | 任何時間 | 暗 |

## 設計說明

- **每秒重讀 alias**：在 GUI 改時間不必重啟 SA。代價是每個 DF 每秒一次 HTTP 請求
- **兩種時間格式都收**：作業提示用 `16:00:00~17:30:00`，投影片用 `18:00~18:05`，`SA_Timer.py` 依冒號數量選 `strptime` 格式
- **未設定或格式錯誤時送 0**：依 `DAN.py`，`get_alias` 出錯（例如裝置還沒綁定）時回傳 `None`；實測未改名時回傳 DF 名稱 `['Time1']`。提示寫的 `(DAN.get_alias('Time1'))[0]` 直接用會在這兩種情況出錯
- **結束時間早於開始時間視為跨午夜**：`23:00~01:00` 代表晚上 11 點到隔天 1 點
- **反向控制**：要區間內關、區間外開，把 IDF function 換成 `return (args[0] ^ 1) * 100`（投影片第 7 頁的 `args[0]^1`）
