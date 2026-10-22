# IoT HW3：M2_315834011 → Dummy_Device

把農場的真實感測器數值（CO2、濕度、溫度）透過自建的 Device Model `M2_315834011` 送上 IoTtalk，經過 Join 轉給 `Dummy_Device` 上自訂的 3 維 ODF `Display_315834011`，最後在 console 印出 3 個數值。

```
farm.iottalk.tw (08002700A202)
   │ MQTT (sensors.py)
   ▼
SA_M2.py ── CO2_315834011 / Humidity_315834011 / Temperature_315834011 ──▶ IoTtalk (class.iottalk.tw)
                                                                              │ Join
                                                                              ▼
SA_Dummy.py ◀── Display_315834011 (y1, y2, y3) ────────────────────────────────┘
   └─ print CO2, Humidity, Temperature
```

## 檔案

| 檔案 | 用途 |
|---|---|
| `Dummy_Device_IoTtalk_v1_py/` | submodule，IoTtalk 官方 `DAI.py` / `DAN.py` / `csmapi.py` |
| `sensors.py` | 課程提供的 M2_library，訂閱農場感測器。已修正 `on_disconnect` 為 paho-mqtt 2.x 的參數格式 |
| `SA_M2.py` | M2 端：`M2_315834011`，3 個 IDF，從 `sensors.py` 取值送出 |
| `SA_Dummy.py` | Dummy 端：`Dummy_Device`，只用自訂 ODF `Display_315834011`，收到就印出 3 個數值 |
| `SA.py` | 先前練習用的 `M315834011`，與 HW3 無關 |

## 1. IoTtalk GUI 設定（https://classgui.iottalk.tw）

1. Device Feature：建立 3 個 IDF，各 1 維
   - `CO2_315834011`、`Humidity_315834011`、`Temperature_315834011`
2. Device Feature：建立 1 個 ODF `Display_315834011`，**3 維**（y1、y2、y3）
   - 內建的 `Dummy_Control` 維度不夠，不能直接輸出 3 個值，所以自己定義一個
3. Device Model：
   - 建立 `M2_315834011`，加入上面 3 個 IDF
   - 在 `Dummy_Device` 加入 `Display_315834011`（`Dummy_Device` 可能是全班共用，只新增、不要改動或刪除既有的 feature）
4. Project：放入 `M2_315834011`（左）與 `Dummy_Device`（右），把 3 個 IDF 拉到同一個 Join，再連到 `Display_315834011`
5. 點 Join 圓點，在右側面板設定 ODF 每一維的來源後按 Save：
   - y1 → `CO2_315834011`
   - y2 → `Humidity_315834011`
   - y3 → `Temperature_315834011`

## 2. 執行

需要先裝 [uv](https://docs.astral.sh/uv/)。相依套件寫在 `pyproject.toml`，`uv run` 會自動安裝。clone 後先 `git submodule update --init`。開兩個 terminal，都在本資料夾執行：

```bash
# Terminal 1: M2 (sensor side)
PYTHONPATH=. uv run python Dummy_Device_IoTtalk_v1_py/DAI.py SA_M2

# Terminal 2: Dummy_Device (display side)
PYTHONPATH=. uv run python Dummy_Device_IoTtalk_v1_py/DAI.py SA_Dummy
```

`PYTHONPATH` 讓 `DAI.py` 找得到本資料夾的 SA。兩邊都看到 `Register successfully` 後：

## 3. 在 GUI 綁定裝置

註冊成功後，裝置還沒和 GUI 上的方塊綁在一起，狀態會是 `SUSPEND`，不會收送資料。

1. 點 `M2_315834011` 方塊上的名稱 → 選 `xxxx.M2_315834011`（`xxxx` 由伺服器產生，terminal 會印出 `Device name: ...`）
2. 點 Join 右邊那個 `Dummy_Device` 方塊的名稱 → 選 `xxxx.Dummy_Device`
3. 綁好後名稱會變成藍色，terminal 出現 `Device state: RESUME.`

## 4. 預期結果

Terminal 2（Dummy）每秒印出收到的 3 個數值，例如：

```
CO2: 418.013824, Humidity: 50.43335, Temperature: 34.151215
```

如果印出的是 list（例如 `[418.013824]`），代表收到的不是 3 個值，回 GUI 檢查 `Display_315834011` 是否 3 維，以及 Join 裡 y1～y3 的對應。

## 常見問題

- **一開始沒有資料**：`sensors.py` 要等農場送下一輪資料，實測約 20 秒以上才有第一筆。沒資料時 IDF 函式回傳 `None`，這一輪就不送
- **GUI 上有數值但都是 0～1 的小數**：那是 GUI 的模擬資料，代表方塊沒綁到真實裝置（名稱是黑字）。照第 3 步綁定
- **一直 `SUSPEND`**：還沒在 GUI 綁定，或綁錯方塊
- **`Attach failed ... No route to host`**：本機網路斷線，程式會自動重試、重新註冊，恢復後不需要重新綁定
- **`ServerURL` 不是 classgui**：`classgui.iottalk.tw` 是網頁，程式要連的是 `class.iottalk.tw`

## HW2

用 ChatGPT 完成和 HW3 相同的作業，結果對錯不計，把對話的分享連結（ChatGPT 右上角 Share → Copy link）交到 E3。
