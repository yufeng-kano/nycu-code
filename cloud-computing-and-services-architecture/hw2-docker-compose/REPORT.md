# Homework #2 報告：以 Docker Compose 建置三層式網頁應用

| 項目 | 內容 |
| --- | --- |
| 課程 | Cloud Computing & Services Architecture |
| 姓名 | 黃毓峰 |
| 學號 | 315834011 |
| 日期 | 2026 年 9 月 29 日 |

## 一、系統架構

題目要求資料庫、後端、前端三個容器。本專案多加一個反向代理，共四個容器，全部由 `docker-compose.yml` 管理。

| 容器名稱 | 技術 | 職責 |
| --- | --- | --- |
| hw2-proxy | Nginx | 唯一對外入口，監聽 8080 埠。`/api` 轉給後端，其餘路徑轉給前端 |
| hw2-frontend | Nginx 提供 Vite + Vue 3 + TypeScript 的靜態建置結果 | 使用者操作介面 |
| hw2-backend | Flask 3，以 gunicorn 執行，套件由 uv 管理 | JSON API，讀寫 SQLite 檔案 |
| hw2-db | Alpine + sqlite3 | 啟動時把 `schema.sql` 套用到共用 volume |

請求流程如下。瀏覽器只會連到 hw2-proxy。使用者開啟頁面時，proxy 把 `/` 轉給 hw2-frontend 取回 HTML 與 JS。頁面上的操作全部打 `/api/...`，proxy 依路徑轉給 hw2-backend。後端透過 Python 內建的 `sqlite3` 模組開啟 `/data/names.db`，這個檔案放在名為 `hw2-sqlite-data` 的 named volume 上，hw2-db 與 hw2-backend 同時掛載。

容器之間靠 Compose 預設建立的 bridge network 互通，服務名稱就是 DNS 名稱。proxy 的設定檔直接寫 `frontend:80` 與 `backend:8000`，不需要知道 IP。只有 proxy 對外開埠，前端與後端從主機無法直接存取。

啟動順序由 healthcheck 控制。hw2-db 的健康檢查是能成功查詢 `names` 資料表，代表 schema 已建好。hw2-backend 設定 `depends_on: db: condition: service_healthy`，健康檢查是 `/api/health` 回 200。hw2-proxy 要等前端與後端都健康才啟動，避免使用者在後端還沒起來時看到 502。

## 二、各層的實作

### 資料庫層

`db/schema.sql` 只有一張 `names` 資料表，`id` 為自增主鍵，`name` 設為 `NOT NULL UNIQUE`。容器的 entrypoint 執行 `sqlite3 /data/names.db < schema.sql` 之後以 `tail -f /dev/null` 常駐。這個常駐沒有實際功能，原因在第四節說明。

### 後端 API

後端是一個 Flask blueprint，掛在 `/api` 前綴下。三個題目要求的端點如下，另加一個 `/api/health` 供 healthcheck 使用。

| Method | 路徑 | 輸入 | 成功 | 失敗 |
| --- | --- | --- | --- | --- |
| POST | `/api/add_name` | JSON `{"name": "..."}` | 201，回傳 id 與 name | 400 名字空白，409 名字重複 |
| GET | `/api/get_names` | 無 | 200，回傳 `{"names": [...]}` | 無 |
| DELETE | `/api/remove_name` | query `?name=...` 或 JSON body | 200 | 400 名字空白，404 找不到 |

所有函式都有 type hint，`mypy --strict` 與 `ruff` 皆通過。每個 request 各開一條 SQLite 連線，request 結束時由 teardown 關閉。連線開啟時設定 `PRAGMA journal_mode=WAL`，讓 gunicorn 的兩個 worker 在一個寫入時另一個仍能讀取。

正式執行用 gunicorn 而非 `flask run`。Flask 內建的伺服器是開發用途，單執行緒且不處理逾時，不適合放進容器長期跑。

`backend/tests/test_api.py` 用 pytest 對四個端點做整合測試，使用暫存 SQLite 檔案，不需要 Docker 就能跑。

### 前端

前端用 Vite 建置 Vue 3 加 TypeScript 的單頁應用，`vue-tsc` 在建置前做型別檢查。頁面提供一個輸入框、新增按鈕、名字清單與每筆的刪除按鈕，所有 API 呼叫集中在 `src/api.ts`。

建置採 multi-stage Dockerfile。第一階段用 Node 22 執行 `npm ci` 與 `npm run build`，第二階段只把 `dist/` 複製進 Nginx 映像，最終映像不含 Node。

前端的 Nginx 只負責靜態檔，`try_files` 把找不到的路徑導回 `index.html`。它不處理 `/api`，這是 proxy 的工作。

### 反向代理

`proxy/nginx.conf` 定義兩個 upstream，`location /api/` 轉後端，`location /` 轉前端。這樣瀏覽器看到的所有請求都是同一個 origin，後端不需要設定 CORS。若把兩個 Nginx 合併成一個，靜態檔與代理規則會綁在同一個設定檔，前端改版就得重建代理層，分開之後兩者可以獨立重建與替換。

## 三、執行方式

```sh
docker compose up --build
```

啟動完成後開啟 http://localhost:8080。`docker compose down -v` 會一併刪除資料 volume。

## 四、遇到的困難

**SQLite 沒有東西可以常駐**。SQLite 是嵌入在程式裡的函式庫，不是伺服器，沒有 daemon 也沒有網路埠。題目要求一個「SQLite 資料庫容器」，但真正讀寫資料庫的一直是後端程序。最後的做法是讓 db 容器負責建 schema，再用 `tail -f /dev/null` 保持存活，並以「能查到資料表」當健康檢查。這個容器實際上只是 schema 的擁有者。

**多程序同時開同一個檔案**。gunicorn 開兩個 worker，加上 db 容器的健康檢查也會開啟同一個檔案。預設的 rollback journal 在寫入時會鎖住整個檔案，讀取端只能等待，逾時就得到 `database is locked`。改用 WAL 模式後讀寫可以並行，但 WAL 要求所有程序在同一台主機且共用同一個檔案系統，換成網路檔案系統就會壞掉。這個限制是第五節主張改用 PostgreSQL 的主要理由之一。

**DELETE 的參數放哪裡**。題目只說 `remove_name` 用 DELETE，沒說名字放在哪。HTTP 規範沒有禁止 DELETE 帶 body，但很多 proxy 與 client 會直接丟掉。後端因此同時接受 query string 與 JSON body，前端固定用 query string。

**啟動順序**。`depends_on` 預設只保證容器啟動順序，不保證服務可用。沒有 healthcheck 時 proxy 可能比後端先就緒，這段時間打 `/api` 會得到 502。每個服務都加上 healthcheck，再用 `condition: service_healthy` 串起依賴，才能保證 proxy 開埠時後端已可回應。

**型別檢查**。`mypy --strict` 對 Flask 的 `g` 物件無法推斷型別，需要在取出連線時明確標註 `sqlite3.Connection`。前端則需要一個 `shims-vue.d.ts` 讓 TypeScript 認得 `.vue` 模組。

## 五、對題目技術選型的批評

以下三點是題目設計上的錯誤。實作仍依題目要求完成，但這些選擇不該出現在一門叫做 Cloud Computing & Services Architecture 的課的作業裡。

### SQLite 不是資料庫層，把它當資料庫層教是錯的

三層式架構的定義是每一層透過網路協定溝通，可以獨立部署、獨立擴展、獨立替換。SQLite 一項都做不到。它是連結進程序裡的函式庫，沒有 daemon、沒有埠、沒有協定。題目要求「一個 SQLite 資料庫容器」，這個容器在本專案裡只能執行一次 `schema.sql` 然後 `tail -f /dev/null`。它不服務任何請求，因為根本沒有請求可以送給它。題目要教容器化的多層架構，卻選了一個無法被容器化成獨立層的元件，這是題目自己和自己矛盾。

具體的缺陷如下。

**沒有並行寫入**。SQLite 任何時刻只允許一個寫入者，鎖的粒度是整個資料庫檔案。WAL 模式只解決讀寫互不阻塞，寫入者之間仍然序列化。兩個 gunicorn worker 同時 `INSERT`，其中一個必須等待。學生在這種環境下學不到交易隔離層級、連線池、鎖競爭，因為 SQLite 只有一種隔離層級，也沒有連線可以池化。

**無法水平擴展**。後端要開第二個容器，就必須掛同一個 volume。跨主機時 WAL 依賴的共享記憶體索引失效，資料直接損毀。一個後端只能有一個副本的架構，不叫分散式系統。

**型別約束形同虛設**。SQLite 的欄位型別是 type affinity，不是約束。欄位宣告 `INTEGER` 照樣可以塞字串，宣告 `TEXT` 照樣可以塞 blob。`VARCHAR(10)` 的長度限制會被無視。學生在這上面養成的習慣，到任何一個真正的關聯式資料庫都會出錯。

**沒有權限系統**。SQLite 沒有使用者、沒有角色、沒有 GRANT。誰拿得到檔案誰就能 `DROP TABLE`。安全邊界只能靠檔案系統權限，這在容器裡等於沒有。

**沒有網路協定**。這一點決定了前面所有問題。沒有協定就沒有 client-server，沒有 client-server 就沒有連線管理、沒有認證、沒有 TLS、沒有讀寫分離、沒有備援。這門課該教的東西，SQLite 一項都碰不到。

PostgreSQL 才是這個作業該用的東西。它本身是網路服務，官方映像 `postgres:16` 拉下來就能跑，healthcheck 用 `pg_isready` 一行解決，後端用 `postgresql://user:pass@db:5432/names` 連線。它有 MVCC，讀寫互不阻塞，寫入者之間只在同一列上衝突。它有四種隔離層級可以選，學生可以實際觀察 read committed 和 serializable 的行為差異。它有角色與 GRANT，可以做到欄位層級的授權，還有 row-level security 讓資料庫在應用層有漏洞時仍能擋住越權查詢。它有 `pgvector`、`PostGIS` 這類延伸套件，同一個資料庫可以直接做向量檢索與地理查詢。這些全部是 SQLite 結構上不可能提供的能力，不是調校或版本更新可以補上的差距。

用 SQLite 教三層式架構，學生學到的只是把三個資料夾各放一個 Dockerfile。把 SQLite 換成 PostgreSQL，學生才會第一次遇到連線字串、認證、啟動順序依賴、資料 volume 的生命週期，這些才是這門課的內容。

### Flask 是錯誤的框架選擇

Flask 把所有型別資訊丟掉。`request.get_json()` 回傳 `Any`，`request.args.get()` 回傳 `str | None`，框架不驗證任何輸入。本專案的 `_name_from_request()` 要手動檢查 `name` 是否存在、是否為字串、是否空白，再手動回 400。這段驗證碼比業務邏輯本身還長。三個端點的業務邏輯合計三行 SQL，驗證與錯誤處理卻佔了大半個檔案。

Flask 的 `g` 物件是一個無型別的全域容器。要通過 `mypy --strict`，取出連線時必須手動標註型別，框架本身不提供任何協助。這種設計在 2010 年可以接受，2026 年還拿來當教材，等於教學生忽略型別系統。

Flask 不產生 API 文件。三個端點的請求格式、回應格式、錯誤碼，全部要另外用手寫。手寫的文件必然與程式碼脫節。

Flask 不是伺服器。它的內建開發伺服器是單執行緒、不處理逾時、官方文件明寫不可用於正式環境。要跑起來必須另外選 gunicorn 或 uwsgi，再研究 worker 數量與模式。題目只說「一個 Flask 應用」，沒提這件事，暗示學生直接用 `flask run` 交作業。

FastAPI 解決以上全部問題。用 Pydantic model 宣告 `name: str`，驗證、錯誤回應、型別檢查由框架完成，程式碼只剩業務邏輯。OpenAPI 規格與 Swagger UI 自動產生，永遠與程式碼一致。原生 async，搭配 uvicorn 一個指令啟動。依賴注入系統讓資料庫連線的取得與釋放有型別、可測試。這個作業用 FastAPI 寫，後端程式碼會少一半，型別檢查會全部通過，還附贈一份文件。

### API 設計不是 RESTful，是把 RPC 硬套在 HTTP 上

題目的三個路徑是 `add_name`、`get_names`、`remove_name`。動詞寫在 URL 裡，這是 RPC 命名。題目同時又要求分別用 POST、GET、DELETE，等於同一個動作講了兩次，一次在 method 一次在路徑。既然要區分 method，路徑裡的動詞就是多餘的。既然路徑裡有動詞，method 就沒有意義。兩者只能選一個，題目兩個都要。

命名本身也不一致。`add_name` 與 `remove_name` 是單數，`get_names` 是複數。同一組 API 對同一種資源用兩種名字。

`remove_name` 用 DELETE 卻沒說識別資料放哪。DELETE 帶 body 的行為在 HTTP 規範裡是未定義的，很多 proxy 和 client 會直接丟掉 body。放 query string 又不符合題目「透過 DELETE 請求」的字面意思。這個模糊是設計缺陷，不是學生要自己猜的東西。

這種設計無法擴展。要加一個修改名字的功能，就得再發明一個 `update_name`。要查單筆，再發明一個 `get_name`。每個功能都是一個新動詞，沒有規律可循。工具鏈也不支援，OpenAPI 產生器、API gateway 的路由規則、HTTP 快取語意，全部假設 URL 表示資源而非動作。

RESTful 的正確寫法：

| Method | 路徑 | 意義 |
| --- | --- | --- |
| POST | `/api/names` | 新增一筆，回 201 |
| GET | `/api/names` | 列出全部 |
| GET | `/api/names/{name}` | 查單筆 |
| DELETE | `/api/names/{name}` | 刪除指定一筆，回 204 |

資源只有一個名字 `names`，動作由 method 表達，識別資料在路徑上。DELETE 不需要 body 也不需要 query string，前面提到的模糊直接消失。要加功能就是加 method 或加子路徑，不需要發明新動詞。這才是 HTTP 設計出來要被使用的方式，也是這門課該教的東西。

## 六、與題目要求的差異說明

| 題目要求 | 本專案 | 說明 |
| --- | --- | --- |
| 三個容器 | 四個容器 | 多一個反向代理，讓前端與 API 同 origin，並隔離內部服務 |
| 簡單 HTML/JavaScript 頁面 | Vite + Vue 3 + TypeScript | 仍由 Nginx 提供靜態檔，建置後產物就是 HTML 與 JS |
| Flask 應用 | Flask 3 以 gunicorn 執行 | 端點路徑與 method 完全依題目 |
| sqlite3 | sqlite3 | 未替換，第五節說明為何不該用 |
