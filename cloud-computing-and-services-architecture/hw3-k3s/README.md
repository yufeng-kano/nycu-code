# Homework 3：把三層式應用搬到 K3s

HW2 的 names 應用（nginx、Flask、PostgreSQL）跑在兩台虛擬機組成的 K3s 叢集上，一個指令啟動，一個指令關閉。架構圖見 [docs/architecture.pdf](docs/architecture.pdf)。

## 前置需求

- **Linux 主機** CPU 支援虛擬化（有 `/dev/kvm`），約 4 vCPU、6 GiB 記憶體空閒，可以連外網
- **Incus 6.0 以上** 用來開虛擬機，使用者要在 `incus-admin` 群組
- **Docker** 用來 build image，使用者要在 `docker` 群組

Ubuntu 24.04 的安裝方式如下，加入群組後要重新登入：

```sh
sudo apt install -y incus qemu-system-x86 docker.io
sudo usermod -aG incus-admin,docker "$USER"
sudo incus admin init --minimal
```

`./k3s-up.sh` 開始前會檢查以上各項，有缺少就列出修正指令並停止。

## 啟動與關閉

```sh
./k3s-up.sh              # 建立或啟動叢集，並部署 App
./k3s-down.sh            # 停 App、停 K3s、關機，資料保留
./k3s-down.sh --delete   # 連同虛擬機、網路和防火牆規則一起刪除
```

- **第一次啟動** 約 1 分鐘：build image、建立兩台虛擬機、安裝 K3s、匯入 image，最後執行 `kubectl apply -k k8s/`
- **防火牆** 主機有開 UFW 或裝了 Docker 時，up 會用 sudo 放行虛擬機網路，可能要輸入密碼
- **重複執行** 兩支腳本都能重跑，已完成的步驟會跳過

## 使用

網址是 <http://192.168.56.11:30080/>，可以新增、列出、刪除名字。`192.168.56.10:30080` 也連得到。

- **驗證** `./scripts/verify.sh` 跑題目的 5 項驗證，紀錄存在 `evidence/`
- **查看叢集** `./scripts/kubectl.sh -n hw3 get pods,svc,pvc`，主機不用另外裝 kubectl
- **進入節點** `./scripts/vm-shell.sh server` 或 `./scripts/vm-shell.sh worker`

## 叢集內容

- **hw3-server**（192.168.56.10）control-plane，加了 taint，不跑 App
- **hw3-worker**（192.168.56.11）執行下面三層
- **web** Deployment。nginx 提供 Vue 頁面，並把 `/api/` 轉給 api，對外是 NodePort 30080
- **api** Deployment。Flask 加 gunicorn，Service 是 `api:8000`
- **db** StatefulSet。PostgreSQL 17，Service 是 `db:5432`，資料放在 PVC（PersistentVolumeClaim）`data-db-0`

## 與題目要求的差異

- **用 Incus 不用 Vagrant** 走題目的 Option 2，理由見 [docs/critique.md](docs/critique.md)
- **db 用 StatefulSet** 題目寫 Deployment。資料庫需要固定的 Pod 名稱和專屬的 PVC，這正是 StatefulSet 的用途
- **web 合併前端與代理** HW2 的兩個 nginx 合成一個，對應題目的「Nginx 前端與反向代理」

## 其他文件

- [docs/guide.md](docs/guide.md)：腳本每一步在做什麼、IP 怎麼分配、防火牆、與 production 的差異
- [comparison.md](comparison.md)：Docker Compose 與 K3s 比較
- `docs/build.sh`：從 `architecture.svg` 和 `architecture.tex` 重新產生架構圖
