# K3s 叢集運作說明

這份文件說明 `k3s-up.sh`、`k3s-down.sh` 每一步在做什麼，以及叢集裡各元件怎麼連線。整體架構見 [architecture.pdf](architecture.pdf)。

## Incus 的三個概念

Incus 是 Linux 上管理容器和虛擬機（VM，Virtual Machine）的工具。本作業只用它開 VM，底層是 Linux 內建的 KVM（Kernel-based Virtual Machine）。

- **instance** 一台容器或 VM，加上 `--vm` 就是 KVM 虛擬機。本作業有 `hw3-server`、`hw3-worker` 兩台
- **profile** 可共用的設定範本，包含 CPU、記憶體、網卡、磁碟。兩台 VM 套用同一份 `hw3-k3s`
- **network** Incus 管理的虛擬交換器（bridge），內建 DHCP（自動分配 IP）、DNS（名稱解析）和 NAT（讓 VM 透過主機上網）。本作業的是 `nycu-cc-hw3-br`，網段 `192.168.56.0/24`

## k3s-up.sh 的流程

腳本依序執行下面 13 步。每一步都會先檢查是否已經完成，所以可以重複執行，中途失敗時直接再跑一次即可。

1. **檢查前置需求** 主機不是 Linux 就直接停下。其他項目一次檢查完，把缺少的東西連同修正指令一起列出再停止，避免改一項、跑一次、才發現下一項。
2. **build image** 在主機用 Docker build 出 `hw3/web`、`hw3/api`、`hw3/db`。這步放在最前面當早退機制，程式碼或 Dockerfile 有錯就在這裡停下，還沒動到任何 VM。
3. **建立網路** `incus network create` 建出 `nycu-cc-hw3-br`。Incus 會在上面跑 dnsmasq 提供 DHCP，並設好 NAT。
4. **放行防火牆** 見下方「防火牆」一節。
5. **寫入 profile** 設定 2 vCPU、3 GiB 記憶體、20 GiB 磁碟，網卡接到 `nycu-cc-hw3-br`。每次都會重新套用，改了 `scripts/common.sh` 的規格再跑一次即可。
6. **建立 VM** `incus init images:ubuntu/24.04 --vm` 下載官方 Ubuntu 映像並建立 VM。接著用 `incus config device override` 指定 DHCP 保留位址，server 固定 `.10`，worker 固定 `.11`，因為 K3s 需要不會變的節點 IP。
7. **開機** `incus start`，已經開著就跳過。
8. **等 VM 就緒** 反覆用 `incus exec` 在 VM 裡執行 `ip addr`，直到拿到指定 IP，最多等 5 分鐘。Incus 的映像內建 incus-agent，所以不需要 SSH 金鑰就能下指令。
9. **安裝 K3s server** 在 `hw3-server` 執行官方安裝腳本，版本固定為 `v1.36.5+k3s1`。參數見下一節。
10. **worker 加入叢集** 從 server 讀出 K3s 自動產生的 join token，在 `hw3-worker` 安裝 k3s agent，連到 `https://192.168.56.10:6443`。
11. **等節點就緒** 先確認兩台的 K3s 服務都已啟動，再等 `kubectl get nodes` 兩台都是 Ready。只看 Ready 不夠，剛開機時 API 可能還回報關機前的舊狀態。
12. **匯入 image** 用 `docker save | incus exec hw3-worker -- k3s ctr images import -` 直接把 image 送進 worker，不經過 image registry。worker 已有同一版（image ID 相同）就跳過。server 不跑 App，所以不用匯入。
13. **部署 App** 先產生資料庫帳密，存成 Secret `db-credentials`，再執行 `kubectl apply -k k8s/`。上一步有匯入新 image 的話，會重啟對應的 Pod，因為 image 名稱沒變，Pod 不會自己換新版。最後從主機打一次 `/api/health`，請求經過 NodePort、nginx、api、PostgreSQL 四段，回傳 `"database":"ok"` 才算完成。

## K3s server 的安裝參數

- **`--cluster-init`** 叢集狀態存在內建的 etcd，不用預設的 SQLite。etcd 可以多台同步，以後加 server 就能擴成 HA（High Availability，高可用）
- **`--node-ip`、`--flannel-iface`** 指定用 192.168.56.x 這張網卡。節點之間的 Pod 流量由 flannel 用 VXLAN（Virtual Extensible LAN）包裝後，走 `nycu-cc-hw3-br`
- **`--node-taint ...:NoSchedule`** 給 server 貼上 taint，App 的 Pod 不會排到這台，說明見下一節
- **`--disable traefik --disable servicelb`** App 用 NodePort 對外，用不到內建的 Ingress 和 LoadBalancer，關掉省資源
- **不指定 token** K3s 自己產生隨機 token，不會寫死在腳本或 git 裡

## K3s 和 VM 的關係

Incus 只負責 VM 開機、關機。K3s 是裝在 VM 裡的 systemd 服務，開機會自動啟動，所以關機再開後叢集會自己恢復。

- **Pod** 實際執行的容器，可能被刪除、重建，或換到另一台節點
- **Service** 固定的名稱加上轉送規則，規則寫在每一台節點上，Pod 在哪台都連得到
- **taint** 貼在節點上的「禁止進入」標記，Pod 要有對應的 toleration（通行證）才能排上去。K3s 自己的 CoreDNS、local-path-provisioner、metrics-server 有通行證，所以仍在 server 上
- **server 停掉時** worker 上已經在跑的 Pod 不受影響，網頁照常可用。失去的是管理能力：`kubectl` 不能用，Pod 壞了也沒人重建

## 三層怎麼連起來

瀏覽器連到 NodePort 30080，Service `web` 把請求轉給 web Pod。nginx 直接回傳 Vue 頁面，`/api/` 開頭的請求轉給 `http://api:8000`。api Pod 用環境變數 `PGHOST=db` 連到 Service `db`，帳密來自 Secret。

- **名稱解析** Service 名稱由 CoreDNS 解析，例如在 api Pod 查 `db` 會得到 `db.hw3.svc.cluster.local`
- **健康檢查** api 的 readiness probe 打 `/api/health`，這個端點會執行 `SELECT 1`，資料庫連不上時 api 就收不到流量。liveness probe 只檢查 port，資料庫暫時斷線不會害 api 被重啟。db 用 `pg_isready` 檢查
- **持久資料** db-0 把 PVC（PersistentVolumeClaim，向叢集申請的儲存空間）`data-db-0` 掛到資料目錄。local-path 把它放在 worker 磁碟的 `/var/lib/rancher/k3s/storage/` 底下
- **建資料表** db 的 image 把 `schema.sql` 放進 `/docker-entrypoint-initdb.d/`。只有資料目錄是空的時候才會執行，已有資料就不會重建

## IP 怎麼分配

叢集裡有三種 IP，由不同機制決定。程式裡一律寫 Service 名稱，從不寫 IP。

- **節點 IP** `192.168.56.10`、`.11`，由 Incus 的 DHCP 依保留位址發放，固定不變
- **Pod IP** 在 `10.42.0.0/16` 裡，每台節點分到一段：server 是 `10.42.0.x`，worker 是 `10.42.1.x`。Pod 每次重建都會拿到新 IP
- **Service IP** 在 `10.43.0.0/16` 裡，建立 Service 時分配，Service 存在期間固定。例如 CoreDNS 是 `10.43.0.10`

Pod 和 Service 的網段只存在叢集內部，不會跟主機網路衝突。唯一要注意的是節點網段 `192.168.56.0/24`：VirtualBox 的 host-only 網路預設也用這一段，主機上已經有的話會衝突。

## k3s-down.sh 的流程

預設依序做三件事，對應題目「停掉 App 和 K3s，但保留持久資料」的要求。

1. **停 App** web、api、db 都縮到 0 個 replica，等 Pod 全部結束。PostgreSQL 收到停止訊號後，會把資料寫回 PVC 再退出。
2. **停 K3s** 先停 worker 的 `k3s-agent`，再停 server 的 `k3s`。
3. **關 VM** 先關 worker 再關 server，送出正常關機訊號。超過 2 分鐘還沒關才強制斷電。

磁碟都會保留。下次執行 `./k3s-up.sh` 時，`kubectl apply` 會把 replica 設回 1，App 和資料都會回來。

加上 `--delete` 時，做完上面三步後，再刪掉兩台 VM 和磁碟（資料庫也一起刪除）、profile、網路，以及 up 時加的防火牆規則。

## 防火牆

主機有開 UFW（Uncomplicated Firewall）或裝了 Docker 時，VM 的網路會被擋住。UFW 預設會丟掉 VM 送給主機的 DHCP 請求，VM 拿不到 IP。Docker 會把封包轉送預設改成拒絕，VM 拿到 IP 也上不了網。

- **`k3s-up.sh`** 用 sudo 加上 UFW 規則 `allow in on nycu-cc-hw3-br` 與 `route allow in/out on nycu-cc-hw3-br`，並在 Docker 的 `DOCKER-USER` 規則鏈插入兩條放行規則。Docker 的規則重開機會消失，所以每次 up 都會補上
- **`k3s-down.sh`** 不動防火牆
- **`k3s-down.sh --delete`** 移除上面加的所有規則
- **沒有 UFW 也沒有 Docker** 整段跳過，不會要求 sudo

## 與 production 的差異

以下是為了符合題目或單機環境，和正式環境不同的地方。

- **control-plane 只有一台** production 會用三台以上 server 讓 etcd 互相備援。這裡用 etcd 是為了日後能直接加 server
- **每層只有一個 replica** Pod 重建的幾秒內服務會中斷。production 會開多個 replica，分散到不同節點
- **資料庫只有一份** local-path 的資料只存在 worker 的磁碟，沒有複寫也沒有備份。production 會用 operator（例如 CloudNativePG）做主從複寫，或直接用雲端資料庫
- **用 NodePort 對外** production 會用 Ingress 或 LoadBalancer，加上 HTTPS
- **兩台 VM 在同一台實體機** 主機壞了兩台一起停。production 的節點會分散在不同機器或機房
- **防火牆整個放行 bridge** production 只開需要的 port
- **image 不經過 registry** production 由 CI（Continuous Integration）build 後推到 image registry，節點自己拉取。這裡直接匯入，助教不需要任何外部服務就能重現
- **Secret 只有 base64 編碼** 沒有加密。production 會開啟 etcd 加密，或用 Vault 之類的工具管理
- **資料庫沒有備份** K3s 每 12 小時會自動備份 etcd，但那只有叢集設定。production 會定期備份資料庫

## 已照 production 做法的部分

- **control-plane 不跑 App** server 加了 taint
- **叢集狀態用 etcd** 不用 SQLite，K3s 每 12 小時自動備份一次
- **資料庫用 StatefulSet** Pod 名稱固定為 `db-0`，並綁定專屬的 PVC
- **機密隨機產生** join token 和資料庫密碼都是部署時產生，不進 git
- **健康檢查與資源上限** 每個 Pod 都有 readiness、liveness probe 和 requests、limits
- **依序關機** 先停 App，再停 K3s，最後關 VM，讓 PostgreSQL 正常結束
- **以實際請求判斷部署完成** 不只看 Pod 狀態
- **版本固定** K3s 版本寫死在腳本裡，每次安裝結果一致

## 常見問題

- **`cannot talk to the Incus daemon`** 使用者還沒加入 `incus-admin`，或加入後還沒重新登入
- **網路名稱為什麼這麼短** Incus 的網路名稱同時是主機上的網卡名稱，Linux 限制最多 15 個字元
- **VM 開機了但拿不到 IP** 多半是防火牆擋住，確認 up 的防火牆步驟沒有被略過
- **VM 開不起來** `incus console hw3-server` 可以看開機畫面，按 Ctrl+a 再按 q 離開
