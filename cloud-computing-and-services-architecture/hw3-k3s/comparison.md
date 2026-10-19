# Docker Compose 與 K3s 比較

單機開發和 demo 用 Compose 就夠。需要多節點、自我修復或水平擴展時，K3s 的額外複雜度才划算。

以下是這次搬遷實際遇到的差異。

- **範圍** Compose 的容器都在同一台主機。K3s 由排程器把 Pod 分到節點，本作業用 taint 讓 App 只跑在 worker。
- **服務發現** Compose 用容器名稱當 DNS（Domain Name System）名稱。K3s 用 Service，Pod 重建後 IP 會變，Service 的名稱和 IP 不變。
- **啟動順序** Compose 靠 `depends_on` 加 healthcheck 排順序。K3s 不保證順序，改用 readiness probe，api 連得到資料庫才接流量。
- **儲存** Compose 用 named volume。K3s 用 PVC（PersistentVolumeClaim）申請空間，db Pod 被刪除後，新的 Pod 掛回同一個 PVC。
- **自我修復** Compose 只依 restart policy 重啟容器。K3s 的控制器持續比對期望狀態，Pod 不見就重建。
- **機密** Compose 常把密碼寫在 `.env`。K3s 有 Secret，本作業在部署時隨機產生密碼，不進 git。
- **成本** Compose 一個指令就能啟動。K3s 要準備 VM（虛擬機）、網路、防火牆和 image 匯入，control-plane 本身也佔記憶體。
