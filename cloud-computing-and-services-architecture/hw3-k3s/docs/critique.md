# 對題目技術選型的批評：Vagrant

題目把 Vagrant 列為第一個選項。本作業改走 Option 2，因為 Vagrant 已經不適合出現在 2026 年的雲端課程裡。

- **不再是開源軟體** 2023 年 8 月，HashiCorp 把 Vagrant 從 Mozilla Public License 2.0 改成 Business Source License 1.1，原始碼看得到，但已經不算開源。2025 年 2 月 HashiCorp 又被 IBM 收購，工具的授權和走向全由單一廠商決定。
- **預設的 VirtualBox 跟 KVM 打架** Linux 6.12 起，KVM（Kernel-based Virtual Machine）模組一載入就佔住 CPU 的虛擬化功能，VirtualBox 的 VM 會開不起來（`VERR_VMX_IN_VMX_ROOT_MODE`）。要改 kernel 參數 `kvm.enable_virt_at_load=0` 才能用，同一台機器只能二選一。
- **換成 KVM 要靠社群 plugin** Vagrant 本身不支援 KVM，要另裝社群維護的 vagrant-libvirt。安裝前得先用 sudo 裝 libvirt 開發套件，再編譯 Ruby 原生擴充。它的共用資料夾預設走 NFS（Network File System），主機又要 root 權限。
- **box 生態在萎縮** Canonical 官方的 `ubuntu/jammy64` 最後更新是 2024 年 10 月，沒有推出 24.04 版。曾經最常用的多平台 box `generic/ubuntu2204` 停在 2024 年 1 月。Chef 維護的 `bento/ubuntu-24.04` 不支援 libvirt。
- **Apple Silicon 支援起步晚** VirtualBox 到 2024 年 9 月的 7.1 版才支援 Apple Silicon 主機。用 M 系列 Mac 的同學走 Option 1，等於拿最晚才支援的組合交作業。
- **多一層卻沒多做事** Vagrant 的 provision 最後還是在跑 shell script。`vagrant halt` 之後能恢復，靠的是 VM 裡 K3s 是 systemd 服務，跟 Vagrant 無關。Option 2 用兩支 shell script 就能做到同樣的事，本作業就是這樣做。
- **唯一的優點也守不住** Vagrant 的賣點是同一份 Vagrantfile 在 Windows、macOS、Linux 都能跑。上面的 KVM 衝突和 Apple Silicon 問題，正好打破這個賣點。

更好的選擇有兩種。要教 VM 層級的叢集，Linux 上用 Incus 或 libvirt 加 cloud-init，直接用 KVM，指令也比 Vagrant 少。只想練 Kubernetes 資源的話，k3d（在 Docker 裡跑 K3s）或 kind（Kubernetes in Docker）幾秒就能開好多節點叢集，完全不用 VM。
