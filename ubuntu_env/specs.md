# UBUNTU環境
1. IPアドレス
    - ローカル：`ip a`
    - グローバル：`curl ifconfig.me`
2. OS
    - `lsb_release -a`
3. CPU
    - `lscpu`
4. メモリ
    - `sudo dmidecode --type memory`
5. ストレージ
    - `sudo parted -l`
6. グラフィックボード(GPU)
    - `lspci  | grep VGA`
7. セキュリティソフト確認
    - `sudo /opt/f-secure/linuxsecurity/bin/lsctl status get subscription`
