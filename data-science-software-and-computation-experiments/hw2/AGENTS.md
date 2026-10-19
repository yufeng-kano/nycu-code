# 作業二：ATP 網球紀錄分析

使用 <https://github.com/JeffSackmann/tennis_atp> 的資料分析 ATP 選手的紀錄，全部分析使用 pandas。

固定版本：<https://github.com/Haz22537/JEFFSM_tennis_atp>

## 1. 使用 GitHub repo

- clone repo 的資料。
- 詳細閱讀 repo 中的說明，了解資料的意義。

## 2. 讀取 atp_players.csv

- 裡面有哪些欄位？
- 總共有多少球員在列表中？
- 全部球員中，有多少 dob 生日資料不完整？
- 依照 ioc 國籍來分類，有最多球員的國家有哪些？
- TPE 和 TWN 各有多少球員？
- TPE 的球員中，有包含哪些姓氏？

## 3. 讀取 atp_matches*202[1234].csv

（英文版題目寫 `atp_matches_202[1-4].csv`，只含 4 個主巡迴賽檔。）

- TPE 球員贏了多少場？輸了多少場？
- 有多少種比賽層級（tourney_level）？
- 有多少場的 winner 身高比 loser 高／相等／低？
  - 依照不同的 tourney_level 統計身高和勝負的關係。
- 先將年齡 `.round()`：
  - `value_counts` winner 的年紀。
  - 根據球員的年齡，統計勝率。
  - 列出勝場數最多的前十位球員。
  - 列出勝率最高的前十位球員。
  - 製作表格統計不同年紀球員之間的對戰數量：index 是 winner 年齡，column 是 loser 年齡，值是對戰場數。只統計年齡 19–28。
  - 製作表格：index 是球員年齡，column 是對手年齡，值是勝率。
  - 有哪些年齡層，對其他年齡層的勝率都在 49% 以上？（只考慮年齡 19–28）

## 4. 自訂問題

提出五個自己的問題，並用 pandas 回答。

## 繳交方式

- 作業連結：Homework 2
- 全部放在同一個 Jupyter Notebook。
- 檔名格式：`<StudentID>_HW2.ipynb`，例如 `314831001_HW2.ipynb`。
