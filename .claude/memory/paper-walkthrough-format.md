---
name: paper-walkthrough-format
description: Paper walkthrough (論文導讀) delivery format, folder/filename rules, claude/index.md management; read before starting a walkthrough or adding project docs
metadata:
  type: feedback
---

User-provided rules for the Paper project (pasted 2026-10-16).

Delivery
- 導讀全部寫在 .md 交付，對話裡不再輸出全文，直接 show .md。.md 不嵌 base64 圖片。
- 2026-09-30 起導讀要有圖：放在該論文資料夾 `./res/<figure_name>`，.md 用 `./res/` 相對路徑引用。
- 開新論文前先確認要單檔完整教學還是逐節（2026-09-21 Transformer 只給 roadmap，使用者很不滿）。
- 2026-10-01 起完整導讀檔名就是論文簡稱，例如 `pi0/pi0.md`，不要 `pi0-full.md`。
- 教學類文件（教科書、overview）導讀不用裝 Codex，也不做獨立 review。

Folders and index
- 論文資料夾用論文簡稱命名（pi0、transformer、vit、emmi），不加 -walkthrough。
- project 文件由 `claude/index.md` 管理（路徑 + 摘要表格，新對話先讀）。每個 .md 放在資料夾裡，`claude/` 根目錄只留 index.md。新增、搬移、刪除文件時同一輪更新 index.md。
- 檔名不准有 `00-` 這類數字前綴，順序寫在 index.md。
- session 連到本機時，本機論文資料夾建立同樣結構：根目錄 index.md + 每篇一個資料夾（index.md、導讀 .md、res/）。2026-09-30 用的本機資料夾是 `~/Documents/論文`。

Content
- 論文縮寫（例如 EMMI）附全名，每篇都一樣。
- 使用者數學很差但要學：數學不能省略。講到 sin、cos、cosine similarity、線性代數性質時先 review 基礎再講公式。
- General readability rules (full names for abbreviations, cut filler, no source annotations, figure captions short) live in [[document-readability]].

**Why:** user's standing rules from the Paper project, pasted for this memory store.

**How to apply:** follow when writing any paper walkthrough; read `claude/index.md` first in that project.
