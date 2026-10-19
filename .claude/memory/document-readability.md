---
name: document-readability
description: Readability rules for every document I write (notebooks, walkthroughs, reports): expand abbreviations, cut filler and meta notes, no source annotations, short figure text
metadata:
  type: feedback
---

Apply to all documents I produce for the user: homework notebooks, paper walkthroughs, reports.

- **越好懂越好，不是越多字越好**：不必要的資訊刪掉，它增加認知負擔。圖的解說也一樣要短。
- **縮寫附全稱**：第一次出現就寫全稱，例如 POMDP、MDP、ATP、RET。使用者看到縮寫會問。
- **不寫後設說明**：開頭不放「怎麼讀這份文件」、全文來源、crop 過、Codex 核對過之類的話。
- **不印出處標註**：內文不寫 (Sec. III, p.4)，表格不放出處欄。查證照做，只是不印在文件上。
- **格式扁平、說人話**：一個 `##` 標題底下用「引導句 + 平列點」，不疊 `###` 小標，不用表格塞代碼。用一般人聽得懂的白話解釋，像講給不懂這領域的人聽，例如「先拿 4 分贏一局」而不是只列術語。
- **領域知識從零講起**：介紹一個領域（例如網球）先講它是什麼、怎麼玩、怎麼對應到資料，再講細部規則。不要假設使用者已經懂基本玩法。
- **README 簡短好讀**：README 只放用途、前置需求、怎麼跑、網址和必要的差異說明。教學放 `docs/`，狀態、決策、驗證紀錄、待辦寫進該資料夾的 `CLAUDE.md`，不寫進 README。
- **數學不省略**：用到基礎數學（三角函數、cosine similarity、線性代數性質）先 review 基礎再講公式。

**Why:** user pasted these rules (2026-10-16) and asked to rewrite the HW2 notebook by them, so they apply beyond paper walkthroughs. Same day the rewrite (### subsections + table + terse term lists) got 「格式不夠扁平然後看不懂，說人話」, then 「我是說看不懂網球幹嘛」 — the real gap was missing basics (what tennis is), not only wording.

2026-10-19 HW3: I had turned README into a tutorial plus status log (300 lines, progress notes, verification tables); user said 「readme 不是讓你這樣用的，readme 應該是簡短好讀，你要記錄狀態專門建立一個 claude.md」.

**How to apply:** check every markdown cell / section against this list before handing it over. Paper-specific delivery rules: [[paper-walkthrough-format]]. Sentence-level style: [[writing-style]].
