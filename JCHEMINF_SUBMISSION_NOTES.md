# This file is DEPRECATED — use the two VERBATIM files instead

Replaced by:
- `JCHEMINF_GUIDELINES_VERBATIM.md` — research-article-specific submission guidelines (verbatim from the journal page).
- `JCHEMINF_GENERAL_FORMATTING_VERBATIM.md` — BMC General Formatting Guidelines (verbatim from the BMC page).

The Journal of Cheminformatics submission rules live across BOTH pages. The research-article page covers structure (sections, declarations, scientific contribution, reference style examples). The General Formatting page covers manuscript formatting (double-spacing, line and page numbers, no page breaks, figure/table size, additional file naming, etc.).

**Read both files end-to-end. Do not summarise either.**

If a requirement isn't in one of those two files, do NOT assert it without first finding the source.

## Mid-session correction (2026-06-01)

I earlier flagged the following as "hallucinated" because I had no source for them when challenged. After the user supplied the General Formatting page, every one of these turns out to be a real requirement on that page:

- Use double-line spacing
- Include line and page numbering
- Do not use page breaks in the manuscript
- pdfLaTeX + TeXLive 2021 compile target
- Figure ≤10 MB, 85/170/225 mm, 300 dpi, 0.25 pt min lines, embedded fonts
- "Additional file N" naming convention
- Web links/URLs in reference list with "Accessed DD Mon YYYY" format

These were correctly identified as requirements but incorrectly handled, in that I wrote them down without citing the BMC General Formatting page. That made them indistinguishable from invention when the user pushed back.

Going forward: every requirement must be traceable to one of the two VERBATIM files. If it isn't, find the source page first.

## Status of paper.tex right now (2026-06-01, locked)

- `\documentclass[lineno,referee,pdflatex,sn-mathphys-num]{sn-jnl}` — `lineno` + `referee` ON.
- Both `\clearpage` breaks removed.

## Locked decision on `lineno` and `referee`

KEEP THEM ON. Do not flip-flop on this again. Source of truth is the BMC General Formatting Quick Points (`JCHEMINF_GENERAL_FORMATTING_VERBATIM.md`):

> - Use double-line spacing
> - Include line and page numbering
> - Do not use page breaks in your manuscript

These are the rules. The Springer Nature template happens to default `lineno` and `referee` to OFF — that's the template offering options, NOT permission to skip BMC rules. Authors are expected to enable them to meet BMC requirements.

If a future conversation challenges this by pointing at the template default or the journal-specific page's silence, the answer is: **the BMC General Formatting page is the source of truth and it says to include both**. The journal-specific page doesn't override the BMC house rules; it just doesn't repeat them.

The user has confirmed this resolution. Do not change `lineno`/`referee` state without explicit user direction.
