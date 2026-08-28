# Journal of Cheminformatics submission — status tracker

**Single source of truth for what's done and what's pending.** History of completed edits lives in `JCHEMINF_FIXES_LOG.md`. Rules live in `JCHEMINF_GUIDELINES_VERBATIM.md` (research-article) and `JCHEMINF_GENERAL_FORMATTING_VERBATIM.md` (BMC general). Both must be read end-to-end before changes — never summarise them.

---

## Status of `paper.tex` documentclass (locked)

`\documentclass[lineno,referee,pdflatex,sn-basic]{sn-jnl}` — line 36 of `paper.tex`.

- `lineno` + `referee` ON. Required by BMC General Formatting Quick Points (double-line spacing, line + page numbering). Do not flip-flop on this.
- Reference style: `sn-basic` (Basic Springer Nature, author-year). Switched from `sn-mathphys-num` on 2026-06-01.
- No `\clearpage` / `\newpage` anywhere in body.

---

## Punch list — overall progress

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Reference style switch + `\cite` → `\citet` fixes | **DONE 2026-06-01** | 9 edits applied. See FIXES_LOG section B.11. |
| 2 | URLs in body → reference list | **DONE 2026-06-01** | 3 `\url{}` removed from body (L345, L585, L595); 2 new `@misc` entries (`noiseinject`, `lhasa`) added to `~/Downloads/sn_bibliography.bib`. See FIXES_LOG §B.12. |
| 3 | LLM disclosure | **REWRITTEN 2026-06-02** | Removed from Methods. Now a `\subsection*{Declaration of generative AI in the writing process}` at L624 inside Declarations, after Acknowledgements — placement matches observed JCheminf practice (5/5 examples in back matter, 0 in Methods). Wording follows the Palmacci / Shah template + Raymond code-disclosure pattern. See FIXES_LOG §B.20. |
| 4 | NoiseInject software block (8 BMC fields) | **DONE 2026-06-01 (DRAFT)** | 8-field `\begin{itemize}` list inserted in Availability section after the NoiseInject descriptive paragraph. Carries `% TODO` marker on the Archived version line — replace placeholder with Zenodo DOI before final submission. See FIXES_LOG §B.14. |
| 5 | Dataset wording in Availability section | **DONE 2026-06-02** | Body rewritten to match BMC formula. Decision 1 = A: hERG-Ki processed file ships inside NoiseInject repo (action item: user to commit the file before submission). Decision 2 = B: dedicated dataset `@misc` entries `qm9_dataset` (Figshare DOI 10.6084/m9.figshare.978904) and `chembl_dataset` (EBI persistent URL) added to bib and cited in body alongside the paper citations. ChEMBL release-specific DOI is a TODO inside the `chembl_dataset` entry. See FIXES_LOG §B.15 + §B.19. |
| 6 | Caption titles | **REVERTED 2026-06-02** | Misinterpreted user's request — she meant the captions themselves should carry the main idea, which the ORIGINAL captions already did via `\textbf{Takeaway:}` sentences. All 7 modified caption blocks restored from `paper.tex.before_jcheminf_fixes`. **Full caption-improvement list from user is open and untackled** — see FIXES_LOG §B.24. |
| 7 | Graphical abstract | TODO (optional, user-supplied image) | Per `JCHEMINF_GUIDELINES_VERBATIM.md` line 16: 920 × 300 px, ≤150 KB, jpeg/png/svg, white background. "Authors are encouraged to make judicious use of colour." Recommended but not required — leave unaddressed unless you want to ship one. |
| 8 | Commas in numbers — table cells | **DONE 2026-06-01 (clean audit)** | Swept every `\begin{tabular}` body in `paper.tex`. Only hit was `ICC(1,1)` which is intraclass-correlation notation (statistic type 1,1), not a numeric thousands separator. No actual violations. See FIXES_LOG §B.17. |

---

## DRAFT placeholders in `paper.tex` for user review

| Location | Content | Status |
|----------|---------|--------|
| Abstract, L129 | Scientific Contribution paragraph | **REWRITTEN 2026-06-02** — insight-led, positions against prior work in isolation, names artifact at end. User to review wording, not structure. |
| Declarations, L618 | Authors' contributions | **REWRITTEN-SHORT 2026-06-02** — 42 words, all three of T.H./S.W./G.M. recorded as supervisors (user correction: they are all supervisors, not just industry contacts). User to verify role attribution. |
| Declarations, L624 | Declaration of generative AI in the writing process | **REWRITTEN 2026-06-02** — JCheminf back-matter convention. User to add specific LLM tool names (Claude, ChatGPT, etc.) if desired. |
| Availability, NoiseInject 8-field block | Archived version DOI | PLACEHOLDER, user to mint a Zenodo DOI before submission |
| Availability, datasets paragraph | hERG-Ki processed file | Decision 1 = A (ship inside NoiseInject repo) — user to commit the file before submission |
| Availability, datasets paragraph | ChEMBL release-specific DOI | TODO inside `chembl_dataset` bib entry — user to replace with `10.6019/CHEMBL.database.NN` for the actual release used |
| Captions — full user list | spelling out NDS/ICC at first use, defining Rep A / Rep B in correlation tables, "lower NDS is better" reminder, supplementary figure labels S1/S2, abbreviations rendering, abbreviations colon formatting, dataset context for supplementary | **NOT YET DONE** — captured in FIXES_LOG §B.24 for the next pass. |

---

## Bibliography issues (user-reported 2026-06-01)

`.bib` edits are user's domain unless delegated. These are documented here so they can be addressed in one pass.

### B1. URLs in body to migrate into reference list — **DONE 2026-06-01**

Per BMC General Formatting Guidelines §References: web links go in the reference list, not the body, in the format `Title. URL. Accessed DD Mon YYYY.`

Three URLs were inline in `paper.tex`; all migrated:

- L345 (Methods, NoiseInject Framework): `\url{github.com/adpunt/noise\_inject}` → `\citep{noiseinject}`
- L585 (Declarations / Availability): `\url{https://github.com/adpunt/noise_inject}` → `\citep{noiseinject}`
- L595 (Declarations / Funding): `\url{https://www.lhasalimited.org/}` → `\citep{lhasa}`

Two new `@misc` entries added at the end of `~/Downloads/sn_bibliography.bib`:

- `noiseinject` — Punt, A. (2026), software repo, MIT licence
- `lhasa` — `{Lhasa Limited}` (2026), organisation site

User to re-upload `sn_bibliography.bib` to Overleaf. The author field on `noiseinject` is set to `Punt, Adelaide` — change if a different author form is preferred.

### B2. Capitalisation lost in title fields — **DONE 2026-06-01**

Under `sn-basic` (author-year), BibTeX lowercases unprotected words in `title = {...}`. Acronyms must be brace-protected to survive.

User-reported acronyms — all addressed:

- `QSAR` — 4 unprotected titles + 1 journal field wrapped (other instances already had `{QSAR}`)
- `IC50` — wrapped in the one title where it appeared (also `{Ki}` brace-protected in same title)
- `ChEMBL` — wrapped in the second occurrence (first was already protected)
- `DFT` — wrapped (also `{ADME}` and `{GPU}` brace-protected in adjacent titles I touched)
- `Gaussian` — global wrap `Gaussian` → `{G}aussian` (canonical proper-noun recipe; preserves rendering as "Gaussian process", "Gaussian Processes", etc. wherever each phrase appears)
- `GPyTorch` — wrapped
- `Bayesian` — global wrap `Bayesian` → `{B}ayesian`

Nested protection (`{{Bayesian Optimization}}` style) appears in two pre-protected entries — harmless under BibTeX; the outer braces are LaTeX grouping, the inner is BibTeX case-protection.

See FIXES_LOG §B.18.

### B3. Dablander2024 `@phdthesis` → DPhil — **DONE 2026-06-01** (year mismatch flagged)

Applied fix option 1: stayed with `@phdthesis` and added `type = {DPhil thesis}`. Renders as "DPhil thesis" instead of the default "PhD thesis".

**Still flagged for user:** the citekey is `Dablander2024` but the entry's `year` field is `2023`. Decision needed — rename key to `Dablander2023` (and update every `\cite{Dablander2024}` in `paper.tex`) or change `year` to `2024` if 2024 is correct. Did not touch this; user to confirm.

See FIXES_LOG §B.18.

### B4. "e.a." artifact in author lists — **DONE 2026-06-01**

Root cause confirmed: 30 `.bib` entries had a literal `et al.` typed inside their author field, e.g. `author = {Cherkasov, Artem and Muratov, Eugene N. and Fourches, Denis et al.}`. BibTeX parses `et al.` as if it were a real author's surname and initial ("et" + "al."), then `sn-basic.bst` collapses it into its truncation glyph `.e.a.` next to the previous initial.

Fix: replaced ` et al.}` with ` and others}` across all 30 affected entries. `and others` is BibTeX's canonical truncation directive — the style sees it and renders the proper `et al.` glyph at the end of the author list with correct spacing.

See FIXES_LOG §B.18.

### B5. Lakshminarayanan2017 missing journal/volume/pages — **DONE 2026-06-01**

Converted from `@misc` (arXiv-only) to `@inproceedings` with NeurIPS venue fields:

```bibtex
@inproceedings{lakshminarayanan2017,
  title     = {Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles},
  author    = {Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017},
  publisher = {Curran Associates, Inc.},
  url       = {https://arxiv.org/abs/1612.01474},
  note      = {NIPS 2017}
}
```

Pages and editors not filled in — user can add precise page range and the NIPS 2017 editor list if desired. Current fields are enough for `sn-basic` to render a clean entry with venue, volume, year, publisher.

See FIXES_LOG §B.18.

---

## When this file gets out of date

After any commit that addresses a punch-list item or bib issue, mark the row DONE and add a one-line entry to `JCHEMINF_FIXES_LOG.md` describing the edit and any smoke tests run. Keep this status doc the single forward-looking source.
