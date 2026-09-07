# How to work on this project

Read this before anything else. The author has had to repeat most of it more than once.

## Write in plain English. This is the rule that matters most.

The author cannot use an answer they cannot read. Several sessions have been wasted on
messages that were technically correct and impossible to follow.

- **One fact per sentence.** Not three facts joined by dashes and commas.
- **Start with the answer.** One sentence saying what is true. Then the detail.
- **No arithmetic strings.** Never write "6 models × 3 datasets × 3 representations × 7
  conditions, split 162 and 216". Write it as a list, with a line each, and say what one
  of the things actually is.
- **No file names, function names or section numbers in conversation.** Say what the thing
  does. Keep the identifiers for the commands and for the written documents.
- **No coined shorthand.** If you invent a label, the author has to learn it to read you.
- **Spell out what something IS before you say anything about it.** "The uncertainty runs"
  means nothing on its own. "The jobs that test whether a model knows when it is wrong"
  does.
- **Before sending: read the whole message back.** Would someone who was not in your head
  know what to do at the end of it? If not, rewrite it.
- **Never open a message by correcting your last message.** The author needs to be able to
  trust the last number you gave them.

## Banned words

- **"arm"** and **"stage"** — say *noise type* or *condition*. For the run design say
  *the screen*, *the main grid*, *the deep run*, *the uncertainty runs*. "Stage" got used
  for four different things in three weeks and became unreadable.
- **"retention area"** — the metric is called **AUC_norm**. Say that.
- **"descriptor vector"** — it is called **PDV**. Say that.

## How to behave

- **Fix things. Do not report them.** A documented problem that is still broken is a
  failure, not progress. Never suggest stopping, pausing or deferring — find the defect,
  fix it, prove it with a test, push it.
- **Discuss before acting.** Do not present a plan as already decided.
- **Anything that changes the science is the author's decision. Anything a measurement can
  settle is yours.** Do not hand back a decision you were told to make. Do not take one you
  were not given.
- **Never state a number from memory.** Trace it to a file or to a command you ran, or say
  you have not checked.
- **Finish one thing before starting another.** If a tool is wrong, fix it once, prove it,
  and go straight back to the experiment.
- **Search the past conversations before saying something was never decided.** They are at
  `~/.claude/projects/-Users-apunt-repos-qsar-qm-models/*.jsonl`. The documents in this repo
  lag behind what was actually agreed.
- **"This is not being run" does not mean "delete it".** Check that nothing calls it, say so,
  and delete only when told to in as many words.

## The cluster

- **The author runs every command. You have no access.** So anything you need from the
  cluster has to be a block they can copy and paste in one go, never a conversation.
- **Batch the commands.** Send everything for several open questions at once. Prefer plain
  `sacct`, `squeue`, `ls` and `tail` over a script you just wrote, because a one-line
  command cannot have a bug you then have to fix in front of them.
- **Never send a command whose answer you cannot act on.**
- **A fix that is not pushed does not exist.** The cluster gets code only when the author
  runs `bash scripts/pull_safely.sh` against a commit that is already on the branch.
- **Never delete results yourself.** Write the delete command down with the reason, and let
  the author run it. A result deleted by mistake costs days.
- **Never type an array range by hand.** The generators write `submit_all.sh` with the right
  range for each script. Typing one has queued out-of-range tasks three times.
- Jobs run under `--account=stat-cadd`. Activate the environment with conda, not micromamba.

## The study

- **Every result is kept separate** by noise type, noise level, model and representation.
  Never average across representations. A median across replicates is fine.
- **The representations are exactly** PDV, MHG-GNN, Avalon, ECFP4, ChemBERTa and Sort &
  Slice. mol2vec is deleted. Never use Ridge.
- **Representation is one of the things being studied**, not a setting. Never propose
  reducing the study to one of them.
- **Every experiment uses scaffold splits.**
- **Noise level means the fraction of the clean training label spread**, not log units.

## Where things are written down

- **`RERUN_PLAN.md`** — what gets run, in what order, and every open thread. Section 13.26
  is the thread register. Section 13.27 is the command sheet.
- **`NOISE_DESIGN.md`** — what the noise is: the conditions, the maths, the sources.
- **Add to those two. Never start a new status file.** Six overlapping ones were purged once
  already.
- **`paper.tex` is never edited.** It is a read-only download from the author's Overleaf.
  Paper changes go into `PAPER_REVISION_GUIDE_FINAL.md` as replacement text.
