#!/usr/bin/env python3
"""No file -- in the tree or in HEAD -- is carrying a mutation the fix-guard harness planted.

WHY THIS EXISTS
---------------
`check_fixes_fail_when_removed.py` proves each guard fails when its fix is
removed. To do that it BREAKS the real source file, runs the guard, and puts the
file back. Between those two steps the repository is holding code nobody would
ship.

Its `finally` covers an exception and a non-zero exit. It does not cover the
process being killed -- a timeout, a Ctrl-C, a session ending -- and it has been
killed at least three times: twice on 2026-08-27 (`utils.py` left with a
`save_results` that raises, `models.py` with the predicted variance sliced off
again) and once on 2026-08-28.

The backup directory recovers those, but only when the harness is next RUN. On
2026-08-28 the gap was longer than that: commit `a22d45a` committed
`models/models.py` carrying

    def bnn_elbo_criterion(base_criterion, model, n_train):
        return base_criterion  # BROKEN ON PURPOSE

and it sat in HEAD through the commit that followed. Nothing failed. The
harness's own guard passes on that file, because the harness restores the
correct version before it reports. BNN-alpha and BNN-beta -- two of the roster's
fourteen models, on every representation -- would have trained with plain MSE
and no KL term, which is the defect the commit above it says it closed.

So: the payload the harness plants must not survive into a committed file, and
the one thing no other check looks at is HEAD.

WHAT IT CHECKS
--------------
The anchor and the payload both come from the harness's own CASES list, read at
run time, so a new case is covered the day it is added and this file never
restates one.

A mutation is the payload PRESENT and the anchor ABSENT, in the working tree or
in `git show HEAD:`, asked of each file's own repository. Both halves are needed.
Some payloads are ordinary code -- `    pass` is one -- so presence alone
over-reports; and a live session may legitimately edit an anchor away, so absence
alone does too.

    python scripts/test_no_harness_mutation_committed.py

DO NOT RUN IT WHILE THE HARNESS IS RUNNING. It will find the mutation the
harness has planted on purpose, which is the correct answer to the question it
asks. It says so rather than guessing.
"""
import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HARNESS = os.path.join(HERE, "check_fixes_fail_when_removed.py")
BACKUP_DIR = os.path.join(HERE, ".harness_unrestored")

# A mutation is the payload PRESENT and the anchor ABSENT -- both halves, because
# neither is enough on its own. Some payloads are ordinary code (`    pass`) and
# some are short (`    CONFORMAL_MODELS = ()`), so presence alone over-reports; and
# a live session may legitimately edit an anchor away, so absence alone does too.
#
# A length rule was tried first and let `CONFORMAL_MODELS = ()` through as
# "unchecked" on the very day that case was added -- which is the shape of every
# other defect in this project: a check that reports a gap rather than closing it.


def harness_cases():
    """(name, path, anchor, payload) for every case in the harness's CASES list.

    Read from the source rather than imported: importing the harness runs it,
    and running it breaks files.
    """
    tree = ast.parse(open(HARNESS).read())
    consts = {n.targets[0].id: n.value.value
              for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
              and isinstance(n.targets[0], ast.Name)}

    def literal(node):
        """ast.literal_eval, plus f-strings whose only holes are module constants."""
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError):
            pass
        if not isinstance(node, ast.JoinedStr):
            return None
        out = ""
        for piece in node.values:
            if isinstance(piece, ast.Constant):
                out += piece.value
            elif isinstance(piece, ast.FormattedValue) and isinstance(piece.value, ast.Name):
                if piece.value.id not in consts:
                    return None
                out += str(consts[piece.value.id])
            else:
                return None
        return out

    for node in tree.body:
        if not (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "CASES"):
            continue
        for entry in node.value.elts:
            name = literal(entry.elts[0])
            path = literal(entry.elts[1])
            anchor = literal(entry.elts[2])
            payload = literal(entry.elts[3])
            if None in (name, path, anchor, payload):
                yield ("<unreadable case>", None, None, None)
            else:
                yield (name, path, anchor, payload)
        return
    sys.exit(f"{HARNESS} has no CASES list -- this check cannot read what it plants.")


def head_text(path):
    """The committed version of `path`, or None if HEAD does not have the file.

    Asked of the file's OWN repository, not this one. The harness mutates
    KIRBy and NoiseInject as well, and those are separate checkouts -- a
    `git -C .` here would report them as "not in HEAD" and check nothing,
    which is the opposite of what this file is for.
    """
    directory = os.path.dirname(os.path.abspath(path))
    top = subprocess.run(["git", "-C", directory, "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    if top.returncode != 0:
        return None
    repo = top.stdout.strip()
    rel = os.path.relpath(os.path.abspath(path), repo)
    done = subprocess.run(["git", "-C", repo, "show", f"HEAD:{rel}"],
                          capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else None


def main():
    if os.path.isdir(BACKUP_DIR) and os.listdir(BACKUP_DIR):
        left = ", ".join(sorted(os.listdir(BACKUP_DIR)))
        sys.exit(
            f"the fix-guard harness is running now, or was killed and has not been re-run.\n"
            f"{BACKUP_DIR} still holds: {left}\n"
            f"Wait for it to finish, or run check_fixes_fail_when_removed.py, which puts the "
            f"file back before it starts. Anything this check found in the meantime would be "
            f"the mutation the harness planted on purpose.")

    findings = []
    unchecked = []
    checked = 0

    anchors = {}
    for name, path, anchor, payload in harness_cases():
        anchors[name] = anchor
        if path is None:
            unchecked.append((name, "the harness case could not be read"))
            continue
        if not os.path.exists(path):
            findings.append((name, path, "the file the harness mutates does not exist"))
            continue
        checked += 1

        for where, text in (("the WORKING TREE", open(path).read()),
                            ("HEAD", head_text(path))):
            if text is None:
                unchecked.append((name, f"{os.path.relpath(path, ROOT)} is not in HEAD"))
                continue
            if payload not in text:
                continue
            if anchors[name] in text:
                # Both present: the payload reads as ordinary code here, not as a
                # replacement for the anchor. `    pass` is the usual reason.
                continue
            findings.append((
                name, path,
                f"{where} has the mutation payload and NOT the line it replaced"
                + (" -- it was COMMITTED broken" if where == "HEAD" else "")))

    for name, why in unchecked:
        print(f"  UNCHECKED  {name:52} {why}")
    if unchecked:
        print()

    if findings:
        print(f"{len(findings)} file(s) carry a fix-guard mutation:\n")
        for name, path, where in findings:
            print(f"  {name}")
            print(f"    {os.path.relpath(path, ROOT)}")
            print(f"    {where}\n")
        sys.exit(
            "Put the file back before anything is submitted. `git show HEAD:<path>` and the "
            "harness's CASES entry between them say what the correct line is; the harness's "
            f"backup of a killed run, if there is one, is in {BACKUP_DIR}.")

    print(f"PASSED -- {checked} mutation payload(s) absent from the working tree and from HEAD"
          + (f"; {len(unchecked)} not checkable" if unchecked else ""))


if __name__ == "__main__":
    main()
