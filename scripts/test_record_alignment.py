#!/usr/bin/env python
"""The reader half of verification gate 8 (RERUN_PLAN.md §2.7, §8).

`parse_mmap` walks a packed byte stream with no delimiters and no per-record
length. If a record is written short, every field after it is decoded from the
wrong offset — and until now the function ended in a bare `except: continue`,
which stepped over the exception and kept going with an offset that could never
be recovered. Whole replicates came out wildly negative and were deleted by the
catastrophic-run filter; this is one of the things that could produce that.

Four properties are asserted here, each one failing if the corresponding guard
is removed:

  1. a short record raises rather than returning silently wrong features;
  2. a well-formed file is consumed exactly, to the last byte;
  3. a representation the writer emits and the reader cannot step over is
     refused by name, instead of shifting every later field by its width;
  4. every representation this reader accepts decodes to a FIXED width, so the
     uniform-width check downstream is a backstop and not live protection.

On (4): `parse_mmap` carries a check that all rows came out the same width,
whose comment claims it catches "two misaligned records of the SAME wrong
width". It cannot, and the audit's suggested test -- pack records of two
different widths -- cannot be written. Every representation the reader accepts
reads a FIXED number of bytes, so a given representation's rows are all the
same width by construction; a truncated block is swallowed by the fixed-size
read and shows up as the NEXT field failing, which the per-entry guard catches
first. The two representations whose rows could genuinely differ in width --
one-hot SMILES and randomized SMILES -- are refused by name before the loop
starts (DROPPED_REPS).

So the check is unreachable today. It is worth keeping as a backstop against a
future variable-width representation, and (4) is the test that makes that
assumption fail loudly if one is ever added: it asserts, per representation,
that two molecules of different SMILES length decode to the same width.

Run it directly:  python scripts/test_record_alignment.py
"""

import io
import os
import struct
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process_and_train import parse_mmap  # noqa: E402

FP_BYTES = 256


def record(smiles, y_clean, y_written, fingerprint=None, short=False):
    """One ECFP4-only record, in the exact layout the Rust writer produces."""
    b = smiles.encode()
    out = struct.pack("I", len(b)) + b          # isomeric SMILES
    out += struct.pack("I", len(b)) + b          # canonical SMILES
    out += struct.pack("f", y_clean)             # raw label
    out += struct.pack("f", y_written)           # written (standardised) label
    if fingerprint is None:
        # Not zeros: a zero-filled tail makes the next record's length prefix
        # read as 0, which decodes cleanly and hides the misalignment. Real
        # fingerprints are not all zeros.
        fingerprint = bytes((i * 13 + 7) % 256 for i in range(FP_BYTES))
    out += fingerprint[: FP_BYTES // 2] if short else fingerprint
    return out


def make_file(records):
    return io.BytesIO(b"".join(records))


def parse(buf, count, reps=("ecfp4",), rep="ecfp4"):
    buf.seek(0)
    return parse_mmap(buf, count, rep, list(reps), 1, 0.0, False)


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False
    except Exception:
        print(f"  FAIL  {name}: unexpected error")
        traceback.print_exc()
        return False
    print(f"  OK    {name}")
    return True


def a_well_formed_file_parses_and_is_consumed_exactly():
    fps = [bytes([(i * 7 + j) % 256 for j in range(FP_BYTES)]) for i in range(5)]
    recs = [record(f"C{'C' * i}O", 4.0 + i, 0.1 * i, fps[i]) for i in range(5)]
    x, y, y_orig = parse(make_file(recs), 5)
    assert x.shape == (5, FP_BYTES * 8), f"unexpected feature shape {x.shape}"
    assert len(y) == 5 and len(y_orig) == 5
    assert np.allclose(y_orig, [4.0, 5.0, 6.0, 7.0, 8.0]), y_orig
    # The bits must be the ones written, not a shifted neighbour's.
    expected = np.unpackbits(np.frombuffer(fps[3], dtype=np.uint8), bitorder="little")
    assert np.array_equal(x[3], expected), "record 3 did not decode to its own fingerprint"


def a_short_record_raises_instead_of_misaligning():
    recs = [record(f"C{'C' * i}O", 4.0 + i, 0.1 * i) for i in range(5)]
    # Record 2 is 128 bytes short — exactly what the deleted `continue`
    # statements produced when a molecule could not be fingerprinted.
    recs[2] = record("CCCO", 6.0, 0.2, short=True)
    try:
        parse(make_file(recs), 5)
    except RuntimeError as e:
        msg = str(e)
        assert "entry" in msg or "consumed" in msg, f"unhelpful message: {msg}"
        assert "misaligned" in msg or "offset" in msg, f"unhelpful message: {msg}"
        return
    raise AssertionError(
        "a short record was parsed without complaint — the reader is silently "
        "misaligned from record 2 onwards"
    )


def trailing_bytes_are_not_ignored():
    recs = [record(f"C{'C' * i}O", 4.0 + i, 0.1 * i) for i in range(4)]
    recs.append(b"\x00" * 32)  # a fragment the record count does not account for
    try:
        parse(make_file(recs), 4)
    except RuntimeError as e:
        assert "consumed" in str(e), f"unhelpful message: {e}"
        return
    raise AssertionError("32 unread bytes were left at the end and nothing said so")


def a_representation_with_no_reader_is_refused_by_name():
    recs = [record(f"C{'C' * i}O", 4.0 + i, 0.1 * i) for i in range(3)]
    try:
        # `morgan` was deleted from the Rust writer on 2026-08-26, so it now
        # stands for any representation the writer might emit that this reader has
        # never had a branch for it.
        parse(make_file(recs), 3, reps=("morgan", "ecfp4"))
    except RuntimeError as e:
        assert "morgan" in str(e), f"the message must name the representation: {e}"
        return
    raise AssertionError("an unreadable representation was accepted silently")


def every_accepted_representation_is_fixed_width():
    """The assumption the uniform-width check rests on, per representation.

    If this fails, someone has added a representation whose rows can differ in
    width -- and the width check in `parse_mmap` then fires on VALID data.
    Revisit that check before the representation ships.
    """
    from process_and_train import (DROPPED_REPS, PARSEABLE_REPS, CHEMBERTA_BYTES,
                                   MHGGNN_BYTES, SNS_RECORD_BYTES)

    # bytes each representation contributes to one record, in the reader's order
    block_bytes = {
        "sns": SNS_RECORD_BYTES,
        "continuous_pdv": 800,
        "chemberta": CHEMBERTA_BYTES,
        "mhggnn": MHGGNN_BYTES,
        "avalon": 256,
        "ecfp4": FP_BYTES,
    }
    accepted = (PARSEABLE_REPS - DROPPED_REPS) - {"graph"}
    missing = accepted - set(block_bytes)
    assert not missing, (
        f"this test does not know the record layout of {sorted(missing)}, so it cannot "
        f"say whether their rows are a fixed width. Add them here."
    )

    for name in sorted(accepted):
        n = block_bytes[name]
        # Deliberately different SMILES lengths: length is the only thing that
        # varies between molecules, so if any width could move, it moves here.
        widths = set()
        for smiles, ylen in (("CO", 4.0), ("CCCCCCCCCCO", 5.0)):
            b = smiles.encode()
            rec = struct.pack("I", len(b)) + b
            rec += struct.pack("I", len(b)) + b
            rec += struct.pack("f", ylen)
            if name == "ecfp4":
                # ECFP4 is written LAST, after the processed target.
                rec += struct.pack("f", 0.5)
                rec += bytes((i * 13 + 7) % 256 for i in range(n))
            else:
                rec += bytes((i * 13 + 7) % 256 for i in range(n))
                rec += struct.pack("f", 0.5)
            buf = io.BytesIO(rec)
            buf.seek(0)
            x, _, _ = parse_mmap(buf, 1, name, [name], 1, 0.0, False)
            widths.add(np.asarray(x).shape[1])
        assert len(widths) == 1, (
            f"{name}: two molecules of different SMILES length decoded to widths "
            f"{sorted(widths)}. The uniform-width check in parse_mmap would fire on "
            f"valid data for this representation."
        )


def main():
    print("record alignment (RERUN_PLAN.md gate 8)")
    results = [
        check("a well-formed file parses and is consumed exactly",
              a_well_formed_file_parses_and_is_consumed_exactly),
        check("a short record raises instead of misaligning",
              a_short_record_raises_instead_of_misaligning),
        check("trailing bytes are not ignored",
              trailing_bytes_are_not_ignored),
        check("a representation with no reader is refused by name",
              a_representation_with_no_reader_is_refused_by_name),
        check("every accepted representation decodes to a fixed width",
              every_accepted_representation_is_fixed_width),
    ]
    if not all(results):
        print("\nFAIL: the reader can be silently misaligned")
        return 1
    print("\nOK: the reader cannot be silently misaligned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
