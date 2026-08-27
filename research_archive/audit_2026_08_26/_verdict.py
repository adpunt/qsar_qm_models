#!/usr/bin/env python3
"""Record a verdict for one audit candidate and drop it from unverified.json.

    python3 _verdict.py <index-in-original-list> <verdict> <evidence...>

Verdicts: real-fixed, real-open, refuted, duplicate, not-a-fault
The original index is the position in the 111-entry list as it was on
2026-08-27; entries carry it in `orig_index` once this script has run once.
"""
import json, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
UNV = os.path.join(HERE, "unverified.json")
VER = os.path.join(HERE, "verdicts.json")


def load():
    unv = json.load(open(UNV))
    for i, e in enumerate(unv):
        e.setdefault("orig_index", i)
    return unv


def main():
    idx = int(sys.argv[1])
    verdict = sys.argv[2]
    evidence = " ".join(sys.argv[3:])
    unv = load()
    hit = [e for e in unv if e["orig_index"] == idx]
    assert hit, f"no unverified entry with orig_index {idx}"
    entry = dict(hit[0])
    entry["verdict"] = verdict
    entry["evidence"] = evidence
    verdicts = json.load(open(VER)) if os.path.exists(VER) else []
    verdicts = [v for v in verdicts if v.get("orig_index") != idx]
    verdicts.append(entry)
    json.dump(verdicts, open(VER, "w"), indent=1)
    rest = [e for e in unv if e["orig_index"] != idx]
    json.dump(rest, open(UNV, "w"), indent=1)
    print(f"#{idx} -> {verdict}; {len(rest)} left in unverified.json")


if __name__ == "__main__":
    main()
