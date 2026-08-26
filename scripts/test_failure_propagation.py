#!/usr/bin/env python
"""A failure in the noise injector must stop the run. RERUN_PLAN.md §2.8e.

Why this exists. Chat A and chat D added a dozen hard failures to the Rust
injector -- dose gates, the molecule-identity assertion, the held-out checks, a
molecule that cannot be fingerprinted, a truncated record, a configuration that
will not open. Every one of them wrote to a pipe that nobody read.
`process_and_run` called `proc_a.communicate()`, printed stdout and stderr, and
carried on to reopen the memory-mapped files and train on whatever was there.
The return code was never inspected. Until that was fixed, every one of those
guards was decorative.

It is worse than a lost message. `preprocess_data` renames the rewritten
training file over the original BEFORE it processes val and test, so a run that
dies partway leaves the training split noised and the held-out splits clean --
and the pipeline would train and score on that combination without complaint,
producing numbers that look entirely reasonable.

Two more layers sat underneath. The noise-level handler printed only `if
logging`, which is off by default, and then `continue`: a failed noise level
produced no rows and no message at all. That is the same shape as jobs 12822693
and 12822694 (§2.8d), sitting in the QM9 pipeline.

Three things are asserted here, and each fails if its guard is removed:

  1. a non-zero exit from the injector raises, naming the exit code;
  2. the run reports the failure whether or not --logging is on;
  3. the process exits non-zero and calls the results INCOMPLETE, so a run that
     lost cells cannot look like a run that did not.

    python scripts/test_failure_propagation.py
"""

import os
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))


def source_still_has_the_check():
    """Cheap, and it fails without running the pipeline at all."""
    src = open(os.path.join(HERE, "process_and_train.py")).read()
    missing = [
        needle for needle in (
            "if proc_a.returncode != 0:",
            "failed_cells.append(",
            "sys.exit(1)",
        ) if needle not in src
    ]
    assert not missing, f"process_and_train.py no longer contains: {missing}"


def a_failing_injector_stops_the_run():
    """The real thing: run the pipeline with an injector that always fails."""
    fake = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
    fake.write("#!/bin/sh\necho 'simulated injector failure' >&2\nexit 3\n")
    fake.close()
    os.chmod(fake.name, 0o755)

    out_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name

    driver = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {HERE!r}); os.chdir({HERE!r})
        import process_and_train as p

        # Swap the injector for one that always fails, leaving every other code
        # path exactly as it runs on the cluster.
        real = p.subprocess.Popen
        class P(real):
            def __init__(self, cmd, *a, **k):
                if cmd and str(cmd[0]).endswith("rust_processor"):
                    cmd = [{fake.name!r}] + list(cmd[1:])
                super().__init__(cmd, *a, **k)
        p.subprocess.Popen = P

        sys.argv = ["x", "-d", "QM9", "-t", "homo_lumo_gap", "-m", "rf", "-r", "ecfp4",
                    "--noise-level", "0.4", "-n", "200", "-b", "1", "-s", "scaffold",
                    "--normalize", "True", "-f", {out_csv!r}]
        p.main()
    """)

    r = subprocess.run([sys.executable, "-c", driver],
                       capture_output=True, text=True, timeout=1800)
    out = (r.stdout + r.stderr).replace("\n", " ")

    try:
        assert r.returncode != 0, (
            "the pipeline exited 0 with a failing injector -- a run that lost every cell "
            "would look identical to a complete one"
        )
        assert "the noise injector exited 3" in out, (
            "the injector's exit code was not reported; the return code is being ignored again"
        )
        assert "ERROR at noise level" in out, (
            "the failed noise level was not reported. Note --logging was NOT passed: the old "
            "code printed only when it was, so the failure was silent by default"
        )
        assert "INCOMPLETE" in out, "the run did not say its results are incomplete"
    finally:
        for path in (fake.name, out_csv):
            try:
                os.remove(path)
            except OSError:
                pass


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        return False
    print(f"  OK    {name}")
    return True


def main():
    print("failure propagation (RERUN_PLAN.md §2.8e)")
    results = [
        check("the return-code check is still in the source",
              source_still_has_the_check),
        check("a failing injector stops the run and marks it incomplete",
              a_failing_injector_stops_the_run),
    ]
    if not all(results):
        print("\nFAIL: the injector's guards can fail without the pipeline noticing")
        return 1
    print("\nOK: a failure in the injector stops the run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
