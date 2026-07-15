#!/usr/bin/env python3
"""
loop_test.py — Soak-test the action-aware pipeline by looping it.

Instead of building one giant video, this re-runs the pipeline (main.py) on the
ORIGINAL clip over and over until a wall-clock time budget elapses.  Every
iteration uses the exact same command you'd normally run, so --out-dir /
--chunks-dir / --run are forwarded unchanged: per-run files overwrite each
iteration and UUID-named chunks accumulate — i.e. it behaves as one run.

This script does NOT touch any pipeline code — it only shells out to `main.py`,
forwarding every flag you pass through verbatim, with the original video as the
positional argument.

── Usage ──────────────────────────────────────────────────────────────────
Do NOT pass the video to main.py's positional slot — this script injects it.
Pass --video + --loop-duration (wall-clock seconds), then the usual main.py
flags:

  python loop_test.py \\
      --video ../baby_vids/N1.mp4 \\
      --loop-duration 3600 \\
      --filter true --detect false --chunk true \\
      --fps 10 --workers 1 --device cuda \\
      --audio-spikes true --benchmark true \\
      --duration 5 --chunks-dir ./chunks_full_test_final \\
      --site site-001 --camera cam-001 --run run-001 \\
      --out-dir ./output_full_test_final

Re-runs the pipeline on N1.mp4 back-to-back for ~1 hour of wall-clock time.

  --loop-duration N   Wall-clock budget in seconds. A new iteration starts only
                      while elapsed < N; the last one may overrun the budget.
  --gap SECONDS       Optional pause between iterations (default 0).
  --stop-on-error     Stop the loop if an iteration exits non-zero (default:
                      keep going and count failures).
  --python EXE        Interpreter used to run main.py (default: this one).
  --dry-run           Print the plan + command and exit without running.
──────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MAIN = _HERE / "main.py"


def _fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def main() -> int:
    p = argparse.ArgumentParser(
        prog="loop_test.py",
        description="Loop the pipeline on a video for a wall-clock duration (single run).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        allow_abbrev=False,   # never swallow a passthrough flag by prefix-matching
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Source video to process each iteration (e.g. ../baby_vids/N1.mp4)")
    p.add_argument("--loop-duration", required=True, type=float, metavar="SECONDS",
                   help="Wall-clock time budget in seconds; loop while elapsed < N")
    p.add_argument("--gap", type=float, default=0.0, metavar="SECONDS",
                   help="Optional pause between iterations (default 0)")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Stop the loop if an iteration exits non-zero")
    p.add_argument("--python", default=sys.executable, metavar="EXE",
                   help="Interpreter used to run main.py (default: this interpreter)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and command, then exit without running")

    # Everything else is forwarded to main.py untouched.
    args, passthrough = p.parse_known_args()

    if not args.video.is_file():
        sys.exit(f"error: video not found: {args.video}")
    if args.loop_duration <= 0:
        sys.exit("error: --loop-duration must be > 0")
    if args.gap < 0:
        sys.exit("error: --gap must be >= 0")
    if not _MAIN.is_file():
        sys.exit(f"error: main.py not found next to this script: {_MAIN}")

    cmd = [args.python, str(_MAIN), str(args.video), *passthrough]

    print(
        f"[loop] video={args.video}  budget={args.loop_duration:.0f}s "
        f"({_fmt_hms(args.loop_duration)})  gap={args.gap:.0f}s",
        file=sys.stderr,
    )
    print("[loop] per-iteration command:\n  " + " ".join(cmd), file=sys.stderr)

    if args.dry_run:
        print("[loop] --dry-run set; not executing.", file=sys.stderr)
        return 0

    start = time.monotonic()
    iteration = 0
    failures = 0
    last_code = 0
    durations: list[float] = []

    try:
        # A new iteration starts only while still inside the budget; the last
        # iteration may overrun (we never kill a run mid-flight).
        while (time.monotonic() - start) < args.loop_duration:
            iteration += 1
            elapsed = time.monotonic() - start
            remaining = args.loop_duration - elapsed
            print(
                f"\n[loop] ── iteration {iteration}  "
                f"elapsed={_fmt_hms(elapsed)}  remaining={_fmt_hms(remaining)} "
                "──────────────",
                file=sys.stderr,
            )

            it_t0 = time.monotonic()
            code = subprocess.run(cmd, cwd=_HERE).returncode
            it_dur = time.monotonic() - it_t0
            durations.append(it_dur)
            last_code = code

            if code != 0:
                failures += 1
                print(f"[loop] iteration {iteration} exited {code} "
                      f"({it_dur:.1f}s)", file=sys.stderr)
                if args.stop_on_error:
                    print("[loop] --stop-on-error set; stopping.", file=sys.stderr)
                    break
            else:
                print(f"[loop] iteration {iteration} ok ({it_dur:.1f}s)",
                      file=sys.stderr)

            if args.gap > 0 and (time.monotonic() - start) < args.loop_duration:
                time.sleep(args.gap)
    except KeyboardInterrupt:
        print("\n[loop] interrupted by user — stopping.", file=sys.stderr)

    total_elapsed = time.monotonic() - start
    avg = sum(durations) / len(durations) if durations else 0.0
    print(
        "\n[loop] ── summary ──────────────────────────────────────────\n"
        f"[loop]   iterations   : {iteration}\n"
        f"[loop]   failures     : {failures}\n"
        f"[loop]   wall elapsed : {_fmt_hms(total_elapsed)} "
        f"(budget {_fmt_hms(args.loop_duration)})\n"
        f"[loop]   avg / iter   : {avg:.1f}s\n"
        "[loop] ─────────────────────────────────────────────────────",
        file=sys.stderr,
    )

    # Non-zero exit if any iteration failed, so CI / scripts can detect it.
    return last_code if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
