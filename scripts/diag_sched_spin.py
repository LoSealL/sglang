#!/usr/bin/env python3
"""Diagnose which threads inside sglang schedulers burn CPU at idle.

No root / no ptrace: samples /proc/<pid>/task/*/stat utime+stime twice and
reports per-thread CPU% with thread names. 100 ticks/s (USER_HZ).

Usage: diag_sched_spin.py [sample_seconds] [--all-pids]
"""
import os
import subprocess
import sys
import time

SAMPLE_S = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "--all-pids" else 2
ALL = "--all-pids" in sys.argv
HZ = os.sysconf("SC_CLK_TCK")


def sched_pids():
    out = subprocess.run(["pgrep", "-f", "sglang::scheduler"],
                         capture_output=True, text=True).stdout.split()
    return [int(p) for p in out] if ALL else [int(p) for p in out][:2]


def snapshot(pid):
    snap = {}
    base = f"/proc/{pid}/task"
    for tid in os.listdir(base):
        try:
            with open(f"{base}/{tid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            utime, stime = int(fields[11]), int(fields[12])  # after comm field
            snap[tid] = utime + stime
        except (FileNotFoundError, ProcessLookupError, IndexError):
            pass
    return snap


def wchan_and_state(pid, tid):
    try:
        with open(f"/proc/{pid}/task/{tid}/status") as f:
            state = next(l for l in f if l.startswith("State:")).split()[1]
        with open(f"/proc/{pid}/task/{tid}/wchan") as f:
            wchan = f.read().strip()
    except (FileNotFoundError, StopIteration):
        state, wchan = "?", "?"
    return state, wchan


for pid in sched_pids():
    try:
        comm = open(f"/proc/{pid}/comm").read().strip()
    except FileNotFoundError:
        continue
    s1 = snapshot(pid)
    time.sleep(SAMPLE_S)
    s2 = snapshot(pid)
    rows = []
    for tid, t2 in s2.items():
        d = t2 - s1.get(tid, t2)
        if d > SAMPLE_S * HZ * 0.10:  # >10% cpu
            name = "?"
            try:
                name = open(f"/proc/{pid}/task/{tid}/comm").read().strip()
            except FileNotFoundError:
                pass
            state, wchan = wchan_and_state(pid, tid)
            rows.append((d / SAMPLE_S / HZ, tid, name, state, wchan))
    rows.sort(reverse=True)
    print(f"== pid {pid} ({comm}) — threads >10% CPU over {SAMPLE_S}s ==")
    if not rows:
        print("   (none — idle)")
    for pct, tid, name, state, wchan in rows[:6]:
        print(f"   tid {tid:<7} {name:<16} {pct:5.0f}%  state={state} wchan={wchan}")
