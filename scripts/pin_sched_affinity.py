#!/usr/bin/env python3
"""Pin each sglang scheduler to a NUMA-local core slice (no root, own procs).

Why: torch-gloo transport threads (24 per rank: 8 PGs x runloop/tcp loops)
busy-poll ~50-70% CPU per rank at idle. Pinning contains the spin and puts
each rank next to its GPU's memory. Undo: pin_sched_affinity.py -u.

Usage: pin_sched_affinity.py [-u]
"""
import os
import re
import subprocess
import sys

UNDO = "-u" in sys.argv


def numa_cpulist():
    """node -> sorted cpu list, from sysfs."""
    nodes = {}
    for n in os.listdir("/sys/devices/system/node"):
        m = re.match(r"node(\d+)$", n)
        if not m:
            continue
        with open(f"/sys/devices/system/node/{n}/cpulist") as f:
            cpus = []
            for part in f.read().strip().split(","):
                if "-" in part:
                    a, b = map(int, part.split("-"))
                    cpus += list(range(a, b + 1))
                else:
                    cpus.append(int(part))
            nodes[int(m.group(1))] = sorted(cpus)
    return nodes


def gpu_numa():
    """gpu index -> numa node (via PCI bus id)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    mapping = {}
    for line in out.strip().splitlines():
        idx, bus = [x.strip() for x in line.split(",")]
        bus = bus.replace("0000", "", 1) if bus.startswith("0000") else bus
        path = f"/sys/bus/pci/devices/{bus.lower()}/numa_node"
        try:
            mapping[int(idx)] = int(open(path).read())
        except (FileNotFoundError, ValueError):
            mapping[int(idx)] = -1
    return mapping


pids = subprocess.run(["pgrep", "-f", "sglang::scheduler"],
                      capture_output=True, text=True).stdout.split()
if not pids:
    sys.exit("no sglang::scheduler processes running")

all_cpus = sorted(c for cpus in numa_cpulist().values() for c in cpus)

if UNDO:
    for pid in pids:
        subprocess.run(["taskset", "-apc", f"{all_cpus[0]}-{all_cpus[-1]}", pid],
                       capture_output=True)
        print(f"pid {pid}: reset to {all_cpus[0]}-{all_cpus[-1]}")
    sys.exit(0)

nodes = numa_cpulist()
gpu_node = gpu_numa()

# group ranks by numa node
rank_node = {}
for pid in pids:
    args = subprocess.run(["ps", "-o", "args=", "-p", pid],
                          capture_output=True, text=True).stdout
    m = re.search(r"TP(\d+)", args)
    rank = int(m.group(1)) if m else None
    if rank is not None and rank in gpu_node and gpu_node[rank] in nodes:
        rank_node[pid] = gpu_node[rank]
    else:  # fallback: round-robin over nodes by pid order
        rank_node[pid] = sorted(nodes)[len(rank_node) % len(nodes)]

# split each node's cpus evenly among its ranks; interleave HT siblings apart
by_node = {}
for pid, node in rank_node.items():
    by_node.setdefault(node, []).append(pid)

for node, pids_ in sorted(by_node.items()):
    cpus = nodes[node]
    half = len(cpus) // 2
    # assume HT siblings are [first_half | second_half] pairing i<->i+half
    phys, ht = cpus[:half], cpus[half:]
    n = len(pids_)
    per = max(1, half // n)
    for i, pid in enumerate(sorted(pids_)):
        lo, hi = i * per, (i + 1) * per if i < n - 1 else half
        core_set = sorted(phys[lo:hi] + [ht[j] for j in range(lo, hi)
                                         if j < len(ht) and ht[j] - phys[j] == half])
        # simpler: take matching HTs by index
        core_set = sorted(set(phys[lo:hi]) | set(ht[lo:hi]))
        spec = ",".join(str(c) for c in core_set)
        r = subprocess.run(["taskset", "-apc", spec, pid], capture_output=True, text=True)
        ok = "ok" if r.returncode == 0 else f"FAILED: {r.stderr.strip()}"
        print(f"pid {pid} node{node}: {len(core_set)} cpus [{spec.split(',')[0]}..{spec.split(',')[-1]}] {ok}")
