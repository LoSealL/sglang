#!/usr/bin/env python3
"""Release page cache of the model tree without root (POSIX_FADV_DONTNEED).

Node0's cache stays pinned-full (465G), forcing per-page eviction during
buffered reads — the 10x slow weight-load path. Run this before restarting
the server to hand the loader a fresh cache headroom.

Usage: drop_model_cache.py [path ...]   (default: the GLM-5.2-FP8 tree)
"""
import ctypes
import glob
import os
import re
import sys

libc = ctypes.CDLL(None, use_errno=True)
POSIX_FADV_DONTNEED = 4


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.strip().split()[0]) // 1024  # MiB
    return d


def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        if libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0:
            e = ctypes.get_errno()
            print(f"  fadvise failed on {path}: {os.strerror(e)}")
    finally:
        os.close(fd)


paths = sys.argv[1:] or ["/home/tangwy/models_src/GLM-5.2-FP8"]
files = []
for p in paths:
    if os.path.isdir(p):
        files += [f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True)
                  if os.path.isfile(f) and not os.path.islink(f)]
    else:
        files.append(p)

before = meminfo()
total = sum(os.path.getsize(f) for f in files)
print(f"advising DONTNEED on {len(files)} files ({total/2**30:.0f} GiB)")
for f in files:
    try:
        drop(f)
    except OSError as e:
        print(f"  skip {f}: {e}")

after = meminfo()
print(f"Cached: {before['Cached']/1024:.1f}G -> {after['Cached']/1024:.1f}G  "
      f"MemFree: {before['MemFree']/1024:.1f}G -> {after['MemFree']/1024:.1f}G")
