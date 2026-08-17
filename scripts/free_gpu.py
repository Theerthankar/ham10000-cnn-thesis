#!/usr/bin/env python3
"""Shut down every idle Jupyter kernel except this notebook's own.

A finished notebook keeps its kernel alive holding a CUDA context and model,
roughly 3 GB for the 224px runs and 11 GB at 300px. Run this before training to
reclaim it. Refuses to act if any kernel is busy, so it cannot kill a live run.

Usable from a notebook cell:
    !python3 scripts/free_gpu.py --keep 14_E12_efficientnet_b3_oversampling_300.ipynb
or from a shell with no --keep to clear everything idle.
"""
import argparse, json, subprocess, time, urllib.request


def server():
    out = subprocess.run(["jupyter", "server", "list"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "token=" in line:
            url = line.split(" :: ")[0].strip()
            base, token = url.split("/?token=")
            return base, token
    raise SystemExit("No running Jupyter server found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default="", help="notebook filename to leave running")
    args = ap.parse_args()
    base, token = server()

    def api(path, method="GET"):
        r = urllib.request.Request(base + path, method=method)
        r.add_header("Authorization", f"token {token}")
        with urllib.request.urlopen(r, timeout=15) as x:
            b = x.read()
            return json.loads(b) if b else None

    sessions = api("/api/sessions")
    busy = [s for s in sessions
            if s["kernel"]["execution_state"] == "busy"
            and not ((s.get("notebook") or {}).get("path", "")).endswith(args.keep or "\0")]
    if busy:
        names = [((s.get("notebook") or {}).get("path", "")).split("/")[-1] for s in busy]
        raise SystemExit(f"Refusing to act: these kernels are busy: {names}")

    stopped = 0
    for s in sessions:
        nb = ((s.get("notebook") or {}).get("path") or "").split("/")[-1]
        if args.keep and nb == args.keep:
            continue
        api(f"/api/sessions/{s['id']}", method="DELETE")
        print(f"stopped {nb}")
        stopped += 1

    if stopped:
        time.sleep(5)
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
