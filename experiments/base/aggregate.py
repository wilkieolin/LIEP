#!/usr/bin/env python3
"""LIEP-BASE aggregation → results/summary.csv (long format: item, arm, model, epoch, setting, metric, value).

Per (arm, model, epoch, epochs-of-run): median, min, max and n over seeds of test accuracy. Standard EP: each arm
(size × activation) uses the tuning grid with the higher best validation accuracy (grid 1 or grid 2; README), and the
chosen grid and its rate/β are recorded. Usage: python3 experiments/base/aggregate.py"""
import csv, glob, os, re, statistics as st
from collections import defaultdict
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = []
def emit(arm, model, epoch, setting, metric, value):
    rows.append(dict(item="LIEP-BASE", arm=arm, model=model, epoch=epoch, setting=setting, metric=metric, value=value))
def summarise(recs, arm, model, setting=""):
    by = defaultdict(list)
    for r in recs:
        by[int(r["epoch"])].append(float(r["value"]))
    for ep, vs in sorted(by.items()):
        for s, v in (("median", st.median(vs)), ("min", min(vs)), ("max", max(vs)), ("n", len(vs))):
            emit(arm, model, ep, setting, f"test_acc_{s}", v)

read = lambda f: list(csv.DictReader(open(f)))
for f in glob.glob(os.path.join(R, "lr", "*.csv")):
    recs = [r for r in read(f) if r["metric"] == "test_acc"]
    summarise([dict(r, epoch=30) for r in recs], "logreg", recs[0]["model"], recs[0]["setting"])
for f in glob.glob(os.path.join(R, "lrbasis", "*.csv")):     # README addendum 2026-11-17
    recs = [r for r in read(f) if r["metric"] == "test_acc"]
    summarise([dict(r, epoch=30) for r in recs], "lrbasis", recs[0]["model"], recs[0]["setting"])
pb = [r for f in glob.glob(os.path.join(R, "phasorbp", "*.csv")) for r in read(f) if r["metric"] == "test_acc"]
summarise(pb, "phasor_bp", "784-256-64")
relu = defaultdict(list)
for f in glob.glob(os.path.join(R, "relu", "*.csv")):
    for r in read(f):
        if r["metric"] == "test_acc":
            relu[(r["model"], r["setting"])].append(r)
for (m, s), recs in relu.items():
    summarise(recs, "relu_bp", m, s)
probes = defaultdict(list)
for f in glob.glob(os.path.join(R, "probe", "*.csv")):
    for r in read(f):
        probes[r["model"]].append(r)
for arm, recs in probes.items():
    summarise(recs, "probe", arm)
for H in ("256", "500"):
    for act in ("clip", "half"):
        best = None
        for g, suffix in (("1", ""), ("2", "_grid2")):
            f = os.path.join(R, "sep", f"sep_{H}_{act}{suffix}.csv")
            if not os.path.exists(f):
                continue
            recs = read(f); val = max(float(r["value"]) for r in recs if r["metric"] == "val_acc_e2")
            emit("standard_ep", f"784-{H}-10 {act}", 2, f"grid{g}", "best_val_acc_e2", val)
            if best is None or val > best[0]:
                best = (val, g, recs)
        val, g, recs = best
        fin = [r for r in recs if r["metric"] == "test_acc"]
        setting = fin[0]["setting"].split(" epochs=")[0]
        emit("standard_ep", f"784-{H}-10 {act}", -1, f"grid{g} {setting}", "chosen", 1)
        summarise([r for r in fin if "epochs=8" in r["setting"]], "standard_ep", f"784-{H}-10 {act}", "epochs=8")
        summarise([r for r in fin if "epochs=30" in r["setting"]], "standard_ep", f"784-{H}-10 {act}", "epochs=30")
with open(os.path.join(R, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["item", "arm", "model", "epoch", "setting", "metric", "value"]); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows")
