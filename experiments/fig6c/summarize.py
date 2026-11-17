"""LIEP-6C summaries (long format; FINDINGS and the Fig 6c export read these).
results/tune.csv:  one block per held-out run (seed 42): last-epoch accuracy (settled; feedforward for EP), rho, amp scale.
results/final.csv: per (panel arm, lr) with >= 3 test seeds: median/min/max/n test accuracy, feedforward accuracy, train
                   accuracy, hidden probe, rho, collapses. L1 pools LIEP-CLOSE's 3 test seeds (same script) with the 3 here.
results/compare.csv: the differences the predictions name (P1-P6), on medians.
results/panel.csv: one row per test run of the Fig 6c arms (arm, seed, acc), for the export."""
import csv, glob, os, statistics as st
from collections import defaultdict
D = os.path.dirname(os.path.abspath(__file__)); R = f"{D}/results"; L = os.path.join(D, "..")
med = st.median

def write(name, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{R}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)

def panel(kind, arm):                                       # CSV arm field -> Fig 6c arm id
    if kind == "ep":
        if arm == "lockin_adam_k0.1": return "L1"
        if arm.startswith("lockin_adam_k0.1_amp"): return "L2"
        if arm == "lockin_adam_k0.1_init": return "L3"
        if arm == "centred_adam_k0.1": return "L4"
    if kind == "bp": return {"adam": "B1", "adam_proj2": "B2", "muon": "B3"}.get(arm)
    if kind == "relu": return {"relu_adam": "R1", "relu_muon": "R2"}.get(arm)
    return None

runs = defaultdict(dict)                                    # (split, panel, csv arm, lr) -> seed -> {(epoch, metric): v}
files = [(k, f) for k in ("ep", "bp", "relu") for f in glob.glob(f"{R}/{k}/*/*.csv")]
files += [("ep", f) for f in glob.glob(f"{L}/close/results/test/lockin_adam_lr1e-2_k0.1_s*_test.csv")]
for kind, f in files:
    rs = list(csv.DictReader(open(f)))
    if not rs: continue
    r0 = rs[0]; p = panel(kind, r0["arm"])
    if p is None: continue
    key = (r0["split"], p, r0["arm"], float(f"{float(r0['lr']):.6g}"))
    runs[key][int(r0["seed"])] = {(int(r["epoch"]), r["metric"]): float(r["value"]) for r in rs}

def last(v, m):                                             # value of metric m at its last logged epoch
    es = [e for (e, mm) in v if mm == m]
    return v[(max(es), m)] if es else float("nan")

tune, final, rows = [], [], []
for (split, p, arm, lr), by in sorted(runs.items()):
    am = "val_acc" if split == "val" else "test_acc"; base = dict(item="LIEP-6C", panel=p, arm=arm, lr=lr)
    if split == "val":
        for s, v in by.items():
            for m, x in (("acc", last(v, am)), ("acc_ff", last(v, am + "_ff")), ("rho", last(v, "rho")),
                         ("amp_scale_median", last(v, "amp_scale_median")), ("train_acc", last(v, "train_acc")),
                         ("collapsed", last(v, "collapsed"))):
                if x == x: tune.append(dict(base, seed=s, metric=m, value=x))
        continue
    for s, v in sorted(by.items()): rows.append(dict(base, seed=s, acc=last(v, am)))
    if len(by) < 3: continue
    a = [last(v, am) for v in by.values()]
    stats = [("test_acc", med(a)), ("test_min", min(a)), ("test_max", max(a)), ("n", len(a))]
    for m in ("test_acc_ff", "train_acc", "probe_hidden", "rho"):
        xs = [last(v, m) for v in by.values()]; xs = [x for x in xs if x == x]
        if xs: stats.append((m, med(xs)))
    stats.append(("collapsed", sum(1 for x in a if x < 0.5)))
    final += [dict(base, metric=m, value=x) for m, x in stats]
write("tune.csv", tune); write("final.csv", final); write("panel.csv", rows)

# comparisons (medians; the pick per arm is the lr with the most test seeds)
F = {}
for r in final:
    if r["metric"] == "n": F.setdefault(r["panel"], []).append((r["value"], r["lr"]))
pick = {p: max(v)[1] for p, v in F.items()}
fv = {(r["panel"], r["metric"]): r["value"] for r in final if pick.get(r["panel"]) == r["lr"]}
tv = defaultdict(dict)
for r in tune:
    if r["metric"] in ("acc", "rho"): tv[(r["panel"], r["arm"], r["lr"])][r["metric"]] = r["value"]
cmp = []
def c(name, x): cmp.append(dict(item="LIEP-6C", metric=name, value=x))
cv = list(csv.DictReader(open(f"{L}/close/results/val/lockin_adam_lr1e-2_k0.1_s42_val.csv")))
l1_val = last({(int(r["epoch"]), r["metric"]): float(r["value"]) for r in cv}, "val_acc"); c("L1_val", l1_val)
l2 = max(((v["acc"], k) for k, v in tv.items() if k[0] == "L2"), default=None)
if l2:
    c("L2_val_best", l2[0]); c("L2_val_minus_L1_val", l2[0] - l1_val); c("L2_val_best_rho", tv[l2[1]]["rho"])
    c("L2_val_best_lr", l2[1][2]); c("L2_val_best_gamma", float(l2[1][1].split("amp")[1]))
pr = []                                                     # the panel's paper ReLU runs (Adam 1e-3 constant)
for line in open(f"{L}/regen/figures/data/fig6_cmp.dat"):
    t = line.split()
    if len(t) == 3 and t[0] == "relu": pr.append(float(t[2]))
if pr: c("paper_relu_median", med(pr)); c("paper_relu_n", len(pr))
pairs = [("L3", "L1"), ("L2", "L1"), ("L4", "L1"), ("B2", "B1"), ("R2", "R1"), ("B3", "B1"), ("L1", "B1"), ("L2", "B1"),
         ("L4", "B1"), ("R1", "B1"), ("B3", "L1"), ("B3", "L2")]
for a, b in pairs:
    if (a, "test_acc") in fv and (b, "test_acc") in fv: c(f"{a}_minus_{b}", fv[(a, "test_acc")] - fv[(b, "test_acc")])
if pr and ("R1", "test_acc") in fv: c("R1_minus_paper_relu", fv[("R1", "test_acc")] - med(pr))
for p in sorted(pick): c(f"{p}_lr", pick[p])
write("compare.csv", cmp)
for r in tune: print("tune", r["panel"], r["arm"], r["lr"], r["metric"], round(r["value"], 4))
for r in final: print("final", r["panel"], r["lr"], r["metric"], round(r["value"], 4) if isinstance(r["value"], float) else r["value"])
for r in cmp: print("cmp", r["metric"], round(r["value"], 4))
# trajectories: median over the test seeds of rho (and L2's per-step row scale) at each logged epoch
tr = []
for (split, p, arm, lr), by in sorted(runs.items()):
    if split != "test" or p not in ("L1", "L2", "L3", "L4") or lr != pick.get(p): continue
    for m in ("rho", "amp_scale_median"):
        for e in sorted({e for v in by.values() for (e, mm) in v if mm == m}):
            xs = [v[(e, m)] for v in by.values() if (e, m) in v]
            tr.append(dict(item="LIEP-6C", panel=p, metric=m, epoch=e, value=med(xs), n=len(xs)))
write("traj.csv", tr)
for r in tr: print("traj", r["panel"], r["metric"], r["epoch"], round(r["value"], 4), r["n"])
