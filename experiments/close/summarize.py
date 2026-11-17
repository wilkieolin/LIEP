"""LIEP-CLOSE summaries (long format; FINDINGS reads these).
results/train_tune.csv: per held-out run (seed 42): accuracy at the last epoch (settled and feedforward), feedback ratio ρ.
results/train_final.csv: per (estimator, rule, lr, κ, epochs) with 3 test seeds: median/min/max test accuracy (settled),
feedforward accuracy, train accuracy, hidden probe, ρ, collapses; and differences to BP Muon (LIEP-OPTIM, 0.8896 at 8
epochs) and to lock-in Muon in the standard model (LIEP-EPOPT).
results/diag_summary.csv: stage D medians over the 2 batches per (state, estimator, layer, metric).
results/xscore_summary.csv, results/fbscale_summary.csv: medians over 3 seeds."""
import csv, glob, os, statistics as st
from collections import defaultdict
D = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(D, "..")
med = st.median
def write(name, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{D}/results/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
runs = defaultdict(dict)                                    # (split, est, opt, lr, kappa, epochs) -> seed -> {(epoch, metric): v}
for split in ("val", "test"):
    for f in glob.glob(f"{D}/results/{split}/*.csv"):
        rs = list(csv.DictReader(open(f)))
        if not rs: continue
        r0 = rs[0]; key = (split, r0["model"], r0["opt"], float(f"{float(r0['lr']):.6g}"), r0["kappa"], int(r0["epochs"]))
        runs[key][int(r0["seed"])] = {(int(r["epoch"]), r["metric"]): float(r["value"]) for r in rs}
tune, final = [], []
for (split, est, opt, lr, kap, eps), by in sorted(runs.items()):
    arm = f"{est}_{opt}_k{kap}"; base = dict(item="LIEP-CLOSE", arm=arm, model=est, opt=opt, lr=lr, kappa=kap, epochs=eps)
    am = "val_acc" if split == "val" else "test_acc"
    last = lambda v, m: v.get((eps, m), float("nan"))
    if split == "val":
        for s, v in by.items():
            for m, x in (("acc", last(v, am)), ("acc_ff", last(v, am + "_ff")), ("rho", last(v, "rho")), ("collapsed", v.get((eps, "collapsed"), 0))):
                tune.append(dict(base, seed=s, metric=m, value=x))
    else:
        if len(by) < 3: continue
        a = [last(v, am) for v in by.values()]
        for m, x in (("test_acc", med(a)), ("test_min", min(a)), ("test_max", max(a)), ("n", len(a)),
                     ("test_acc_ff", med([last(v, am + "_ff") for v in by.values()])), ("train_acc", med([last(v, "train_acc") for v in by.values()])),
                     ("probe_hidden", med([last(v, "probe_hidden") for v in by.values()])), ("rho", med([last(v, "rho") for v in by.values()])),
                     ("gap_hidden", med([last(v, am) - last(v, "probe_hidden") for v in by.values()])),
                     ("collapsed", sum(v.get((eps, "collapsed"), 0) for v in by.values())),
                     ("epoch_seconds", med([v[(e, "epoch_seconds")] for v in by.values() for e in range(2, eps + 1) if (e, "epoch_seconds") in v]))):
            final.append(dict(base, metric=m, value=x))
opt8 = {(r["arm"], r["epochs"]): float(r["test_acc"]) for r in csv.DictReader(open(f"{L}/optim/results/final_table.csv")) if r["arm"] != "spread"}
ep = {(r["arm"], r["metric"]): float(r["value"]) for r in csv.DictReader(open(f"{L}/epopt/results/final.csv"))}
for r in [r for r in final if r["metric"] == "test_acc"]:
    bp = opt8.get(("muon", str(r["epochs"])))
    if bp is not None: final.append(dict({k: r[k] for k in r if k not in ("metric", "value")}, metric="bp_muon_minus", value=bp - r["value"]))
    if r["epochs"] == 8: final.append(dict({k: r[k] for k in r if k not in ("metric", "value")}, metric="minus_epopt_lockin_muon", value=r["value"] - ep[("lockin_muon", "test_acc")]))
write("train_tune.csv", tune); write("train_final.csv", final)
dd = defaultdict(list)
for r in csv.DictReader(open(f"{D}/results/diag.csv")):
    dd[(r["model"], r["arm"], r["layer"], r["metric"])].append(float(r["value"]))
write("diag_summary.csv", [dict(item="LIEP-CLOSE", model=k[0], arm=k[1], layer=int(k[2]), metric=k[3], value=med(v), n=len(v)) for k, v in sorted(dd.items())])
xs = defaultdict(list)
for r in csv.DictReader(open(f"{D}/results/xscore.csv")):
    xs[(r["arm"], r["metric"])].append(float(r["value"]))
write("xscore_summary.csv", [dict(item="LIEP-CLOSE", arm=a, metric=m + s, value=f(v), n=len(v)) for (a, m), v in sorted(xs.items())
                             for s, f in (("", med), ("_min", min), ("_max", max))])
fs = defaultdict(list)
for r in csv.DictReader(open(f"{D}/results/fbscale.csv")):
    fs[(r["arm"], round(float(r["scale"]), 3), r["metric"])].append(float(r["value"]))
write("fbscale_summary.csv", [dict(item="LIEP-CLOSE", arm=a, scale=s, metric=m + t, value=f(v), n=len(v)) for (a, s, m), v in sorted(fs.items())
                              for t, f in (("", med), ("_min", min), ("_max", max))])
for r in tune: print("tune", r["arm"], r["lr"], r["epochs"], r["metric"], round(r["value"], 4))
for r in final: print("final", r["arm"], r["lr"], r["epochs"], r["metric"], round(r["value"], 4) if isinstance(r["value"], float) else r["value"])
nk = defaultdict(list)
if os.path.exists(f"{D}/results/noise_kappa.csv"):
    for r in csv.DictReader(open(f"{D}/results/noise_kappa.csv")):
        nk[(r["model"], round(float(r["scale"]), 3), float(f"{float(r['sigma']):.1g}"), round(float(r["eps"]), 3), int(r["layer"]))].append(float(r["value"]))
    write("noise_summary.csv", [dict(item="LIEP-CLOSE", model=k[0], scale=k[1], sigma=k[2], eps=k[3], layer=k[4], metric="cos_vs_exact", value=med(v), n=len(v))
                                for k, v in sorted(nk.items())])
