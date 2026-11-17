"""LIEP-EPOPT summaries (FINDINGS reads these; long format, one number per row).
results/tune.csv: seed-42 held-out accuracy at epoch 8 per (estimator, rule, lr), and the pick per (estimator, rule).
results/final.csv: per (estimator, rule) at the pick: median/min/max over 3 seeds of test accuracy, train accuracy, hidden
and output probes, network − hidden probe, collapses, epoch seconds; the Adam-1e-3-constant control from LIEP-EP30
(epoch 8 of the constant-lr 30-epoch run, GB300) and the BP rows from LIEP-OPTIM (tuned) and LIEP-GAP (paper rule, via LIEP-EP30 summary.csv); differences."""
import csv, glob, os, statistics as st
from collections import defaultdict
D = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(D, "..")
SEEDS = (42, 123, 456)
d = defaultdict(dict)                                   # (split, est, opt, lr, epoch, metric) -> {seed: value}
for split in ("val", "test"):
    for f in glob.glob(f"{D}/results/{split}/*.csv"):
        for r in csv.DictReader(open(f)):
            d[(split, r["model"], r["opt"], float(f"{float(r['lr']):.6g}"), int(r["epoch"]), r["metric"])][int(r["seed"])] = float(r["value"])
med = lambda v: st.median(v)
out_t, out_f = [], []
def t(est, opt, lr, metric, value): out_t.append(dict(item="LIEP-EPOPT", arm=f"{est}_{opt}", model=est, opt=opt, lr=lr, metric=metric, value=value))
def f(arm, metric, value, **kw): out_f.append(dict(item="LIEP-EPOPT", arm=arm, metric=metric, value=value, **kw))
pick = {}
for est in ("lockin", "centred"):
    for opt in ("adam", "muon"):
        lrs = sorted(k[3] for k in d if k[:3] == ("val", est, opt) and k[4:] == (8, "val_acc"))
        for lr in lrs:
            t(est, opt, lr, "val_acc_e8", d[("val", est, opt, lr, 8, "val_acc")][42])
        best = max(lrs, key=lambda lr: d[("val", est, opt, lr, 8, "val_acc")][42])
        pick[(est, opt)] = best
        t(est, opt, best, "pick", 1); t(est, opt, best, "pick_on_edge", int(best in (lrs[0], lrs[-1])))
final = {}
for (est, opt), lr in pick.items():
    acc = d.get(("test", est, opt, lr, 8, "test_acc"), {})
    if len(acc) < 3: continue
    arm = f"{est}_{opt}"; kw = dict(model=est, opt=opt, lr=lr)
    a = [acc[s] for s in SEEDS]; final[arm] = med(a)
    for nm, v in (("test_acc", med(a)), ("test_min", min(a)), ("test_max", max(a)), ("n", len(a))): f(arm, nm, v, **kw)
    for m in ("train_acc", "probe_hidden", "probe_output"):
        f(arm, m, med([d[("test", est, opt, lr, 8, m)][s] for s in SEEDS]), **kw)
    f(arm, "gap_hidden", med([acc[s] - d[("test", est, opt, lr, 8, "probe_hidden")][s] for s in SEEDS]), **kw)
    f(arm, "collapsed", sum(d[("test", est, opt, lr, 8, "collapsed")][s] for s in SEEDS), **kw)
    f(arm, "epoch_seconds", med([d[("test", est, opt, lr, e, "epoch_seconds")][s] for s in SEEDS for e in range(2, 9)]), **kw)
ep30 = {(r["arm"], r["epoch"]): r for r in csv.DictReader(open(f"{L}/ep30/results/summary.csv"))}
for est in ("lockin", "centred"):
    r = ep30[(f"{est}_e30", "8")]
    for nm in ("test_acc", "test_min", "test_max", "probe_hidden", "gap_hidden"):
        f(f"{est}_adam_const", nm, float(r[nm]), model=est, opt="adam_const", lr=1e-3)
    final[f"{est}_adam_const"] = float(r["test_acc"])
bp = {(r["arm"], r["epochs"]): r for r in csv.DictReader(open(f"{L}/optim/results/final_table.csv"))}
for opt in ("adam", "muon", "adam_const"):                # LIEP-OPTIM, each tuned (adam_const: tuned constant lr)
    r = bp[(opt, "8")]; f(f"bp_{opt}", "test_acc", float(r["test_acc"]), model="bp", opt=opt, lr=float(r["lr"]))
    final[f"bp_{opt}"] = float(r["test_acc"])
r = ep30[("bp_cb64_t1", "8")]                               # LIEP-GAP cb64_t1: BP under the paper's rule, Adam 1e-3 constant
f("bp_paper", "test_acc", float(r["test_acc"]), model="bp", opt="adam_const", lr=1e-3); final["bp_paper"] = float(r["test_acc"])
for est in ("lockin", "centred"):
    for a, b, nm in ((f"{est}_muon", f"{est}_adam", "muon_minus_adam"), (f"{est}_adam", f"{est}_adam_const", "adam_minus_const"),
                     (f"{est}_muon", f"{est}_adam_const", "muon_minus_const"), ("bp_muon", f"{est}_muon", "bp_minus_ep_muon"),
                     ("bp_paper", f"{est}_adam_const", "bp_minus_ep_const"), ("bp_adam", f"{est}_adam", "bp_minus_ep_adam")):
        if a in final and b in final: f(f"diff_{est}", nm, final[a] - final[b], model=est)
for name, rows in (("tune.csv", out_t), ("final.csv", out_f)):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{D}/results/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
for r in out_t: print("tune", r["arm"], r["lr"], r["metric"], round(r["value"], 4))
for r in out_f: print("final", r["arm"], r["metric"], round(r["value"], 4) if isinstance(r["value"], float) else r["value"])
