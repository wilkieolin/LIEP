#!/usr/bin/env python3
"""LIEP-REGEN stage 1 aggregation: results/train/*.csv → results/train_summary.csv (long format).

Per (arm, point, optimiser, lr, T_free, epoch): median, min and max over seeds of test accuracy (10k and the first 2k), the number
of seeds, and (final epoch) the number collapsed (10k accuracy < 0.5). arm `lockin64` = harness lock-in with Float64
accumulators (stage 1b); T_free 1000 = stage 1c. Usage: python3 experiments/regen/aggregate.py
"""
import csv, glob, math, os, re, statistics as st
from collections import defaultdict

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
acc = defaultdict(list); coll = defaultdict(list)
for f in sorted(glob.glob(os.path.join(base, "train", "*.csv"))):
    tag = os.path.basename(f)[:-4]
    if tag.endswith("_smoke") or re.search(r"_val\d+$", tag):    # stage-2 tuning runs: stage2_choice.csv, not here
        continue
    tf = int(re.search(r"_tf(\d+)", tag).group(1)) if "_tf" in tag else 200
    opt = "sgd" if "_sgd" in tag else "adam"
    # stage 3d/3e variants keep their own arm name, so they never pool with the plain runs of the same point and rate
    var = "".join(v for v in ("_frozen1", "_dt0.1") if tag.endswith(v) or v + "_" in tag)
    for r in csv.DictReader(open(f)):
        k = (r["arm"] + var, r["model"], opt, f"{float(r['lr']):g}", tf)        # lr as %g: the CSVs hold Float32 values
        if r["metric"] in ("test_acc_10k", "test_acc_2k"):
            acc[k + (int(r["epoch"]), r["metric"])].append(float(r["value"]))
        if r["metric"] == "collapsed":
            coll[k].append(float(r["value"]))
out = []
for (arm, pt, opt, lr, tf, ep, m), vs in sorted(acc.items()):
    for stat, v in (("median", st.median(vs)), ("min", min(vs)), ("max", max(vs)), ("n", len(vs))):
        out.append(dict(item="LIEP-REGEN", arm=arm, model=pt, opt=opt, lr=lr, t_free=tf, epoch=ep, metric=f"{m}_{stat}", value=v))
for (arm, pt, opt, lr, tf), vs in sorted(coll.items()):
    out.append(dict(item="LIEP-REGEN", arm=arm, model=pt, opt=opt, lr=lr, t_free=tf, epoch=8, metric="n_collapsed", value=sum(vs)))
with open(os.path.join(base, "train_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["item", "arm", "model", "opt", "lr", "t_free", "epoch", "metric", "value"])
    w.writeheader(); w.writerows(out)
print(f"{len(out)} rows")

# ---- stage 3a grid → results/grid_summary.csv: per (ε, ω_p, n_cyc, warm-up, dt, metric) median, min and fail rate
# (cos < 0.9) over draws; parameters as short strings (%g) so FINDINGS placeholders can match them exactly
gp = os.path.join(base, "grid.csv")
if os.path.exists(gp):
    g = defaultdict(list); ref = defaultdict(list)
    for r in csv.DictReader(open(gp)):
        if r["arm"] == "lockin" and r["metric"].startswith("cos"):
            k = tuple(f"{float(r[c]):g}" for c in ("eps", "omega_p", "n_cyc", "warm_c", "dt")) + (r["metric"],)
            g[k].append(float(r["value"]))
        elif r["arm"] in ("kappa", "centred_ref", "exact"):
            ref[(r["arm"], r["metric"])].append(float(r["value"]))
    rows = []
    for (e, o, n, w, d, m), vs in sorted(g.items()):
        for stat, v in (("median", st.median(vs)), ("min", min(vs)), ("fail_rate", sum(x < 0.9 for x in vs) / len(vs))):
            rows.append(dict(eps=e, omega_p=o, n_cyc=n, warm_c=w, dt=d, arm="lockin", metric=f"{m}_{stat}", value=v))
    for (a, m), vs in sorted(ref.items()):
        for stat, v in (("median", st.median(vs)), ("min", min(vs)), ("max", max(vs))):
            rows.append(dict(eps="", omega_p="", n_cyc="", warm_c="", dt="", arm=a, metric=f"{m}_{stat}", value=v))
    with open(os.path.join(base, "grid_summary.csv"), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["eps", "omega_p", "n_cyc", "warm_c", "dt", "arm", "metric", "value"])
        w_.writeheader(); w_.writerows(rows)
    print(f"grid: {len(rows)} rows")

# ---- stage 4a Table 2 → results/table2_summary.csv: per (config, pair, layer, parameter, metric) median, min, max
tp = os.path.join(base, "table2.csv")
if os.path.exists(tp):
    t = defaultdict(list)
    for r in csv.DictReader(open(tp)):
        if r["metric"] in ("cos", "relerr"):
            t[(r["model"], r["arm"], r["layer"], r["param"], r["metric"])].append(float(r["value"]))
    rows = []
    for (c, a, l, p, m), vs in sorted(t.items()):
        for stat, v in (("median", st.median(vs)), ("min", min(vs)), ("max", max(vs))):
            rows.append(dict(config=c, pair=a, layer=l, param=p, metric=f"{m}_{stat}", value=v))
    with open(os.path.join(base, "table2_summary.csv"), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["config", "pair", "layer", "param", "metric", "value"])
        w_.writeheader(); w_.writerows(rows)
    print(f"table2: {len(rows)} rows")

# ---- stage 4b recovery → results/recovery_summary.csv: per (damage cell, arm) median, min, max over 3 draws of the
# EP-settle accuracy, and the maximum stuck_drift / layer1_drift over all runs (both must be 0)
rec = sorted(glob.glob(os.path.join(base, "recovery", "*_d[0-9].csv")))
if rec:
    acc = defaultdict(list); drift = defaultdict(float)
    for f in rec:
        for r in csv.DictReader(open(f)):
            if r["metric"] == "acc_ep":
                acc[(r["model"], r["arm"])].append(float(r["value"]))
            elif r["metric"] in ("stuck_drift", "layer1_drift"):
                drift[r["metric"]] = max(drift[r["metric"]], float(r["value"]))
    rows = []
    for (m, a), vs in sorted(acc.items()):
        for stat, v in (("median", st.median(vs)), ("min", min(vs)), ("max", max(vs)), ("n", len(vs))):
            rows.append(dict(cell=m, arm=a, metric=f"acc_ep_{stat}", value=v))
    for k, v in drift.items():
        rows.append(dict(cell="all", arm="all", metric=f"max_{k}", value=v))
    with open(os.path.join(base, "recovery_summary.csv"), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["cell", "arm", "metric", "value"]); w_.writeheader(); w_.writerows(rows)
    print(f"recovery: {len(rows)} rows")

# ---- stages 3c–3e → results/stage3_summary.csv: hidden-layer probes (probe/), the extended grid (grid_ext.csv), the
# reference-phase test (phase_resp.csv) and κ from the Jacobian (kappa_jac.csv). Median, min, max over seeds or draws,
# and for cosines the fraction below 0.9. Parameters as %g strings so FINDINGS placeholders match them exactly.
def g_(x):
    return f"{float(x):.3g}" if x not in ("", None) else ""
S = defaultdict(list)
for f in sorted(glob.glob(os.path.join(base, "probe", "*.csv"))):
    tag = os.path.basename(f)[:-4]
    var = "_dt0.1" if tag.endswith("_dt0.1") else ""
    for r in csv.DictReader(open(f)):
        if r["metric"] == "probe_test_acc":
            S[("probe", r["arm"] + var, r["model"], "", "", "", "", "acc")].append(float(r["value"]))
for r in csv.DictReader(open(os.path.join(base, "grid_ext.csv"))):
    S[("grid_ext", r["arm"], r["model"], g_(r["eps"]), g_(r["omega_p"]), "0.5", "", r["metric"])].append(float(r["value"]))
for r in csv.DictReader(open(os.path.join(base, "phase_resp.csv"))):
    S[("phase", r["arm"], r["model"], "0.03", g_(r["omega_p"]), g_(r["dt"]), r["layer"], r["metric"])].append(float(r["value"]))
for r in csv.DictReader(open(os.path.join(base, "kappa_jac.csv"))):
    S[("kappa_jac", r["arm"], r["model"], "", "", "", "", r["metric"])].append(float(r["value"]))
rows = []
for (stage, a, m, e, o, d, l, met), vs in sorted(S.items()):
    stats = [("median", st.median(vs)), ("min", min(vs)), ("max", max(vs)), ("n", len(vs))]
    if met.startswith("cos"):
        stats.append(("fail_rate", sum(x < 0.9 for x in vs) / len(vs)))
    for stat, v in stats:
        rows.append(dict(stage=stage, arm=a, model=m, eps=e, omega_p=o, dt=d, layer=l, metric=f"{met}_{stat}", value=v))
with open(os.path.join(base, "stage3_summary.csv"), "w", newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
print(f"stage 3: {len(rows)} rows")

# ---- stage 3c–4c single numbers the FINDINGS prose quotes → results/stage3_keys.csv (key, value)
K = {}
def s3(stage, arm, model, metric, eps="", om="", dt="", layer=""):
    m = [r for r in rows if r["stage"] == stage and r["arm"] == arm and r["model"] == model and r["metric"] == metric
         and r["eps"] == eps and r["omega_p"] == om and r["dt"] == dt and r["layer"] == layer]
    assert len(m) == 1, (stage, arm, model, metric, eps, om, dt, layer, len(m)); return m[0]["value"]
# prediction 1: the pre-fix arm against the submitted map (layer-1 cos against centred EP, median over draws), 30 cells
leg = [l.split() for l in open(os.path.join(base, "..", "..", "common", "manuscript_figures", "figures", "data", "fig1_map.dat")) if not l.startswith("#")][1:]
dif = [abs(s3("grid_ext", "prefix", "synthetic", "cos_l1_vs_centred_median", g_(e), g_(o), "0.5") - float(c)) for _, e, c, o in leg]
K.update(legacy_cells=len(dif), legacy_maxdiff=max(dif), legacy_meddiff=st.median(dif))
for arm in ("prefix", "prefix_nodc", "corrected"):
    K[f"{arm}.synthetic.cos_l1.0.1"] = s3("grid_ext", arm, "synthetic", "cos_l1_vs_exact_median", "0.03", "0.1", "0.5")
for om in ("0.5", "1", "2"):
    for m in ("synthetic", "real_init", "real_trained"):
        K[f"corrected.{m}.cos_l1.{om}"] = s3("grid_ext", "corrected", m, "cos_l1_vs_exact_median", "0.03", om, "0.5")
for m in ("synthetic", "real_init", "real_trained"):
    K[f"kappa_tail.{m}"] = s3("grid_ext", "kappa", m, "kappa_tail_median", "0", "0", "0.5")
K["kappa_legacy_window.synthetic"] = s3("grid_ext", "kappa", "synthetic", "kappa_legacy_window_median", "0", "0", "0.5")
K["kappa_ratio_trained_init"] = K["kappa_tail.real_trained"] / K["kappa_tail.real_init"]
# predictions 11, 12: the reference-phase test, layer 1 unless named
for m in ("real_init", "real_trained"):
    for om in ("0.5", "1", "2"):
        K[f"phase.{m}.best.{om}"] = s3("phase", "phase_resp", m, "cos_best_phase_median", "0.03", om, "0.5", "1")
for om in ("0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"):
    for l in ("1", "2"):
        K[f"phase.lag_deg.l{l}.{om}"] = math.degrees(s3("phase", "phase_resp", "real_init", "lag_rad_median", "0.03", om, "0.5", l))
    K[f"phase.phase0.dt0.1.{om}"] = s3("phase", "phase_resp", "real_init", "cos_phase0_median", "0.03", om, "0.1", "1")
    K[f"phase.phase0.dt0.5.{om}"] = s3("phase", "phase_resp", "real_init", "cos_phase0_median", "0.03", om, "0.5", "1")
oms = ["0.02", "0.05", "0.1", "0.2", "0.5", "1"]
K["phase.lag_ratio_median"] = st.median([K[f"phase.lag_deg.l1.{o}"] / K[f"phase.lag_deg.l2.{o}"] for o in oms])
K["phase.l1_90deg_first"] = float(next(o for o in oms + ["2"] if K[f"phase.lag_deg.l1.{o}"] >= 90))
K["phase.max_shift_dt_le0.5"] = max(abs(K[f"phase.phase0.dt0.1.{o}"] - K[f"phase.phase0.dt0.5.{o}"]) for o in ("0.02", "0.05", "0.1", "0.2", "0.5"))
with open(os.path.join(base, "stage3_keys.csv"), "w", newline="") as fh:
    w_ = csv.writer(fh); w_.writerow(["key", "value"]); w_.writerows(K.items())
print(f"stage 3 keys: {len(K)}")
