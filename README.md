# LIEP — reproduction code and data for *Lock-in EP: An In-Situ Training Algorithm for Oscillatory Hardware*

This repository rebuilds the paper's data figures, **Fig 6** (gradient fidelity and training from scratch),
**Fig 7** (recovery from analog weight damage) and the **Fig 6c appendix table**, from the shipped per-run results.
It can also re-run any of the runs behind them. The model code is
[PhasorNetworks.jl](https://github.com/wilkieolin/PhasorNetworks.jl), pinned as a submodule at `903cf06`, the commit
every run used.

## Quick start

```bash
git clone --recursive https://github.com/wilkieolin/LIEP.git && cd LIEP
make image            # Docker image: Julia 1.12.7 + pinned environment + FashionMNIST + figure toolchain (~20 min)
make docker-figures   # tier 1: rebuild every summary, the figure data and Figs 6, 7 (CPU, a few minutes)
```

The figures land in `experiments/regen/figures/` (`fig6_basin.pdf/.svg`, `fig7_recovery.pdf/.svg`), next to their
LaTeX captions (`*.tex`; the numbers in them are macros from `figures/data/*_macros.tex`). The last step compares
every regenerated table with the shipped one. It should report that all of them reproduce, and 0 Type 3 fonts in the PDFs.

## Three tiers

| Tier | What | Hardware | Time |
|---|---|---|---|
| 1 `make figures` | Per-run CSVs → summaries → figure tables → Figs 6, 7 and the appendix table; compares with the shipped versions | CPU | minutes |
| 2 `make smoke` | One cheap run per family: the Fig 6a grid, the logistic-regression controls, one Fig 7 damage cell | 1 NVIDIA GPU (Fig 7) | ~1 h |
| 3 `scripts/run_jobs.sh jobs/<file>` | Every run behind a figure, as recorded | NVIDIA GPUs | see below |

Each job file in `jobs/` lists the exact commands, one per line, generated from the shipped runs' provenance and
checked against each script's output naming. A re-run overwrites its shipped CSV in place. `git diff --stat
experiments` then shows what moved, and `make figures` redraws from the new numbers. GPU training is not bitwise
deterministic, so expect seed-level agreement, not identical digits.

| Job file | Runs | Recorded run time, summed |
|---|---|---|
| `fig6a.txt` | 1 (grid over ε, ω_p; κ) | ~20 min |
| `fig6b_1_train.txt`, then `fig6b_2_probe.txt` | 33 training + 30 probe | ~22 h + probes (CPU) |
| `fig6c.txt` | 24 training + 2 logistic regression | ~8 h |
| `fig6c_appendix.txt` | 51 | ~41 h |
| `fig7_1_pretrain.txt`, then `fig7_2_damage.txt` | 1 + 256 (32 cells × 8 draws) | ~59 h |
| `tuning.txt` (optional) | 78 held-out runs that chose the learning rates | ~32 h |

Times were recorded on NVIDIA GB10 and GH200 GPUs with 2–3 runs sharing a GPU. The runs are independent, so
`-P` sets how many run at once:

```bash
D="docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/opt/liep -w /opt/liep liep"
$D scripts/run_jobs.sh -P 3 jobs/fig7_2_damage.txt
$D scripts/run_jobs.sh -g "gaussian 24" jobs/fig7_2_damage.txt
```

The `-g` filter picks lines matching a pattern; `-n` takes the first N lines.

**Checked before release (2026-11-17, image built on an NVIDIA GB10, arm64):**
- **Tier 1:** all 24 regenerated tables match the shipped ones; Fig 6 renders pixel-identical. Fig 7 differs only
  in subscript and arrow placement in the top-row labels: same fonts, a different Chromium build.
- **Logistic-regression control:** all 9 values identical.
- **Fig 6a grid:** 10,640 cosines within 4e-5.
- **Fig 7 cell (Gaussian 7×, draw 1, GPU):** every arm's accuracy identical to the shipped run.

## What each figure rests on

| Figure | Runs (folder) | Script |
|---|---|---|
| 6a fidelity vs ω_p | `regen/results/grid_ext` | `regen/grid_ext.jl` |
| 6b training at each ω_p, hidden-layer gain | `regen/results/train`, `regen/results/probe`, `base/results/probe` | `regen/train_fm.jl`, `regen/probe_points.jl`, `base/run.jl probe` |
| 6c networks from scratch, probes | `fig6c/results/{relu,bp,ep}/test` | `fig6c/train_relu.jl`, `optim/train_optim.jl`, `close/train_close.jl` |
| 6c linear controls | `base/results/lr` | `base/run.jl lr` |
| 6c appendix table | `fig6c`, `close`, `epopt` results; rows from `optim` and `ep30` ship as data | `close/train_close.jl`, `epopt/train_epopt.jl` |
| 7 recovery | `regen/results/recovery` | `regen/recovery.jl` |

Shared code: `experiments/common/liep_lib.jl` (the paper's network, input encoding and settle),
`experiments/base/base_lib.jl`, `experiments/common/provenance.jl`. Summaries: `*/aggregate.py`, `*/summarize.py`.
Figure pipeline: `regen/figures/export.py` (tables and macros) and `regen/figures/src/*.py` (Plotly renderers).

`experiments/provenance.csv` has one row per run: script, arguments, configuration, library commit, host and time.
`RELEASE.toml` gives each code file's source and sha256 in the development tree it was exported from. The only
changes from that tree are the flattened include paths, marked `rewritten = true`.

## Setup without Docker

- **Julia 1.12.7:** `git submodule update --init`, then
  `julia --project=. -e 'using Pkg; Pkg.instantiate()'`. FashionMNIST downloads on first use.
- **Python 3.12:** `pip install -r requirements.txt`. kaleido needs a Chrome or Chromium; point `BROWSER_PATH` at it.
  The renderers use Liberation Sans, since TrueType keeps Type 3 fonts out of the PDFs.

## Apptainer (HPC)

```bash
make sif                                                         # docker save + apptainer build
apptainer run --nv --bind $PWD:/opt/liep liep.sif scripts/run_jobs.sh -P 3 jobs/fig7_2_damage.txt
```

The entry point keeps a writable Julia depot in `~/.julia-liep` in front of the image's read-only one. Set
`JULIA_USER_DEPOT` to put it elsewhere.

## License

MIT, the same terms as PhasorNetworks.jl (UChicago Argonne, LLC); see `LICENSE`.
