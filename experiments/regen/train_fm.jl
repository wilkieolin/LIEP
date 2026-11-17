# LIEP-REGEN stage 1: Fig 6c,d training with the corrected estimator (library main ep_gradient) on the GPU.
# Usage (repo root, spark):
#   JULIA_CUDA_HARD_MEMORY_LIMIT=8GiB julia --project=. \
#     experiments/regen/train_fm.jl <lockin|centred> <A|B|C|D|E> <lr> <seed>
# Writes results/train/<tag>.csv (test accuracy on 10k and on the first 2k at epochs 0/1/4/8, train loss per epoch,
# seconds) and results/ckpt/<tag>_e<k>.jls (weights at those epochs, CPU arrays; DVC). Settings: README.md.
using CUDA, LuxCUDA
include(joinpath(@__DIR__, "..", "common", "liep_lib.jl"))
include(joinpath(@__DIR__, "..", "common", "provenance.jl"))
using CSV, DataFrames, Optimisers, Printf, Serialization

const ARM = ARGS[1]; const POINT = ARGS[2]; const LR = parse(Float32, ARGS[3]); const SEED = parse(Int, ARGS[4])
const NTRAIN = parse(Int, get(ENV, "REGEN_NTRAIN", "60000")); const EPOCHS = parse(Int, get(ENV, "REGEN_EPOCHS", "8"))
const POINTS = Dict("A" => (0.03f0, 0.01f0), "B" => (0.03f0, 0.02f0), "C" => (0.03f0, 0.05f0), "D" => (0.03f0, 0.10f0),
                    "E" => (0.003f0, 0.01f0), "F" => (0.03f0, 0.20f0),       # F: stage 3b, the grid's worst ω_p
                    "G" => (0.03f0, 0.5f0), "H" => (0.03f0, 1f0), "I" => (0.03f0, 2f0))   # stage 3c: past the old grid
const OPT = get(ENV, "REGEN_OPT", "adam")                            # stage 2: adam | sgd (Optimisers.Descent)
const VAL = get(ENV, "REGEN_VAL", "0") == "1"                         # stage 2 tuning: train on 50k, score on the held-out 10k
const ACC = VAL ? "val" : "test"                                      # metric prefix: val_acc_10k is the held-out 10k
const TFREE = parse(Int, get(ENV, "REGEN_TFREE", "200"))          # stage 1c: a longer, settled free phase
const DT = parse(Float32, get(ENV, "REGEN_DT", "0.5"))               # stage 3e: the integrator step during the gradient (T_free in time kept); inference stays at 0.5
const FREEZE = get(ENV, "REGEN_FREEZE", "0") == "1"                  # stage 3d: layer 1 stays at its random init, only layer 2 trains
const TAG = "$(ARM)_$(POINT)_lr$(ARGS[3])_s$(SEED)" * (FREEZE ? "_frozen1" : "") * (DT != 0.5f0 ? "_dt$(DT)" : "") * (TFREE != 200 ? "_tf$(TFREE)" : "") * (OPT != "adam" ? "_$(OPT)" : "") *
            (VAL ? "_val$(EPOCHS)" : "") * (NTRAIN < 60000 ? "_smoke" : "")
const OUT = joinpath(@__DIR__, "results"); mkpath(joinpath(OUT, "train")); mkpath(joinpath(OUT, "ckpt"))

ε, ω = POINTS[POINT]
est = ARM == "lockin" ? LockinEP(ε = ε, ω_p = ω, n_cycles = 4, T_warmup_cycles = 1, T_free = round(Int, TFREE * 0.5f0 / DT), dt = DT) :
      ARM == "centred" ? StaticEP(β = 0.1f0, T_free = 200, T_nudge = 100, dt = 0.5f0, centered = true) :
      ARM in ("lockin64", "lockin_prefix") ? Probe(ε = ε, ω_p = ω, T_free = TFREE) : error(ARM)   # harness; Float64 sums / the pre-fix leak (stage 3c)
Xtr, ytr = fmnist(:train, NTRAIN); Xte, yte = fmnist(:test)
if VAL                                                               # held-out validation: the last 10k training images
    Xte, yte = Xtr[:, 50_001:end], ytr[50_001:end]; Xtr, ytr = Xtr[:, 1:50_000], ytr[1:50_000]
end
chain, ps, st = paper_net(SEED); codes = paper_codes(SEED)
gdev = gpu_device(); cpu = cpu_device()
psd, std_ = gdev(ps), gdev(st); codes_d = cu(codes)
todev(X) = cu(ComplexF32.(PN._phase_input_to_complex(X)))
const XTE_D = todev(Xte)

function evaluate(psd)
    pred = Int[]
    for i in 1:2000:length(yte)
        e = min(i + 1999, length(yte))
        logits = Array(ep_predict(chain, psd, std_, XTE_D[:, i:e], codes_d; T = 200, dt = 0.5f0, K_mode = :zero))
        append!(pred, [argmax(view(logits, :, b))[1] for b in 1:size(logits, 2)])
    end
    return mean(pred .== yte), mean(pred[1:2000] .== yte[1:2000])
end

rows = NamedTuple[]
row!(ep, metric, v) = push!(rows, (item = "LIEP-REGEN", arm = ARM, model = POINT, seed = SEED, lr = Float64(LR),
                                   epoch = ep, metric = metric, value = Float64(v)))
save_ckpt(ep) = serialize(joinpath(OUT, "ckpt", "$(TAG)_e$(ep).jls"), cpu(psd))
a10, a2 = evaluate(psd); row!(0, ACC * "_acc_10k", a10); row!(0, ACC * "_acc_2k", a2); save_ckpt(0)
@printf("%s epoch 0  10k %.4f  2k %.4f\n", TAG, a10, a2); flush(stdout)
opt = Optimisers.setup(OPT == "sgd" ? Optimisers.Descent(LR) : Optimisers.Adam(LR), psd); rng = Xoshiro(SEED)
t0 = time()
for ep in 1:EPOCHS
    perm = randperm(rng, length(ytr)); te = time(); lsum = 0.0; nb = 0
    for k in 1:(length(ytr) ÷ 256)
        idx = perm[(k - 1) * 256 .+ (1:256)]
        g = ARM == "lockin_prefix" ?
            gdev(est_lockin_prefix(cpu(psd), Xtr[:, idx], CodebookCost(codes_d, ytr[idx]), est; to = cu)) :
            ARM == "lockin64" ?
            gdev(est_lockin(cpu(psd), Xtr[:, idx], CodebookCost(codes_d, ytr[idx]), est; to = cu)) :
            first(ep_gradient(est, chain, psd, std_, todev(Xtr[:, idx]), CodebookCost(codes_d, ytr[idx])))
        FREEZE && (g = merge(g, (layer_1 = NamedTuple(q => (g.layer_1[q] === nothing ? nothing : zero(g.layer_1[q])) for q in keys(g.layer_1)),)))
        global opt, psd = Optimisers.update(opt, psd, g)
        nb += 1
    end
    row!(ep, "epoch_seconds", time() - te)
    if ep in (1, 4, 8) || ep == EPOCHS
        a10, a2 = evaluate(psd); row!(ep, ACC * "_acc_10k", a10); row!(ep, ACC * "_acc_2k", a2); save_ckpt(ep)
        @printf("%s epoch %d  10k %.4f  2k %.4f  (%.0f s)\n", TAG, ep, a10, a2, time() - te); flush(stdout)
    else
        @printf("%s epoch %d  (%.0f s)\n", TAG, ep, time() - te); flush(stdout)
    end
end
row!(EPOCHS, "collapsed", rows[end - 1].value < 0.5 ? 1 : 0)
row!(EPOCHS, "total_seconds", time() - t0)
CSV.write(joinpath(OUT, "train", "$TAG.csv"), DataFrame(rows))
write_provenance(joinpath(OUT, "train", TAG); seeds = [SEED],
                 config = Dict("arm" => ARM, "point" => POINT, "lr" => LR, "epsilon" => ε, "omega_p" => ω, "T_free" => TFREE, "frozen_layer1" => FREEZE, "dt" => DT,
                               "optimiser" => OPT, "validation_split" => VAL,
                               "n_train" => NTRAIN, "epochs" => EPOCHS, "estimator" => string(est)))
println("DONE $TAG")
