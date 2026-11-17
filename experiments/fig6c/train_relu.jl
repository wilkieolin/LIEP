# LIEP-6C arms R1/R2: the ReLU MLP of Fig 6c (784→256→64→10, softmax CE, standardised pixels, Lux default init, as
# LIEP-BASE `relu 256-64`) under Adam + cosine and Muon, with the same step code as the phasor arms
# (close/train_close.jl, optim/train_optim.jl). One configuration × one seed, CPU:
#   julia -t 4 --project=. experiments/fig6c/train_relu.jl \
#     <adam|muon> <lr> <seed> <val|test>
# adam: Adam β (0.9, 0.999), ε 1e-8, bias-corrected, lr × ½(1 + cos π(t−1)/T) on every parameter.
# muon: weights m = 0.95 m + g, θ −= lr·cos·max(1, rows/cols)^½·NS5(g + 0.95 m); biases Adam 1e-3·cos.
# Batch 256, 8 epochs (FIG6C_EPOCHS). `val`: first 50k training images, score the last 10k; `test`: 60k/10k.
# Writes results/relu/<split>/<tag>.csv (long format) + provenance.
include(joinpath(@__DIR__, "..", "base", "base_lib.jl"))
using Random, Statistics, LinearAlgebra
BLAS.set_num_threads(parse(Int, get(ENV, "FIG6C_BLAS", "4")))
const OPT = ARGS[1]; const LR = parse(Float32, ARGS[2]); const SEED = parse(Int, ARGS[3]); const SPLIT = ARGS[4]
OPT in ("adam", "muon") || error("rule $OPT"); SPLIT in ("val", "test") || error("split $SPLIT")
const EPOCHS = parse(Int, get(ENV, "FIG6C_EPOCHS", "8"))
const TAG = "relu_$(OPT)_lr$(ARGS[2])_s$(SEED)_$(SPLIT)" * (EPOCHS != 8 ? "_e$EPOCHS" : "")
const R = joinpath(@__DIR__, "results", "relu"); mkpath(joinpath(R, SPLIT))

Ptr, ytr_all = pixels(:train); Pte, yte = pixels(:test)
if SPLIT == "val"
    Xtr_raw, ytr, Xev_raw, yev = Ptr[:, 1:50_000], ytr_all[1:50_000], Ptr[:, 50_001:end], ytr_all[50_001:end]
else
    Xtr_raw, ytr, Xev_raw, yev = Ptr, ytr_all, Pte, yte
end
Xtr, Xev = standardise(Xtr_raw, Xev_raw); Y = onehot(ytr)
m, ps, st = build_relu([784, 256, 64, 10], Xoshiro(SEED))
predict(p, X) = (L = first(m(X, p, st)); [argmax(view(L, :, j)) for j in 1:size(L, 2)])   # one forward pass
acc(p) = mean(predict(p, Xev) .== yev)
loss(p, x, y) = logsoftmax_ce(first(m(x, p, st)), y)

function ns5(G)
    a, b, c = 3.4445f0, -4.7750f0, 2.0315f0
    X = G ./ (sqrt(sum(abs2, G)) + 1f-7); tr = size(X, 1) > size(X, 2); tr && (X = permutedims(X))
    for _ in 1:5; A = X * transpose(X); X = a .* X .+ (b .* A .+ c .* (A * A)) * X; end
    tr ? permutedims(X) : X
end
const S = Dict{Any, Any}(); st0(k, x) = get!(() -> zero(x), S, k)
function step!(p, g, t, sch)
    β1, β2 = 0.9f0, 0.999f0; c1, c2 = 1 - β1^t, 1 - β2^t
    for l in keys(p), f in keys(getfield(p, l))
        x = getfield(getfield(g, l), f); θ = getfield(getfield(p, l), f)
        if OPT == "muon" && f == :weight
            mk = st0((:mw, l), x); mk .= 0.95f0 .* mk .+ x
            θ .-= (LR * sch * sqrt(max(1f0, size(θ, 1) / size(θ, 2)))) .* ns5(x .+ 0.95f0 .* mk)
        else
            η = (OPT == "muon" ? 1f-3 : LR) * sch
            mk = st0((:m, l, f), x); vk = st0((:v, l, f), x)
            mk .= β1 .* mk .+ (1 - β1) .* x; vk .= β2 .* vk .+ (1 - β2) .* x .^ 2
            θ .-= η .* (mk ./ c1) ./ (sqrt.(vk ./ c2) .+ 1f-8)
        end
    end
end

rows = NamedTuple[]
const AM = SPLIT == "val" ? "val_acc" : "test_acc"
row!(metric, v; epoch = -1) = push!(rows, (item = "LIEP-6C", arm = "relu_$OPT", model = "relu_784-256-64-10", opt = OPT, lr = Float64(LR),
                                           kappa = "", seed = SEED, split = SPLIT, epochs = EPOCHS, epoch = epoch, metric = metric, value = Float64(v)))
t0 = time(); row!(AM, acc(ps); epoch = 0)
n = length(ytr); nb = cld(n, 256); T = EPOCHS * nb; idx = collect(1:n); t = 0
for ep in 1:EPOCHS
    global t
    shuffle!(Xoshiro(SEED + 7919ep), idx)
    for i in 1:256:n
        t += 1; sch = 0.5f0 * (1 + cos(Float32(π) * (t - 1) / T)); b = idx[i:min(i + 255, n)]
        g = Zygote.gradient(p -> loss(p, Xtr[:, b], Y[:, b]), ps)[1]
        step!(ps, g, t, sch)
    end
    a = acc(ps); row!(AM, a; epoch = ep); @printf("%s epoch %d %s %.4f\n", TAG, ep, AM, a); flush(stdout)
end
row!("train_acc", mean(predict(ps, Xtr) .== ytr); epoch = EPOCHS)
row!("collapsed", rows[findlast(r -> r.metric == AM, rows)].value < 0.5 ? 1 : 0; epoch = EPOCHS)
row!("total_seconds", time() - t0)
CSV.write(joinpath(R, SPLIT, "$TAG.csv"), DataFrame(rows))
write_provenance(joinpath(R, SPLIT, TAG); seeds = [SEED], config = Dict("rule" => OPT, "lr" => LR, "epochs" => EPOCHS, "split" => SPLIT,
                 "batch" => 256, "arch" => "784-256-64-10", "muon_bias_lr" => 1f-3, "host" => gethostname()))
println("DONE $TAG ($(round(Int, time() - t0)) s)")
