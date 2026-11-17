# LIEP-EPOPT: LIEP-OPTIM's winners (Muon on the weights + Adam on the biases; Adam) on EP, 8 epochs, cosine decay.
#   julia -t 2 --project=. experiments/epopt/train_epopt.jl \
#     <lockin|centred> <adam|muon|adam_const> <lr> <seed> <val|test>
# Estimators, network, codes, cost, batch and batch order as LIEP-EP30 (point B lock-in or centred static EP, β 0.1).
# Rules (step! below, on weight, bias_real, bias_imag of each layer; same code as LIEP-OPTIM train_optim.jl):
#   adam        Adam β (0.9, 0.999), ε 1e-8, bias-corrected, lr·½(1 + cos π(t−1)/T) per step
#   adam_const  … at constant lr (the paper's rule; control)
#   muon        weights: m = 0.95 m + g, u = g + 0.95 m, θ −= η·cos(t)·max(1, rows/cols)^½·NS5(u);
#               biases: Adam at 1e-3·cos(t)
# `val`: train on the first 50k training images, score ep_predict on the last 10k (tuning). `test`: 60k / 10k test, and
# at epoch 8 train accuracy and LIEP-BASE's hidden/output probes (as LIEP-EP30). Writes results/<split>/<tag>.csv + provenance.
using CUDA, LuxCUDA
include(joinpath(@__DIR__, "..", "base", "base_lib.jl"))     # liep_lib, provenance, softmax_regression, standardise
BLAS.set_num_threads(parse(Int, get(ENV, "EPOPT_BLAS", "6")))
const KIND = ARGS[1]; const OPT = ARGS[2]; const LR = parse(Float32, ARGS[3]); const SEED = parse(Int, ARGS[4]); const SPLIT = ARGS[5]
KIND in ("lockin", "centred") || error("estimator $KIND"); OPT in ("adam", "adam_const", "muon") || error("rule $OPT")
SPLIT in ("val", "test") || error("split $SPLIT")
const EPOCHS = parse(Int, get(ENV, "EPOPT_EPOCHS", "8")); const NTRAIN = parse(Int, get(ENV, "EPOPT_NTRAIN", "60000"))
const TAG = "$(KIND)_$(OPT)_lr$(ARGS[3])_s$(SEED)_$(SPLIT)" * (NTRAIN < 60000 ? "_smoke" : "")
const R = joinpath(@__DIR__, "results"); mkpath(joinpath(R, SPLIT))

est = KIND == "lockin" ? LockinEP(ε = 0.03f0, ω_p = 0.02f0, n_cycles = 4, T_warmup_cycles = 1, T_free = 200, dt = 0.5f0) :
                         StaticEP(β = 0.1f0, T_free = 200, T_nudge = 100, dt = 0.5f0, centered = true)
Xall, yall = fmnist(:train)
if SPLIT == "val"
    Xtr, ytr, Xev, yev = Xall[:, 1:50_000], yall[1:50_000], Xall[:, 50_001:end], yall[50_001:end]
else
    (Xtr, ytr) = (Xall, yall); Xev, yev = fmnist(:test)
end
Xtr, ytr = Xtr[:, 1:min(NTRAIN, end)], ytr[1:min(NTRAIN, end)]
chain, ps, st = paper_net(SEED); codes = paper_codes(SEED)
gdev = gpu_device(); cpu = cpu_device()
psd, std_ = gdev(ps), gdev(st); codes_d = cu(codes)
todev(X) = cu(ComplexF32.(PN._phase_input_to_complex(X)))
const XEV_D = todev(Xev)

"Settled free state (T 200, dt 0.5, K_mode :zero) in chunks of 2000: hidden and output states on the host."
function settled(psd, X)
    H = Matrix{ComplexF32}(undef, 256, size(X, 2)); O = Matrix{ComplexF32}(undef, 64, size(X, 2))
    for i in 1:2000:size(X, 2)
        j = i:min(i + 1999, size(X, 2))
        s = phasor_settle(chain, psd, std_, X isa CuArray ? X[:, j] : todev(X[:, j]), PN.NullCost(), 0f0; T = 200, dt = 0.5f0, K_mode = :zero)
        H[:, j] = Array(s[1]); O[:, j] = Array(s[end])
    end
    H, O
end
acc_of(O, y) = mean([argmax(view(real.(adjoint(codes) * O), :, b)) for b in 1:size(O, 2)] .== y)
"Accuracy on the evaluation split through the library's ep_predict (the number LIEP-REGEN/EP30 report)."
function evaluate(psd)
    pred = Int[]
    for i in 1:2000:length(yev)
        L = Array(ep_predict(chain, psd, std_, XEV_D[:, i:min(i + 1999, end)], codes_d; T = 200, dt = 0.5f0, K_mode = :zero))
        append!(pred, [argmax(view(L, :, b))[1] for b in 1:size(L, 2)])
    end
    mean(pred .== yev)
end

const LAYERS = (:layer_1, :layer_2); const FIELDS = (:weight, :bias_real, :bias_imag)
leaf(p, l, f) = getfield(getfield(p, l), f)
function ns5(G)
    a, b, c = 3.4445f0, -4.7750f0, 2.0315f0
    X = G ./ (sqrt(sum(abs2, G)) + 1f-7); tr = size(X, 1) > size(X, 2); tr && (X = permutedims(X))
    for _ in 1:5
        A = X * transpose(X); X = a .* X .+ (b .* A .+ c .* (A * A)) * X
    end
    tr ? permutedims(X) : X
end
const S = Dict{Any, Any}()
st0(k, x) = get!(() -> zero(x), S, k)
"One step at step t; sch is the cosine factor (1 for adam_const)."
function step!(p, g, t, sch)
    β1, β2 = 0.9f0, 0.999f0; c1, c2 = 1 - β1^t, 1 - β2^t
    for l in LAYERS
        gl = getfield(g, l)
        fs = OPT == "muon" ? (:bias_real, :bias_imag) : FIELDS
        ηa = OPT == "muon" ? 1f-3 * sch : LR * sch
        for f in fs
            x = getfield(gl, f); mk = st0((:m, l, f), x); vk = st0((:v, l, f), x)
            mk .= β1 .* mk .+ (1 - β1) .* x; vk .= β2 .* vk .+ (1 - β2) .* x .^ 2
            leaf(p, l, f) .-= ηa .* (mk ./ c1) ./ (sqrt.(vk ./ c2) .+ 1f-8)
        end
        if OPT == "muon"
            x = getfield(gl, :weight); mk = st0((:mw, l), x); mk .= 0.95f0 .* mk .+ x
            W = leaf(p, l, :weight); W .-= (LR * sch * sqrt(max(1f0, size(W, 1) / size(W, 2)))) .* ns5(x .+ 0.95f0 .* mk)
        end
    end
end

rows = NamedTuple[]
const AM = SPLIT == "val" ? "val_acc" : "test_acc"
row!(metric, v; epoch = -1) = push!(rows, (item = "LIEP-EPOPT", arm = "$(KIND)_$(OPT)", model = KIND, opt = OPT, lr = Float64(LR),
                                           seed = SEED, split = SPLIT, epoch = epoch, metric = metric, value = Float64(v)))
function full_eval(ep)
    Htr, Otr = settled(psd, Xtr); Hte, Ote = settled(psd, XEV_D)
    row!("train_acc", acc_of(Otr, ytr); epoch = ep)
    for (nm, a, b) in (("probe_hidden", Htr, Hte), ("probe_output", Otr, Ote))
        A, B = standardise(vcat(real.(a), imag.(a)), vcat(real.(b), imag.(b)))
        p_acc, W, bb = softmax_regression(A, ytr, B, yev; lr = 1f-3, wd = 0f0, epochs = 30, seed = SEED)
        row!(nm, p_acc; epoch = ep)
    end
    @printf("%s epoch %d  train %.4f  probe_hidden %.4f\n", TAG, ep, rows[end - 2].value, rows[end - 1].value); flush(stdout)
end

t0 = time(); row!(AM, evaluate(psd); epoch = 0)
rng = Xoshiro(SEED); nb = length(ytr) ÷ 256; T = EPOCHS * nb; t = 0; diverged = false
for ep in 1:EPOCHS
    global t, diverged
    perm = randperm(rng, length(ytr)); te = time()
    for k in 1:nb
        t += 1; sch = OPT == "adam_const" ? 1f0 : 0.5f0 * (1 + cos(Float32(π) * (t - 1) / T))
        idx = perm[(k - 1) * 256 .+ (1:256)]
        g = first(ep_gradient(est, chain, psd, std_, todev(Xtr[:, idx]), CodebookCost(codes_d, ytr[idx])))
        step!(psd, g, t, sch)
    end
    row!("epoch_seconds", time() - te; epoch = ep)
    if any(isnan, Array(leaf(psd, :layer_1, :weight))); row!("diverged", 1; epoch = ep); diverged = true; break; end
    if ep in (1, 2, 4, 6, EPOCHS)
        a = evaluate(psd); row!(AM, a; epoch = ep)
        @printf("%s epoch %d  %s %.4f  (%.0f s)\n", TAG, ep, AM, a, time() - te); flush(stdout)
    else
        @printf("%s epoch %d  (%.0f s)\n", TAG, ep, time() - te); flush(stdout)
    end
end
!diverged && SPLIT == "test" && full_eval(EPOCHS)
row!("collapsed", diverged || rows[findlast(r -> r.metric == AM, rows)].value < 0.5 ? 1 : 0; epoch = EPOCHS)
row!("total_seconds", time() - t0)
CSV.write(joinpath(R, SPLIT, "$TAG.csv"), DataFrame(rows))
write_provenance(joinpath(R, SPLIT, TAG); seeds = [SEED],
                 config = Dict("estimator" => string(est), "rule" => OPT, "lr" => LR, "epochs" => EPOCHS, "split" => SPLIT,
                               "batch" => 256, "n_train" => length(ytr), "point" => "B", "muon_bias_lr" => 1f-3,
                               "device" => "cuda:" * CUDA.name(CUDA.device()), "host" => gethostname()))
println("DONE $TAG ($(round(Int, time() - t0)) s)")
