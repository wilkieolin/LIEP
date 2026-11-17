# LIEP-CLOSE training: LIEP-EPOPT's 8-epoch EP training with a feedback scale κ on layer 2, and an exact-gradient
# oracle. One configuration × one seed:
#   julia -t 2 --project=. experiments/close/train_close.jl \
#     <lockin|centred|exact> <muon|adam> <lr> <kappa|free> <seed> <val|test>
# Estimators as LIEP-EPOPT (point-B LockinEP; centred StaticEP β 0.1); `exact` = the settled network's exact gradient
# (exact_gpu.jl, implicit differentiation, the gradient EP estimates). Rules as LIEP-EPOPT (cosine decay; Muon on the
# weights with Adam 1e-3·cos on the biases).
# κ: layer 2's output is invariant to its row scale, but its feedback into layer 1 (W₂ᵀz₂) is not. With κ given, layer
# 2's rows (W₂ᵢ, Re b₂ᵢ, Im b₂ᵢ) start at κ × their initial norm and are projected back to it after every step, and
# layer 2's step is scaled by κ (same angular step as at κ = 1). `free`: no projection (LIEP-EPOPT).
# Norm control (LIEP-6C), env CLOSE_NORM (default `proj`, the above): `amp` = per-neuron amplitude homeostasis instead
# of the projection: unit i of layer 2 keeps a running mean aᵢ ← (1−γ)aᵢ + γ⟨|W₂ᵢz₁ + b₂ᵢ|²⟩ of its own drive at the
# settled free state (γ = CLOSE_GAMMA) and after each step rescales its row by (Aᵢ/aᵢ)^½, with Aᵢ = aᵢ at the start
# (rows already at κ × init), then sets aᵢ ← Aᵢ; it never computes a weight norm. `init` = rows scaled to κ at the
# start only, layer-2 step still scaled by κ, no control afterwards. CLOSE_OUT / CLOSE_ITEM redirect the output folder
# and item label.
# Writes results/<split>/<tag>.csv (accuracy settled via ep_predict and feedforward, feedback ratio ρ, train acc and
# hidden probe on test runs) + provenance.
using CUDA, LuxCUDA
include(joinpath(@__DIR__, "..", "base", "base_lib.jl")); include(joinpath(@__DIR__, "exact_gpu.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "CLOSE_BLAS", "4")))
const KIND = ARGS[1]; const OPT = ARGS[2]; const LR = parse(Float32, ARGS[3]); const KAP = ARGS[4]
const SEED = parse(Int, ARGS[5]); const SPLIT = ARGS[6]
KIND in ("lockin", "centred", "exact") || error("estimator $KIND"); OPT in ("adam", "muon") || error("rule $OPT")
const FREE = KAP == "free"; const κ = FREE ? 1f0 : parse(Float32, KAP)
const NORM = get(ENV, "CLOSE_NORM", "proj"); NORM in ("proj", "amp", "init") || error("CLOSE_NORM $NORM")
const GAMMA = parse(Float32, get(ENV, "CLOSE_GAMMA", "0.1")); (FREE && NORM != "proj") && error("CLOSE_NORM needs a numeric kappa")
const ITEM = get(ENV, "CLOSE_ITEM", "LIEP-CLOSE")
const EPOCHS = parse(Int, get(ENV, "CLOSE_EPOCHS", "8")); const NTRAIN = parse(Int, get(ENV, "CLOSE_NTRAIN", "60000"))
const TAG = "$(KIND)_$(OPT)_lr$(ARGS[3])_k$(KAP)_s$(SEED)_$(SPLIT)" * (EPOCHS != 8 ? "_e$EPOCHS" : "") * (NTRAIN < 60000 ? "_smoke" : "") *
            (NORM == "amp" ? "_amp$(get(ENV, "CLOSE_GAMMA", "0.1"))" : NORM == "init" ? "_init" : "")
const R = get(ENV, "CLOSE_OUT", joinpath(@__DIR__, "results")); mkpath(joinpath(R, SPLIT))

est = KIND == "lockin" ? LockinEP(ε = 0.03f0, ω_p = 0.02f0, n_cycles = 4, T_warmup_cycles = 1, T_free = 200, dt = 0.5f0) :
      KIND == "centred" ? StaticEP(β = 0.1f0, T_free = 200, T_nudge = 100, dt = 0.5f0, centered = true) : :exact
Xall, yall = fmnist(:train)
if SPLIT == "val"
    Xtr, ytr, Xev, yev = Xall[:, 1:50_000], yall[1:50_000], Xall[:, 50_001:end], yall[50_001:end]
else
    (Xtr, ytr) = (Xall, yall); Xev, yev = fmnist(:test)
end
Xtr, ytr = Xtr[:, 1:min(NTRAIN, end)], ytr[1:min(NTRAIN, end)]
chain, ps, st = paper_net(SEED); codes = paper_codes(SEED)
gdev = gpu_device(); psd, std_ = gdev(ps), gdev(st); codes_d = cu(codes)
todev(X) = cu(ComplexF32.(PN._phase_input_to_complex(X)))
const XEV_D = todev(Xev)

const LAYERS = (:layer_1, :layer_2); const FIELDS = (:weight, :bias_real, :bias_imag)
leaf(p, l, f) = getfield(getfield(p, l), f)
rownorm(p, l) = sqrt.(vec(sum(abs2, leaf(p, l, :weight); dims = 2)) .+ leaf(p, l, :bias_real) .^ 2 .+ leaf(p, l, :bias_imag) .^ 2)
scalerows!(p, l, s) = (leaf(p, l, :weight) .*= s; leaf(p, l, :bias_real) .*= s; leaf(p, l, :bias_imag) .*= s; p)
const R2 = κ .* rownorm(psd, :layer_2)
FREE || scalerows!(psd, :layer_2, κ)

function evaluate(psd)
    pred = Int[]
    for i in 1:2000:length(yev)
        L = Array(ep_predict(chain, psd, std_, XEV_D[:, i:min(i + 1999, end)], codes_d; T = 200, dt = 0.5f0, K_mode = :zero))
        append!(pred, [argmax(view(L, :, b))[1] for b in 1:size(L, 2)])
    end
    mean(pred .== yev)
end
"Feedforward accuracy of the same weights (z_l = normalise(W_l z_{l-1} + b_l), codebook readout)."
function evaluate_ff(psd)
    z = XEV_D
    for l in LAYERS
        y = leaf(psd, l, :weight) * z .+ (leaf(psd, l, :bias_real) .+ 1f0im .* leaf(psd, l, :bias_imag)); z = y ./ (abs.(y) .+ 1f-12)
    end
    L = Array(real.(adjoint(codes_d) * z)); mean([argmax(view(L, :, b)) for b in 1:size(L, 2)] .== yev)
end
"Feedback ratio: median over hidden units and 2000 images of |W₂ᵀz₂| / |W₁x + b₁| at the settled state."
function rho(psd)
    x = XEV_D[:, 1:2000]; s = phasor_settle(chain, psd, std_, x, PN.NullCost(), 0f0; T = 200, dt = 0.5f0, K_mode = :zero)
    ff = abs.(psd.layer_1.weight * x .+ (psd.layer_1.bias_real .+ 1f0im .* psd.layer_1.bias_imag)); fb = abs.(transpose(psd.layer_2.weight) * s[end])
    median(Array(fb ./ (ff .+ 1f-12)))
end
function ns5(G)
    a, b, c = 3.4445f0, -4.7750f0, 2.0315f0
    X = G ./ (sqrt(sum(abs2, G)) + 1f-7); tr = size(X, 1) > size(X, 2); tr && (X = permutedims(X))
    for _ in 1:5; A = X * transpose(X); X = a .* X .+ (b .* A .+ c .* (A * A)) * X; end
    tr ? permutedims(X) : X
end
const S = Dict{Any, Any}(); st0(k, x) = get!(() -> zero(x), S, k)
function step!(p, g, t, sch)
    β1, β2 = 0.9f0, 0.999f0; c1, c2 = 1 - β1^t, 1 - β2^t
    for l in LAYERS
        gl = getfield(g, l); lk = (l == :layer_2 && !FREE) ? κ : 1f0
        fs = OPT == "muon" ? (:bias_real, :bias_imag) : FIELDS
        ηa = (OPT == "muon" ? 1f-3 : LR) * sch * lk
        for f in fs
            x = getfield(gl, f); mk = st0((:m, l, f), x); vk = st0((:v, l, f), x)
            mk .= β1 .* mk .+ (1 - β1) .* x; vk .= β2 .* vk .+ (1 - β2) .* x .^ 2
            leaf(p, l, f) .-= ηa .* (mk ./ c1) ./ (sqrt.(vk ./ c2) .+ 1f-8)
        end
        if OPT == "muon"
            x = getfield(gl, :weight); mk = st0((:mw, l), x); mk .= 0.95f0 .* mk .+ x
            W = leaf(p, l, :weight); W .-= (LR * sch * lk * sqrt(max(1f0, size(W, 1) / size(W, 2)))) .* ns5(x .+ 0.95f0 .* mk)
        end
    end
    (FREE || NORM != "proj") || scalerows!(p, :layer_2, R2 ./ rownorm(p, :layer_2))
end
"Amplitude homeostasis (CLOSE_NORM=amp): update each layer-2 unit's running drive power from the settled free state z₁,
then rescale its row toward the set point. Returns the median scale factor applied."
const AMP = Dict{Symbol, Any}()
function homeostasis!(p, z1)
    y = leaf(p, :layer_2, :weight) * z1 .+ (leaf(p, :layer_2, :bias_real) .+ 1f0im .* leaf(p, :layer_2, :bias_imag))
    pw = vec(sum(abs2, y; dims = 2)) ./ size(y, 2)
    haskey(AMP, :A) || (AMP[:A] = copy(pw); AMP[:a] = copy(pw))
    AMP[:a] .= (1 - GAMMA) .* AMP[:a] .+ GAMMA .* pw
    s = sqrt.(AMP[:A] ./ AMP[:a]); scalerows!(p, :layer_2, s); AMP[:a] .= AMP[:A]
    median(Array(s))
end
grad(psd, x, y) = est === :exact ? (first(exact_gradient_gpu(psd, todev(x), codes_d, cu(onehot(y)))), nothing) :
                                   ep_gradient(est, chain, psd, std_, todev(x), CodebookCost(codes_d, y))

rows = NamedTuple[]
const AM = SPLIT == "val" ? "val_acc" : "test_acc"
const ARM = "$(KIND)_$(OPT)_k$(KAP)" * (NORM == "amp" ? "_amp$(get(ENV, "CLOSE_GAMMA", "0.1"))" : NORM == "init" ? "_init" : "")
row!(metric, v; epoch = -1) = push!(rows, (item = ITEM, arm = ARM, model = KIND, opt = OPT, lr = Float64(LR),
                                           kappa = KAP, seed = SEED, split = SPLIT, epochs = EPOCHS, epoch = epoch, metric = metric, value = Float64(v)))
t0 = time(); row!(AM, evaluate(psd); epoch = 0); row!("rho", rho(psd); epoch = 0)
rng = Xoshiro(SEED); nb = length(ytr) ÷ 256; T = EPOCHS * nb; t = 0; diverged = false
for ep in 1:EPOCHS
    global t, diverged
    perm = randperm(rng, length(ytr)); te = time(); ampf = Float32[]
    for k in 1:nb
        t += 1; sch = 0.5f0 * (1 + cos(Float32(π) * (t - 1) / T))
        idx = perm[(k - 1) * 256 .+ (1:256)]
        g, sfree = grad(psd, Xtr[:, idx], ytr[idx])
        step!(psd, g, t, sch)
        NORM == "amp" && (push!(ampf, homeostasis!(psd, sfree[1])))
    end
    row!("epoch_seconds", time() - te; epoch = ep)
    NORM == "amp" && row!("amp_scale_median", median(ampf); epoch = ep)
    if any(isnan, Array(leaf(psd, :layer_1, :weight))); row!("diverged", 1; epoch = ep); diverged = true; break; end
    if ep in (1, 2, 4, 6, 8, 12, 16, 24, EPOCHS)
        a = evaluate(psd); row!(AM, a; epoch = ep); r = rho(psd); row!("rho", r; epoch = ep)
        @printf("%s epoch %d  %s %.4f  rho %.3f  (%.0f s)\n", TAG, ep, AM, a, r, time() - te); flush(stdout)
    else
        @printf("%s epoch %d  (%.0f s)\n", TAG, ep, time() - te); flush(stdout)
    end
end
if !diverged
    row!(AM * "_ff", evaluate_ff(psd); epoch = EPOCHS)
    if SPLIT == "test"
        Htr = Matrix{ComplexF32}(undef, 256, length(ytr)); Otr = Matrix{ComplexF32}(undef, 64, length(ytr))
        Hte = Matrix{ComplexF32}(undef, 256, length(yev))
        for (X, H, O) in ((todev(Xtr), Htr, Otr), (XEV_D, Hte, nothing))
            for i in 1:2000:size(X, 2)
                j = i:min(i + 1999, size(X, 2)); s = phasor_settle(chain, psd, std_, X[:, j], PN.NullCost(), 0f0; T = 200, dt = 0.5f0, K_mode = :zero)
                H[:, j] = Array(s[1]); O === nothing || (O[:, j] = Array(s[end]))
            end
        end
        row!("train_acc", mean([argmax(view(real.(adjoint(codes) * Otr), :, b)) for b in 1:size(Otr, 2)] .== ytr); epoch = EPOCHS)
        A, Bm = standardise(vcat(real.(Htr), imag.(Htr)), vcat(real.(Hte), imag.(Hte)))
        row!("probe_hidden", first(softmax_regression(A, ytr, Bm, yev; lr = 1f-3, wd = 0f0, epochs = 30, seed = SEED)); epoch = EPOCHS)
    end
end
row!("collapsed", diverged || rows[findlast(r -> r.metric == AM, rows)].value < 0.5 ? 1 : 0; epoch = EPOCHS)
row!("total_seconds", time() - t0)
CSV.write(joinpath(R, SPLIT, "$TAG.csv"), DataFrame(rows))
write_provenance(joinpath(R, SPLIT, TAG); seeds = [SEED],
                 config = Dict("estimator" => string(est), "rule" => OPT, "lr" => LR, "kappa" => KAP, "norm" => NORM, "gamma" => GAMMA, "epochs" => EPOCHS, "split" => SPLIT,
                               "batch" => 256, "n_train" => length(ytr), "muon_bias_lr" => 1f-3,
                               "device" => "cuda:" * CUDA.name(CUDA.device()), "host" => gethostname()))
println("DONE $TAG ($(round(Int, time() - t0)) s)")
