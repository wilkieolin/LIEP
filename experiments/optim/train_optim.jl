# LIEP-OPTIM: optimiser screen on the paper's phasor network under backprop (README). One configuration × one seed.
#   julia -t 2 --project=. experiments/optim/train_optim.jl \
#     <opt> <epochs> <lr> <wd> <seed> <val|test>
# Network and readout as LIEP-GAP `cb64_t1` (784→256→64 PhasorDense, hard projection, codebook logits Re⟨code,z⟩/64,
# softmax CE), batch 256. `val`: train on the first 50k training images, score the last 10k (tuning); `test`: 60k/10k.
# Every optimiser runs through `step!` below on the six real parameter tensors (weight, bias_real, bias_imag of each
# layer; log_neg_lambda is unused by the static complex path and gets no gradient). lr is multiplied by the cosine
# factor ½(1 + cos π(t−1)/T) per step for every optimiser except `adam_const`.
#   sgd        θ −= η g
#   sgdm       m = 0.9 m + g; θ −= η m                                  (the control, with cosine)
#   sgdm_wd    sgdm on g + wd·θ (coupled L2 on all of a unit's parameters: the row (W_i, b_i))
#   sgdm_clip  sgdm; the step's global gradient norm clipped at 2× its running mean (EMA 0.99 of clipped norms)
#   sphere     sgdm, then each unit's row (W_i, Re b_i, Im b_i) rescaled to its initial norm (projected SGD on spheres)
#   adam       Adam β (0.9, 0.999), ε 1e-8, bias-corrected        adam_const   … at constant lr (the paper's rule)
#   adam_proj2 adam, then layer 2's rows rescaled to their initial norm (LIEP-6C's control B2: the norm constraint EP
#              uses at κ, which leaves BP's function unchanged since layer 2 is scale-invariant; κ itself is immaterial
#              here because the step is scaled with it)
#   adamw      adam, plus decoupled decay θ −= η·wd·θ on every row parameter
#   cadam      adam with one second moment per complex bias, (g_re² + g_im²)/2 (rotation-equivariant on the bias)
#   lion       c = 0.9 m + 0.1 g; θ −= η sign(c); m = 0.99 m + 0.01 g
#   muon       weights: m = 0.95 m + g, u = g + 0.95 m, θ −= η·max(1, rows/cols)^½·NS5(u) (Newton–Schulz orthogonalised);
#              biases: adam at 1e-3 with the same schedule
# Writes results/<split>/<tag>.csv (long format) + provenance; `test` runs also the end-of-training diagnostics (README
# stage D: scale invariance, near-origin units with and without the bias, gradient share, sharpness at unit row norm).
include(joinpath(@__DIR__, "..", "gap", "gap_lib.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "OPTIM_BLAS", "6")))
const OPT = ARGS[1]; const EPOCHS = parse(Int, ARGS[2]); const LR = parse(Float32, ARGS[3]); const WD = parse(Float32, ARGS[4])
const SEED = parse(Int, ARGS[5]); const SPLIT = ARGS[6]
OPT in ("sgd", "sgdm", "sgdm_wd", "sgdm_clip", "sphere", "adam", "adam_const", "adam_proj2", "adamw", "cadam", "lion", "muon") || error("optimiser $OPT")
SPLIT in ("val", "test") || error("split $SPLIT")
const TAG = "$(OPT)_e$(EPOCHS)_lr$(ARGS[3])_wd$(ARGS[4])_s$(SEED)_$(SPLIT)"
const R = get(ENV, "OPTIM_OUT", joinpath(@__DIR__, "results")); mkpath(joinpath(R, SPLIT))
const ITEM = get(ENV, "OPTIM_ITEM", "LIEP-OPTIM")   # LIEP-6C reuses this script with its own output folder
const COSINE = OPT != "adam_const"; const DOUT = 64; const TAU = 1f0

m, ps0, st = build_net(SEED, DOUT); st = dev(st); ps = dev(ps0)
codes = dev(paper_codes(SEED; dout = DOUT))
Etr_h, ytr_all = fmnist(:train); Ete_h, yte_h = fmnist(:test)
if SPLIT == "val"
    Etr_h, ytr, Eev_h, yev = Etr_h[:, 1:50_000], ytr_all[1:50_000], Etr_h[:, 50_001:end], ytr_all[50_001:end]
else
    ytr, Eev_h, yev = ytr_all, Ete_h, yte_h
end
Xtr = dev(to_complex(Etr_h)); Xev = dev(to_complex(Eev_h)); Y = dev(onehot(ytr)); Yev = dev(onehot(yev))

logits(p, x) = readout_logits(:codebook, TAU, codes, p, forward(m, p, st, x)[4])
loss(p, x, Yb) = logsoftmax_ce(logits(p, x), Yb)
acc(p, X, y) = (pred = Int[]; for i in 1:2000:size(X, 2); L = host(logits(p, X[:, i:min(i + 1999, end)])); append!(pred, [argmax(view(L, :, j)) for j in 1:size(L, 2)]); end; mean(pred .== y))

const LAYERS = (:layer_1, :layer_2); const FIELDS = (:weight, :bias_real, :bias_imag)
leaf(p, l, f) = getfield(getfield(p, l), f)
"Per-unit row norms ‖(W_i, Re b_i, Im b_i)‖ of layer l (device vector)."
rownorm(p, l) = sqrt.(vec(sum(abs2, leaf(p, l, :weight); dims = 2)) .+ leaf(p, l, :bias_real) .^ 2 .+ leaf(p, l, :bias_imag) .^ 2)
"Scale each unit's row of layer l by s (device vector), in place."
function scalerows!(p, l, s)
    leaf(p, l, :weight) .*= s; leaf(p, l, :bias_real) .*= s; leaf(p, l, :bias_imag) .*= s
end
const R0 = Dict(l => rownorm(ps, l) for l in LAYERS)

function ns5(G)
    a, b, c = 3.4445f0, -4.7750f0, 2.0315f0
    X = G ./ (sqrt(sum(abs2, G)) + 1f-7); tr = size(X, 1) > size(X, 2); tr && (X = permutedims(X))
    for _ in 1:5
        A = X * transpose(X); X = a .* X .+ (b .* A .+ c .* (A * A)) * X
    end
    tr ? permutedims(X) : X
end

const S = Dict{Any, Any}()                                  # optimiser state: (:m|:v, layer, field) → array
st0(k, x) = get!(() -> zero(x), S, k)
mutable struct Stats; ema::Float32; nclip::Int; gn::Vector{Float32}; end
const STATS = Stats(0f0, 0, Float32[])

"One optimiser step at step t with learning rate η (schedule applied by the caller)."
function step!(p, g, t, η, sch)
    G = Dict((l, f) => getfield(getfield(g, l), f) for l in LAYERS for f in FIELDS)
    n = sqrt(sum(sum(abs2, x) for x in values(G))); push!(STATS.gn, n)
    if OPT == "sgdm_wd"
        for (k, x) in G; G[k] = x .+ WD .* leaf(p, k...); end
    elseif OPT == "sgdm_clip"
        t == 1 && (STATS.ema = n); thr = 2f0 * STATS.ema
        if n > thr; for (k, x) in G; G[k] = x .* (thr / n); end; STATS.nclip += 1; end
        STATS.ema = 0.99f0 * STATS.ema + 0.01f0 * min(n, thr)
    end
    if OPT == "sgd"
        for (k, x) in G; leaf(p, k...) .-= η .* x; end
    elseif OPT in ("sgdm", "sgdm_wd", "sgdm_clip", "sphere")
        for (k, x) in G; mk = st0((:m, k...), x); mk .= 0.9f0 .* mk .+ x; leaf(p, k...) .-= η .* mk; end
        if OPT == "sphere"; for l in LAYERS; scalerows!(p, l, R0[l] ./ rownorm(p, l)); end; end
    elseif OPT in ("adam", "adam_const", "adam_proj2", "adamw", "cadam") || OPT == "muon"
        β1, β2 = 0.9f0, 0.999f0; c1, c2 = 1 - β1^t, 1 - β2^t
        for l in LAYERS
            if OPT == "adamw"; for f in FIELDS; leaf(p, l, f) .-= (η * WD) .* leaf(p, l, f); end; end
            fs = OPT == "muon" ? (:bias_real, :bias_imag) : FIELDS
            ηa = OPT == "muon" ? 1f-3 * sch : η
            if OPT == "cadam"                                # one second moment per complex bias
                vb = st0((:v, l, :bias), G[(l, :bias_real)])
                vb .= β2 .* vb .+ (1 - β2) .* (G[(l, :bias_real)] .^ 2 .+ G[(l, :bias_imag)] .^ 2) ./ 2
            end
            for f in fs
                x = G[(l, f)]; mk = st0((:m, l, f), x); mk .= β1 .* mk .+ (1 - β1) .* x
                v = (OPT == "cadam" && f != :weight) ? S[(:v, l, :bias)] : (vk = st0((:v, l, f), x); vk .= β2 .* vk .+ (1 - β2) .* x .^ 2; vk)
                leaf(p, l, f) .-= ηa .* (mk ./ c1) ./ (sqrt.(v ./ c2) .+ 1f-8)
            end
            if OPT == "muon"
                x = G[(l, :weight)]; mk = st0((:mw, l), x); mk .= 0.95f0 .* mk .+ x
                W = leaf(p, l, :weight); W .-= (η * sqrt(max(1f0, size(W, 1) / size(W, 2)))) .* ns5(x .+ 0.95f0 .* mk)
            end
        end
        OPT == "adam_proj2" && scalerows!(p, :layer_2, R0[:layer_2] ./ rownorm(p, :layer_2))
    elseif OPT == "lion"
        for (k, x) in G; mk = st0((:m, k...), x); leaf(p, k...) .-= η .* sign.(0.9f0 .* mk .+ 0.1f0 .* x); mk .= 0.99f0 .* mk .+ 0.01f0 .* x; end
    end
end

rows = NamedTuple[]
row!(metric, v; epoch = -1, layer = "") = push!(rows, (item = ITEM, arm = OPT, model = "bp_784-256-64_cb64_t1", seed = SEED,
                                                      epochs = EPOCHS, lr = Float64(LR), wd = Float64(WD), split = SPLIT,
                                                      epoch = epoch, layer = layer, metric = metric, value = Float64(v)))
const AM = SPLIT == "val" ? "val_acc" : "test_acc"
function log_epoch(ep, ang)
    row!(AM, acc(ps, Xev, yev); epoch = ep)
    for (i, l) in enumerate(LAYERS)
        row!("rownorm_median", median(host(rownorm(ps, l))); epoch = ep, layer = string(l))
        ep > 0 && row!("angular_step_median", ang[i]; epoch = ep, layer = string(l))
    end
end

t0 = time(); n = length(ytr); nb = cld(n, 256); T = EPOCHS * nb; idx = collect(1:n); t = 0
log_epoch(0, nothing); evals = Set([1, 2, 4, 8, 16, 24, EPOCHS])
for ep in 1:EPOCHS
    global t
    shuffle!(Xoshiro(SEED + 7919ep), idx); ang = [Float32[], Float32[]]
    for i in 1:256:n
        t += 1; sch = COSINE ? 0.5f0 * (1 + cos(Float32(π) * (t - 1) / T)) : 1f0
        b = idx[i:min(i + 255, n)]
        g = Zygote.gradient(p -> loss(p, Xtr[:, b], Y[:, b]), ps)[1]
        old = t % 10 == 0 ? Dict(l => (copy(leaf(ps, l, :weight)), copy(leaf(ps, l, :bias_real)), copy(leaf(ps, l, :bias_imag))) for l in LAYERS) : nothing
        step!(ps, g, t, LR * sch, sch)
        if old !== nothing                                  # median over units of the angle a row turned this step
            for (j, l) in enumerate(LAYERS)
                W0, r0, i0 = old[l]; W = leaf(ps, l, :weight)
                dot_ = vec(sum(W0 .* W; dims = 2)) .+ r0 .* leaf(ps, l, :bias_real) .+ i0 .* leaf(ps, l, :bias_imag)
                n0 = sqrt.(vec(sum(abs2, W0; dims = 2)) .+ r0 .^ 2 .+ i0 .^ 2)
                push!(ang[j], median(host(acos.(clamp.(dot_ ./ (n0 .* rownorm(ps, l)), -1f0, 1f0)))))
            end
        end
    end
    any(isnan, host(leaf(ps, :layer_1, :weight))) && (row!("diverged", 1; epoch = ep); break)
    if ep in evals
        log_epoch(ep, [mean(a) for a in ang]); @printf("%s epoch %d %s %.4f (%.0f s)\n", TAG, ep, AM, rows[findlast(r -> r.metric == AM, rows)].value, time() - t0); flush(stdout)
    end
end
row!("train_acc", acc(ps, Xtr, ytr); epoch = EPOCHS)
L = host(logits(ps, Xtr[:, 1:10_000])); L = L .- maximum(L; dims = 1); P = exp.(L); P ./= sum(P; dims = 1)
row!("train_ce", -mean(log.(P[CartesianIndex.(ytr[1:10_000], 1:10_000)])); epoch = EPOCHS)
gn = STATS.gn
row!("gradnorm_median", median(gn)); row!("gradnorm_p99", quantile(gn, 0.99)); row!("gradnorm_max_over_median", maximum(gn) / median(gn))
row!("gradnorm_kurtosis", mean(((gn .- mean(gn)) ./ std(gn)) .^ 4))
OPT == "sgdm_clip" && row!("clip_fraction", STATS.nclip / length(gn))

# ------------------------------------------------------------------ stage D diagnostics on the trained net (test runs)
if SPLIT == "test"
    xb = Xev[:, 1:2048]; Yb = Yev[:, 1:2048]; L0 = loss(ps, xb, Yb)
    # D1 scale invariance: per-unit random factors in [0.5, 2] leave the loss unchanged; gradients are ⟂ to each row
    pc = deepcopy(ps); rng = Xoshiro(SEED)
    for l in LAYERS; scalerows!(pc, l, dev(Float32.(2 .^ (2 .* rand(rng, length(host(R0[l]))) .- 1)))); end
    row!("scale_inv_rel_dloss", abs(loss(pc, xb, Yb) - L0) / L0)
    g = Zygote.gradient(p -> loss(p, xb, Yb), ps)[1]
    for l in LAYERS
        gw, gr, gi = getfield(g, l).weight, getfield(g, l).bias_real, getfield(g, l).bias_imag
        dot_ = vec(sum(gw .* leaf(ps, l, :weight); dims = 2)) .+ gr .* leaf(ps, l, :bias_real) .+ gi .* leaf(ps, l, :bias_imag)
        gnr = sqrt.(vec(sum(abs2, gw; dims = 2)) .+ gr .^ 2 .+ gi .^ 2)
        row!("grad_row_cos_median", median(host(abs.(dot_) ./ (gnr .* rownorm(ps, l) .+ 1f-30))); layer = string(l))
    end
    # D3/D4 near-origin units: |y| < 0.1 × the unit's median |y| over the test set, with and without the bias
    y1, z1, y2, _ = forward(m, ps, st, Xev)
    p0 = deepcopy(ps); for l in LAYERS; leaf(p0, l, :bias_real) .= 0; leaf(p0, l, :bias_imag) .= 0; end
    y1n = first(Lux.apply(m.layers[1], Xev, p0.layer_1, st.layer_1)); y2n = first(Lux.apply(m.layers[2], z1, p0.layer_2, st.layer_2))
    for (nm, y, yn, l) in (("layer_1", y1, y1n, :layer_1), ("layer_2", y2, y2n, :layer_2))
        a = host(abs.(y)); an = host(abs.(yn)); med = median(a; dims = 2)
        row!("near_origin_frac", mean(a .< 0.1f0 .* med); layer = nm); row!("near_origin_frac_nobias", mean(an .< 0.1f0 .* med); layer = nm)
        row!("absy_q05_over_median", median(vec(mapslices(r -> quantile(r, 0.05), a; dims = 2)) ./ vec(med)); layer = nm)
        b = abs.(host(leaf(ps, l, :bias_real)) .+ im .* host(leaf(ps, l, :bias_imag)))
        row!("bias_over_drive_median", median(b ./ vec(median(an; dims = 2))); layer = nm)
    end
    # D3 gradient share of near-origin entries: Σ|∂L/∂y| over (unit, sample) with |y| < 0.1 median, over the total
    yb1, zb1, yb2, _ = forward(m, ps, st, xb)
    δ1 = Zygote.gradient(y -> logsoftmax_ce(readout_logits(:codebook, TAU, codes, ps, hard(first(Lux.apply(m.layers[2], hard(y), ps.layer_2, st.layer_2)))), Yb), yb1)[1]
    δ2 = Zygote.gradient(y -> logsoftmax_ce(readout_logits(:codebook, TAU, codes, ps, hard(y)), Yb), yb2)[1]
    for (nm, y, δ) in (("layer_1", yb1, δ1), ("layer_2", yb2, δ2))
        a = host(abs.(y)); d = host(abs.(δ)); near = a .< 0.1f0 .* median(a; dims = 2)
        row!("near_origin_grad_share", sum(d[near]) / sum(d); layer = nm)
    end
    # D5 sharpness: top Hessian eigenvalue by power iteration (central-difference HVP, 20 iterations) on 4096 training
    # images, raw and at unit row norm (every row rescaled to norm 1; the function is unchanged)
    xs = Xtr[:, 1:4096]; Ys = Y[:, 1:4096]
    flat(p) = vcat((vec(leaf(p, l, f)) for l in LAYERS for f in FIELDS)...)
    function unflat!(p, v)
        o = 0; for l in LAYERS, f in FIELDS; x = leaf(p, l, f); x .= reshape(v[o + 1:o + length(x)], size(x)); o += length(x); end
    end
    gradflat(p) = (g = Zygote.gradient(q -> loss(q, xs, Ys), p)[1]; vcat((vec(getfield(getfield(g, l), f)) for l in LAYERS for f in FIELDS)...))
    function sharpness(p)
        θ = flat(p); q = deepcopy(p); v = dev(randn(Xoshiro(SEED), Float32, length(θ))); v ./= sqrt(sum(abs2, v)); λ = 0f0; h = 1f-2
        for _ in 1:20
            unflat!(q, θ .+ h .* v); gp = gradflat(q); unflat!(q, θ .- h .* v); gm = gradflat(q)
            Hv = (gp .- gm) ./ (2h); λ = sum(v .* Hv); v = Hv ./ sqrt(sum(abs2, Hv))
        end
        λ
    end
    row!("sharpness_raw", sharpness(ps))
    pu = deepcopy(ps); for l in LAYERS; scalerows!(pu, l, 1f0 ./ rownorm(pu, l)); end
    row!("sharpness_unit_rows", sharpness(pu))
end
if SPLIT == "test"                                          # hidden probe (as LIEP-GAP): the representation, not the readout
    H(X) = (H = Matrix{Float32}(undef, 512, size(X, 2)); for i in 1:2000:size(X, 2); j = i:min(i + 1999, size(X, 2)); z = forward(m, ps, st, X[:, j])[2]; H[:, j] = host(vcat(real.(z), imag.(z))); end; H)
    A, B = standardise(H(Xtr), H(Xev)); p_acc, = softmax_regression(A, ytr, B, yev; lr = 1f-3, wd = 0f0, epochs = 30, seed = SEED)
    row!("probe_hidden", p_acc; epoch = EPOCHS)
end
row!("seconds", time() - t0)
CSV.write(joinpath(R, SPLIT, "$TAG.csv"), DataFrame(rows))
write_provenance(joinpath(R, SPLIT, TAG); seeds = [SEED], config = Dict("optimiser" => OPT, "epochs" => EPOCHS, "lr" => LR, "wd" => WD,
                 "split" => SPLIT, "cosine" => COSINE, "readout" => "cb64_t1", "batch" => 256,
                 "device" => DEV == "gpu" ? "cuda:" * CUDA.name(CUDA.device()) : "cpu", "host" => gethostname()))
println("DONE $TAG ($(round(Int, time() - t0)) s)")
