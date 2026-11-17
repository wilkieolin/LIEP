# LIEP-BASE job runner (README "Method"). One job per call; results/<part>/<tag>.csv (long format) + provenance.
#   julia --project=. -t 4 experiments/base/run.jl lr <pixels|phase>
#   … run.jl phasorbp <seed>             … run.jl relu <256|500|256-64> <seed> <epochs>
#   … run.jl sep <256|500> <clip|half>   (10-trial tuning on the held-out split, then 3 seeds × 8 epochs + 1 × 30)
#   … run.jl probe <lockin|centred|phasorbp> <seed>
#   … run.jl lrbasis <pixels|z|tanh|sin|cos|cossin_raw|cossin|zz2|xx2>   (README addendum 2026-11-17)
include(joinpath(@__DIR__, "base_lib.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "BASE_BLAS", "4")))
const PART = ARGS[1]
rows = NamedTuple[]
row!(arm, model, seed, metric, v; epoch = -1, extra = "") = push!(rows, (item = "LIEP-BASE", arm = arm, model = model, seed = seed,
                                                                        epoch = epoch, setting = extra, metric = metric, value = Float64(v)))
Ptr_raw, ytr = pixels(:train); Pte_raw, yte = pixels(:test)
t0 = time()

if PART == "lr"                      # logistic regression on the input
    feat = ARGS[2]
    if feat == "pixels"
        Xtr, Xte = standardise(Ptr_raw, Pte_raw)
    else
        Etr, _ = fmnist(:train); Ete, _ = fmnist(:test); Xtr, Xte = phase_features(Etr), phase_features(Ete)
    end
    best = (0.0, 1f-3, 0f0)
    for lr in (1f-3, 3f-3), wd in (0f0, 1f-4, 1f-3)
        a, = softmax_regression(splits(Xtr, ytr, Xte, yte; val = true)...; lr = lr, wd = wd, epochs = 30, seed = 42)
        row!("logreg", feat, 42, "val_acc", a; extra = "lr=$lr wd=$wd"); a > best[1] && (global best = (a, lr, wd))
        @printf("lr %s: lr %.0e wd %.0e val %.4f\n", feat, lr, wd, a); flush(stdout)
    end
    for s in SEEDS
        a, = softmax_regression(Xtr, ytr, Xte, yte; lr = best[2], wd = best[3], epochs = 30, seed = s)
        row!("logreg", feat, s, "test_acc", a; extra = "lr=$(best[2]) wd=$(best[3])"); @printf("lr %s seed %d test %.4f\n", feat, s, a)
    end
    tag = "lr_$feat"
elseif PART == "lrbasis"             # README addendum 2026-11-17: which step of the encoding helps logistic regression
    feat = ARGS[2]
    zs(X) = (X .- mean(X; dims = 1)) ./ (std(X; dims = 1) .+ 1f-6)
    us(X) = 0.5f0 .* tanh.(zs(X))
    F = Dict("pixels" => X -> X, "z" => zs, "tanh" => us, "sin" => X -> sinpi.(us(X)), "cos" => X -> cospi.(us(X)),
             "cossin_raw" => X -> vcat(cospi.(us(X)), sinpi.(us(X))), "cossin" => X -> vcat(cospi.(us(X)), sinpi.(us(X))),
             "zz2" => X -> vcat(zs(X), zs(X) .^ 2), "xx2" => X -> vcat(X, X .^ 2))[feat]
    Xtr, Xte = F(Ptr_raw), F(Pte_raw)
    feat == "cossin_raw" || ((Xtr, Xte) = standardise(Xtr, Xte))
    best = (0.0, 1f-3, 0f0)
    for lr in (1f-3, 3f-3), wd in (0f0, 1f-4, 1f-3)
        a, = softmax_regression(splits(Xtr, ytr, Xte, yte; val = true)...; lr = lr, wd = wd, epochs = 30, seed = 42)
        row!("lrbasis", feat, 42, "val_acc", a; extra = "lr=$lr wd=$wd"); a > best[1] && (global best = (a, lr, wd))
        @printf("lrbasis %s: lr %.0e wd %.0e val %.4f\n", feat, lr, wd, a); flush(stdout)
    end
    for s in SEEDS
        a, = softmax_regression(Xtr, ytr, Xte, yte; lr = best[2], wd = best[3], epochs = 30, seed = s)
        row!("lrbasis", feat, s, "test_acc", a; extra = "lr=$(best[2]) wd=$(best[3])"); @printf("lrbasis %s seed %d test %.4f\n", feat, s, a); flush(stdout)
    end
    tag = "lrbasis_$feat"
elseif PART == "phasorbp"            # the paper's feedforward phasor arm (E8), checkpoints for the probes
    seed = parse(Int, ARGS[2]); m, ps, st = build_phasor(Xoshiro(seed)); codes = paper_codes(seed)
    Etr, _ = fmnist(:train); Ete, _ = fmnist(:test); Y = onehot(ytr)
    mkpath(joinpath(OUT, "ckpt"))
    acc(p) = mean([argmax(view(phasor_logits(m, p, st, Ete[:, i:min(i + 999, end)], codes), :, j)) for i in 1:1000:10_000
                   for j in 1:min(1000, 10_001 - i)] .== yte)
    hist, _ = train_bp((p, x, y) -> logsoftmax_ce(phasor_logits(m, p, st, x, codes), y), acc, ps, Etr, Y, length(ytr);
                       seed = seed, ckpt = (ep, p) -> serialize(joinpath(OUT, "ckpt", "phasorbp_s$(seed)_e$(ep).jls"), p))
    for (ep, a) in hist; row!("phasor_bp", "784-256-64", seed, "test_acc", a; epoch = ep); end
    tag = "phasorbp_s$seed"
elseif PART == "relu"                # ReLU MLP with backprop, the ceiling
    sz = ARGS[2]; seed = parse(Int, ARGS[3]); ep = parse(Int, ARGS[4])
    sizes = [784; parse.(Int, split(sz, "-")); 10]; m, ps, st = build_relu(sizes, Xoshiro(seed))
    Xtr, Xte = standardise(Ptr_raw, Pte_raw); Y = onehot(ytr)
    acc(p) = mean([argmax(view(first(m(Xte, p, st)), :, j)) for j in 1:10_000] .== yte)
    hist, _ = train_bp((p, x, y) -> logsoftmax_ce(first(m(x, p, st)), y), acc, ps, Xtr, Y, length(ytr); epochs = ep, seed = seed,
                       evals = (0, 1, 4, 8, ep))
    for (e, a) in hist; row!("relu_bp", "784-$sz-10", seed, "test_acc", a; epoch = e, extra = "epochs=$ep"); end
    tag = "relu_$(sz)_s$(seed)_e$(ep)"
elseif PART == "sep"                 # standard centred EP
    H = parse(Int, ARGS[2]); act = ARGS[3]; σ = act == "clip" ? σ_clip : σ_half
    # grid 2 (README, after grid 1's optimum sat on its edge): the same 10 trials, shifted to bracket it
    G2 = get(ENV, "SEP_GRID", "1") == "2"
    LRS = G2 ? (0.01f0, 0.02f0, 0.05f0, 0.1f0, 0.2f0) : (0.02f0, 0.05f0, 0.1f0, 0.2f0, 0.5f0)
    BETAS = G2 ? (0.1f0, 0.25f0) : (0.25f0, 0.5f0)
    best = (0.0, 0.1f0, 0.5f0)
    for lr in LRS, β in BETAS
        h = train_sep(H, σ, splits(Ptr_raw, ytr, Pte_raw, yte; val = true)...; lr = lr, β = β, epochs = 2, seed = 42, evals = (2,))
        a = h[end][2]; row!("standard_ep", "784-$H-10 $act", 42, "val_acc_e2", a; epoch = 2, extra = "lr=$lr beta=$β")
        @printf("sep %d %s lr %.2f β %.2f val %.4f\n", H, act, lr, β, a); flush(stdout); a > best[1] && (global best = (a, lr, β))
    end
    for (s, ep) in [(s, 8) for s in SEEDS] ∪ [(42, 30)]
        h = train_sep(H, σ, Ptr_raw, ytr, Pte_raw, yte; lr = best[2], β = best[3], epochs = ep, seed = s, evals = (0, 1, 4, 8, 30))
        for (e, a) in h; row!("standard_ep", "784-$H-10 $act", s, "test_acc", a; epoch = e, extra = "lr=$(best[2]) beta=$(best[3]) epochs=$ep"); end
        @printf("sep %d %s seed %d (%d epochs) test %.4f\n", H, act, s, ep, h[end][2]); flush(stdout)
    end
    tag = "sep_$(H)_$(act)" * (G2 ? "_grid2" : "")
elseif PART == "probe"               # linear probe on the hidden layer at epochs 0/1/4/8
    arm = ARGS[2]; seed = parse(Int, ARGS[3])
    Etr, _ = fmnist(:train); Ete, _ = fmnist(:test); codes = paper_codes(seed)
    ckdir(ep) = arm == "phasorbp" ? joinpath(OUT, "ckpt", "phasorbp_s$(seed)_e$(ep).jls") :
                joinpath(@__DIR__, "..", "regen", "results", "ckpt", "$(arm)_B_lr1e-3_s$(seed)_e$(ep).jls")
    _, _, st = paper_net(seed); m, _, stm = build_phasor(Xoshiro(seed))
    function hidden(ps, E)                # [Re z₁; Im z₁] for all columns of E
        F = Matrix{Float32}(undef, 512, size(E, 2))
        for i in 1:2000:size(E, 2)
            j = i:min(i + 1999, size(E, 2)); x = E[:, j]
            if arm == "phasorbp"          # the BP net's own inference: the forward pass of layer 1
                z = ComplexF32.(angle_to_complex(first(Lux.apply(m.layers[1], x, ps.layer_1, stm.layer_1))))
            else                          # the EP net's inference: the settled free state (T 200, dt 0.5)
                c = Ctx(ps, x, CodebookCost(codes, ones(Int, length(j))), IDEAL, 0.5f0); z = settle(c, 0f0, 200, Xoshiro(0))[1]
            end
            F[:, j] = vcat(real.(z), imag.(z))
        end
        F
    end
    for ep in (0, 1, 4, 8)
        ps = deserialize(ckdir(ep)); Ftr, Fte = standardise(hidden(ps, Etr), hidden(ps, Ete))
        a, = softmax_regression(Ftr, ytr, Fte, yte; lr = 1f-3, wd = 0f0, epochs = 30, seed = seed)
        row!("probe", arm, seed, "test_acc", a; epoch = ep); @printf("probe %s seed %d epoch %d test %.4f (%.0f s)\n", arm, seed, ep, a, time() - t0); flush(stdout)
    end
    tag = "probe_$(arm)_s$seed"
end
mkpath(joinpath(OUT, PART))
CSV.write(joinpath(OUT, PART, "$tag.csv"), DataFrame(rows))
write_provenance(joinpath(OUT, PART, tag); seeds = SEEDS, config = Dict("args" => join(ARGS, " ")))
println("DONE $tag ($(round(Int, time() - t0)) s)")
