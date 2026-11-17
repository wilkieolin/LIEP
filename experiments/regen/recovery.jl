# LIEP-REGEN stage 4b: Fig 7, recovery of a backprop-trained network from weight damage, every arm given the same
# fault mask (README). Usage (repo root, spark GPU):
#   julia -t 2 --project=. experiments/regen/recovery.jl pretrain
#   … recovery.jl <none|gaussian|lognormal|stuck_zero|stuck_sat> <level> <draw>      (all four arms on one damage draw)
# Writes results/recovery/<tag>.csv (long format) + provenance; the pretrained weights to results/recovery/pretrain.jls.
using CUDA, LuxCUDA
include(joinpath(@__DIR__, "..", "common", "liep_lib.jl"))
include(joinpath(@__DIR__, "..", "common", "provenance.jl"))
using CSV, DataFrames, Optimisers, Printf, Serialization, Zygote
import PhasorNetworks: normalize_to_unit_circle

const OUT = joinpath(@__DIR__, "results", "recovery"); mkpath(OUT)
const SEED = 0x42; const FT_EPOCHS = 3; const BATCH = 512; const WD = 1f-4
gdev = gpu_device(); cpu = cpu_device()
Xtr, ytr = fmnist(:train, parse(Int, get(ENV, "REC_NTRAIN", "60000"))); Xte, yte = fmnist(:test)   # REC_NTRAIN: smoke only
todev(X) = cu(ComplexF32.(PN._phase_input_to_complex(X)))
const XTE_D = todev(Xte)
chain, ps0, st = paper_net(Int(SEED)); codes = paper_codes(Int(SEED)); codes_d = cu(codes); std_ = gdev(st)
onehot(y) = (Y = zeros(Float32, 10, length(y)); for (i, c) in enumerate(y); Y[c, i] = 1f0; end; Y)
todev_phase(X) = cu(X)                                       # Phase input for the feedforward chain

# backprop through the feedforward chain with the codebook readout (legacy bp_loss)
ff_logits(p, x) = similarity_outer(ComplexF32.(angle_to_complex(first(Lux.apply(chain, x, p, std_)))), codes_d)
function bp_loss(p, x, Y)
    L = ff_logits(p, x); L = L .- maximum(L; dims = 1)
    -sum(Y .* (L .- log.(sum(exp.(L); dims = 1)))) / size(L, 2)
end
"Accuracy on 10k by the EP settle (T 200, dt 0.5), as the paper evaluates every arm."
function ep_acc(psd)
    pred = Int[]
    for i in 1:2000:10_000
        lg = Array(ep_predict(chain, psd, std_, XTE_D[:, i:i + 1999], codes_d; T = 200, dt = 0.5f0, K_mode = :zero))
        append!(pred, [argmax(view(lg, :, b))[1] for b in 1:2000])
    end
    mean(pred .== yte)
end
ff_acc(psd) = mean(vcat([[argmax(view(Array(ff_logits(psd, cu(Xte[:, i:i + 1999]))), :, b))[1] for b in 1:2000] for i in 1:2000:10_000]...) .== yte)

masked(g, M) = M === nothing ? g : NamedTuple(k => merge(g[k], (weight = g[k].weight .* M[k],)) for k in keys(g))
# Weight decay goes into the gradient BEFORE masking and freezing, then plain Adam: a frozen or stuck parameter then
# gets exactly zero gradient and never moves. (Optimisers.WeightDecay inside the chain would add wd·θ after the mask,
# and Adam would turn it into full-size steps: the first run's frozen layer 1 decayed and readout-only fell to 0.10.)
decayed(g, p) = NamedTuple(k => NamedTuple(q => (q === :weight && g[k][q] !== nothing ? g[k][q] .+ WD .* p[k][q] : g[k][q])
                                            for q in keys(g[k])) for k in keys(g))      # weights only, not the dynamics parameter
"One fine-tune: `grad(p, x, y) -> g` per batch, Adam 1e-3 with weight decay 1e-4, the fault mask on every update."
function finetune(psd, grad, M; epochs = FT_EPOCHS, seed = 1, freeze = nothing)
    opt = Optimisers.setup(Optimisers.Adam(1f-3), psd)
    rng = Xoshiro(seed); n = length(ytr); Md = M === nothing ? nothing : NamedTuple(k => cu(M[k]) for k in keys(M))
    for ep in 1:epochs
        perm = randperm(rng, n)
        for i in 1:BATCH:(n - BATCH + 1)
            idx = perm[i:i + BATCH - 1]
            g = masked(decayed(grad(psd, Xtr[:, idx], ytr[idx]), psd), Md)
            freeze === nothing || (g = freeze(g))
            opt, psd = Optimisers.update(opt, psd, g)
        end
    end
    psd
end

if ARGS[1] == "pretrain"
    psd = gdev(ps0); opt = Optimisers.setup(Optimisers.Adam(1f-3), psd); Y = onehot(ytr); rng = Xoshiro(Int(SEED))
    for ep in 1:5                                            # legacy: BP 5 epochs, Adam 1e-3, batch 128
        perm = randperm(rng, length(ytr))
        for i in 1:128:(length(ytr) - 127)
            idx = perm[i:i + 127]
            g = Zygote.gradient(p -> bp_loss(p, todev_phase(Xtr[:, idx]), cu(Y[:, idx])), psd)[1]
            global opt, psd = Optimisers.update(opt, psd, g)
        end
        @printf("pretrain epoch %d  EP-settle acc %.4f\n", ep, ep_acc(psd)); flush(stdout)
    end
    sm = length(ytr) < 60_000 ? "_smoke" : ""
    serialize(joinpath(OUT, "pretrain$(sm).jls"), cpu(psd))
    CSV.write(joinpath(OUT, "pretrain$(sm).csv"), DataFrame([(item = "LIEP-REGEN", arm = "bp_pretrain", model = "clean", seed = Int(SEED),
              metric = "acc_ep", value = ep_acc(psd)), (item = "LIEP-REGEN", arm = "bp_pretrain", model = "clean", seed = Int(SEED),
              metric = "acc_ff", value = ff_acc(psd))]))
    write_provenance(joinpath(OUT, "pretrain$(sm)"); seeds = [Int(SEED)], config = Dict("epochs" => 5, "lr" => 1e-3, "batch" => 128,
                                                                                       "n_train" => length(ytr)))
    println("DONE pretrain"); exit()
end
kind = ARGS[1]; level = parse(Float32, ARGS[2]); draw = parse(Int, ARGS[3])
ps = deserialize(joinpath(OUT, "pretrain" * (length(ytr) < 60_000 ? "_smoke" : "") * ".jls")); rng = Xoshiro(10_000 * draw + round(Int, 1000level))
"Damage (legacy functions, Gaussian made relative): returns damaged weights and the stuck-device mask (or nothing)."
function damage(ps, kind, level, rng)
    M = Dict{Symbol, Any}(); W = Dict{Symbol, Any}()
    for k in keys(ps)
        w = ps[k].weight
        if kind == "none"                                    # the undamaged control: the same fine-tune from the clean network
            W[k] = w
        elseif kind == "gaussian"                                # σ = level × this layer's weight RMS
            W[k] = w .+ (level * sqrt(mean(abs2, w))) .* randn(rng, Float32, size(w))
        elseif kind == "lognormal"
            W[k] = w .* exp.(level .* randn(rng, Float32, size(w)))
        else
            keep = Float32.(rand(rng, Float32, size(w)) .>= level); M[k] = keep
            stuck = kind == "stuck_zero" ? zero(w) : rand(rng, [1f0, -1f0], size(w)) .* maximum(abs, w)
            W[k] = w .* keep .+ (1f0 .- keep) .* stuck
        end
    end
    NamedTuple(k => merge(ps[k], (weight = W[k],)) for k in keys(ps)), (isempty(M) ? nothing : NamedTuple(M))
end
psdam, M = damage(ps, kind, level, rng)
rows = NamedTuple[]
row!(arm, metric, v) = push!(rows, (item = "LIEP-REGEN", arm = arm, model = "$(kind)_$(level)", seed = draw, metric = metric, value = Float64(v)))
row!("clean", "acc_ep", ep_acc(gdev(ps))); row!("damaged", "acc_ep", ep_acc(gdev(psdam)))
lockin = LockinEP(ε = 0.03f0, ω_p = 0.02f0, n_cycles = 4, T_warmup_cycles = 2, T_free = 200, dt = 0.5f0)
centred = StaticEP(β = 0.1f0, T_free = 200, T_nudge = 100, dt = 0.5f0, centered = true)
ep_grad(m) = (p, x, y) -> first(ep_gradient(m, chain, p, std_, todev(x), CodebookCost(codes_d, y)))
bp_grad = (p, x, y) -> (xd = todev_phase(x); Yd = cu(onehot(y));        # targets built outside the differentiated closure
                         Zygote.gradient(q -> bp_loss(q, xd, Yd), p)[1])
readout_grad = bp_grad                                       # backprop into layer 2 only: layer 1 frozen after decay
freeze_layer1(g) = merge(g, (layer_1 = NamedTuple(q => (g.layer_1[q] === nothing ? nothing : zero(g.layer_1[q])) for q in keys(g.layer_1)),))
t0 = time()
for (arm, gr) in (("lockin", ep_grad(lockin)), ("centred", ep_grad(centred)), ("bp", bp_grad), ("readout_only", readout_grad))
    t = time(); out = finetune(gdev(psdam), gr, M; seed = draw, freeze = arm == "readout_only" ? freeze_layer1 : nothing)
    a = ep_acc(out); row!(arm, "acc_ep", a); arm in ("bp", "readout_only") && row!(arm, "acc_ff", ff_acc(out))
    o = cpu(out)
    if M !== nothing                                         # the mask held: stuck devices still at their stuck values
        row!(arm, "stuck_drift", maximum(maximum(abs.((o[k].weight .- psdam[k].weight) .* (1f0 .- M[k]))) for k in keys(M)))
    end
    arm == "readout_only" && row!(arm, "layer1_drift", maximum(abs.(o.layer_1.weight .- psdam.layer_1.weight)))   # must be 0
    row!(arm, "seconds", time() - t); @printf("%s %.2f draw %d %-12s EP-settle acc %.4f (%.0f s)\n", kind, level, draw, arm, a, time() - t); flush(stdout)
end
tag = "$(kind)_$(ARGS[2])_d$(draw)" * (length(ytr) < 60_000 ? "_smoke" : "")
CSV.write(joinpath(OUT, "$tag.csv"), DataFrame(rows))
write_provenance(joinpath(OUT, tag); seeds = [draw], config = Dict("kind" => kind, "level" => level, "ft_epochs" => FT_EPOCHS, "batch" => BATCH))
println("DONE $tag ($(round(Int, time() - t0)) s)")
