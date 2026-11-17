# LIEP-BASE library: data, softmax regression (logistic regression and probes), backprop baselines (ReLU MLP, the
# paper's feedforward phasor arm from legacy e8_baselines.jl), and standard centred EP on the Hopfield energy
# (README "Method"). CPU, Float32.
include(joinpath(@__DIR__, "..", "common", "liep_lib.jl"))
include(joinpath(@__DIR__, "..", "common", "provenance.jl"))
using CSV, DataFrames, Optimisers, Printf, Serialization, Zygote
import PhasorNetworks: normalize_to_unit_circle

const SEEDS = [42, 123, 456]
const OUT = joinpath(@__DIR__, "results")

# ------------------------------------------------------------------ data
"Pixels in [0, 1] (784 × N) and labels 1..10."
function pixels(split)
    d = fashion_mnist_data(split)
    Float32.(reshape(d.features, 784, :)), Int.(d.targets) .+ 1
end
"Train/test (or the held-out split: first 50k / last 10k of the training set)."
function splits(Xtr, ytr, Xte, yte; val = false)
    val ? (Xtr[:, 1:50_000], ytr[1:50_000], Xtr[:, 50_001:end], ytr[50_001:end]) : (Xtr, ytr, Xte, yte)
end
"Per-feature standardisation with the training set's statistics."
function standardise(Xtr, Xte)
    μ = mean(Xtr; dims = 2); σ = std(Xtr; dims = 2) .+ 1f-6
    (Xtr .- μ) ./ σ, (Xte .- μ) ./ σ
end
"[cos θ; sin θ] of the paper's phase encoding (what the phasor network sees)."
phase_features(P) = (z = PN._phase_input_to_complex(P); Float32.(vcat(real.(z), imag.(z))))
onehot(y) = (Y = zeros(Float32, 10, length(y)); for (i, c) in enumerate(y); Y[c, i] = 1f0; end; Y)

# ------------------------------------------------------------------ softmax regression (closed-form gradient)
"Multinomial logistic regression, Adam, batch 256; weight decay `wd` on W. Returns (test accuracy, W, b)."
function softmax_regression(Xtr, ytr, Xte, yte; lr = 1f-3, wd = 0f0, epochs = 30, seed = 42, batch = 256)
    D = size(Xtr, 1); rng = Xoshiro(seed)
    p = (W = zeros(Float32, 10, D), b = zeros(Float32, 10))
    opt = Optimisers.setup(Optimisers.Adam(Float32(lr)), p)
    Y = onehot(ytr); n = length(ytr); idx = collect(1:n)
    for ep in 1:epochs
        shuffle!(rng, idx)
        for i in 1:batch:n
            b = idx[i:min(i + batch - 1, n)]; X = Xtr[:, b]
            L = p.W * X .+ p.b; L = L .- maximum(L; dims = 1); P = exp.(L); P ./= sum(P; dims = 1)
            G = (P .- Y[:, b]) ./ length(b)
            g = (W = G * transpose(X) .+ Float32(wd) .* p.W, b = vec(sum(G; dims = 2)))
            opt, p = Optimisers.update(opt, p, g)
        end
    end
    pred = [argmax(view(p.W * Xte .+ p.b, :, j)) for j in 1:size(Xte, 2)]
    return mean(pred .== yte), p.W, p.b
end

# ------------------------------------------------------------------ backprop baselines (legacy e8_baselines.jl)
logsoftmax_ce(L, Y) = (L = L .- maximum(L; dims = 1); -sum(Y .* (L .- log.(sum(exp.(L); dims = 1)))) / size(L, 2))
function build_relu(sizes, rng)
    layers = [Dense(sizes[i] => sizes[i + 1], i < length(sizes) - 1 ? relu : identity) for i in 1:length(sizes) - 1]
    m = Chain(layers...); ps, st = Lux.setup(rng, m); m, ps, st
end
function build_phasor(rng)                                   # 784→256→64, unit circle, bias, init × 0.4 (as paper_net)
    m = Chain(PhasorDense(784 => 256, normalize_to_unit_circle, use_bias = true),
              PhasorDense(256 => 64, normalize_to_unit_circle, use_bias = true))
    ps, st = Lux.setup(rng, m)
    ps = NamedTuple(k => merge(ps[k], (weight = 0.4f0 .* ps[k].weight,)) for k in keys(ps))
    m, ps, st
end
phasor_logits(m, ps, st, x, codes) = similarity_outer(ComplexF32.(angle_to_complex(first(Lux.apply(m, x, ps, st)))), codes)

"Backprop training, Adam, batch 256; evaluates at `evals` epochs; `ckpt(ep, ps)` saves. Returns rows of (epoch, acc)."
function train_bp(lossfn, accfn, ps, X, Y, n; epochs = 8, lr = 1f-3, seed = 42, evals = (0, 1, 4, 8), ckpt = nothing)
    opt = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps); idx = collect(1:n); out = Tuple{Int, Float64}[]
    0 in evals && (push!(out, (0, accfn(ps))); ckpt === nothing || ckpt(0, ps))
    for ep in 1:epochs
        shuffle!(Xoshiro(seed + 7919ep), idx)
        for i in 1:256:n
            b = idx[i:min(i + 255, n)]
            g = Zygote.gradient(p -> lossfn(p, X[:, b], Y[:, b]), ps)[1]
            opt, ps = Optimisers.update(opt, ps, g)
        end
        (ep in evals || ep == epochs) && (push!(out, (ep, accfn(ps))); ckpt === nothing || ckpt(ep, ps))
    end
    return out, ps
end

# ------------------------------------------------------------------ standard centred EP (Hopfield energy MLP)
"""Standard EP on E = ½‖h‖² + ½‖o‖² − hᵀW₁x − oᵀW₂h − b₁ᵀh − b₂ᵀo, h ∈ [0, 1], identity output, MSE cost.
Asynchronous exact sweeps: h ← σ(W₁x + W₂ᵀo + b₁), then o ← (W₂h + b₂ + βy)/(1 + β) (y one-hot; β = 0 free)."""
struct SEP
    W1::Matrix{Float32}; b1::Vector{Float32}; W2::Matrix{Float32}; b2::Vector{Float32}
end
σ_clip(x) = clamp(x, 0f0, 1f0)
σ_half(x) = clamp(x / 2f0, 0f0, 1f0)
function sep_init(H, rng; gain = 1f0)                      # Kaiming-uniform U(−c, c), c = gain·√(3/fan_in)
    c1 = gain * sqrt(3f0 / 784f0); c2 = gain * sqrt(3f0 / Float32(H))
    SEP(c1 .* (2 .* rand(rng, Float32, H, 784) .- 1), zeros(Float32, H), c2 .* (2 .* rand(rng, Float32, 10, H) .- 1), zeros(Float32, 10))
end
function sep_settle(m::SEP, x, Y, β, σ, T; h = nothing, o = nothing)
    h = h === nothing ? zeros(Float32, size(m.W1, 1), size(x, 2)) : copy(h)
    o = o === nothing ? zeros(Float32, 10, size(x, 2)) : copy(o)
    d1 = m.W1 * x .+ m.b1
    for _ in 1:T
        h = σ.(d1 .+ transpose(m.W2) * o)
        o = (m.W2 * h .+ m.b2 .+ (β == 0 ? 0f0 : β .* Y)) ./ (1f0 + β)
    end
    h, o
end
sep_predict(m::SEP, x, σ; T = 60) = (o = sep_settle(m, x, nothing, 0f0, σ, T)[2]; [argmax(view(o, :, j)) for j in 1:size(o, 2)])

"Centred EP training: SGD momentum 0.9, batch 128, rates (lr, lr/2) for (W₁, W₂) and biases alike. Returns rows."
function train_sep(H, σ, Xtr, ytr, Xte, yte; lr = 0.1f0, β = 0.5f0, T = 60, K = 15, epochs = 8, seed = 42,
                   evals = (0, 1, 4, 8))
    rng = Xoshiro(seed); m = sep_init(H, rng); Y = onehot(ytr); n = length(ytr); idx = collect(1:n)
    v = [zero(m.W1), zero(m.b1), zero(m.W2), zero(m.b2)]; rates = Float32[lr, lr, lr / 2, lr / 2]
    acc() = mean(sep_predict(m, Xte, σ; T = T) .== yte)
    out = Tuple{Int, Float64}[]; 0 in evals && push!(out, (0, acc()))
    for ep in 1:epochs
        shuffle!(rng, idx)
        for i in 1:128:n
            b = idx[i:min(i + 127, n)]; x = Xtr[:, b]; y = Y[:, b]; B = Float32(length(b))
            hf, of = sep_settle(m, x, y, 0f0, σ, T)
            hp, op = sep_settle(m, x, y, β, σ, K; h = hf, o = of)
            hm, om = sep_settle(m, x, y, -β, σ, K; h = hf, o = of)
            # −∂L/∂θ ≈ [∂(−E)/∂θ(s₊) − ∂(−E)/∂θ(s₋)]/(2β): Hebbian differences
            g = [(hp * transpose(x) .- hm * transpose(x)) ./ (2β * B), vec(sum(hp .- hm; dims = 2)) ./ (2β * B),
                 (op * transpose(hp) .- om * transpose(hm)) ./ (2β * B), vec(sum(op .- om; dims = 2)) ./ (2β * B)]
            for (j, P) in enumerate((m.W1, m.b1, m.W2, m.b2))
                v[j] .= 0.9f0 .* v[j] .+ g[j]; P .+= rates[j] .* v[j]
            end
        end
        (ep in evals || ep == epochs) && push!(out, (ep, acc()))
    end
    return out
end
