# LIEP-REGEN stage 3c: the fidelity grid past ω_p 0.2, corrected against pre-fix lock-in, synthetic and real-data states;
# κ by the tail fit and by the legacy fit window. Usage (repo root): julia --project=. -t 40 …/regen/grid_ext.jl
# Writes results/grid_ext.csv (long format) + provenance. Settings and predictions: README.md.
include(joinpath(@__DIR__, "..", "common", "liep_lib.jl"))
include(joinpath(@__DIR__, "..", "common", "provenance.jl"))
using CSV, DataFrames, Printf, Serialization
BLAS.set_num_threads(1)
const OUT = joinpath(@__DIR__, "results"); const REPS = 8; const B = 32
const EPS = Float32[0.003, 0.01, 0.03, 0.1, 0.3]; const OMEGA = Float32[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
function make_input(seed, b = B)
    rng = Xoshiro(seed); pix = rand(rng, Float32, 784, b); pix[rand(rng, Float32, 784, b) .< 0.8f0] .= 0f0
    μ = mean(pix; dims = 1); σ = std(pix; dims = 1) .+ 1f-6
    Phase.(0.5f0 .* tanh.((pix .- μ) ./ σ))
end
_, ps7, _ = paper_net(7); codes8 = paper_codes(8)
synth = [(make_input(100 + r), CodebookCost(codes8, rand(Xoshiro(100 + r + 5_000), 1:10, B))) for r in 1:REPS]
_, ps42, _ = paper_net(42); codes42 = paper_codes(42)
pstr = deserialize(joinpath(OUT, "ckpt", "lockin_B_lr3e-3_s42_e8.jls"))
Xte, yte = fmnist(:test)
real_ = [(Xte[:, (r - 1) * B .+ (1:B)], CodebookCost(codes42, yte[(r - 1) * B .+ (1:B)])) for r in 1:REPS]
STATES = (synthetic = (ps7, synth, EPS), real_init = (ps42, real_, Float32[0.03]), real_trained = (pstr, real_, Float32[0.03]))
leg = CSV.read(joinpath(@__DIR__, "..", "adb", "legacy", "results", "e7_kappa", "kappa_ddbaa3d.csv"), DataFrame)
layer_w(ps, g, l) = (nw = [length(ps[k].weight) for k in keys(ps)]; nb = [length(ps[k].bias_real) for k in keys(ps)]; off = cumsum([0; nw .+ 2 .* nb]); g[off[l] .+ (1:nw[l])])

"Distance to the fixed point per step, from the zero state (Float64), and the slope of its log over a step range."
function trace(ps, x; dt = 0.5, T = 6000)
    θ = _Φparts(ps); z0 = ComplexF64.(PN._phase_input_to_complex(x)); b = size(z0, 2)
    run(f) = (zr = [zeros(size(t.W, 1), b) for t in θ]; zi = [zeros(size(t.W, 1), b) for t in θ];
              for k in 1:T; o = _Φ(zr, zi, θ, real.(z0), imag.(z0), IDEAL, dt); zr = [q[1] for q in o]; zi = [q[2] for q in o]; f(k, zr, zi); end; (zr, zi))
    zs = run((k, a, c) -> nothing); e = zeros(T)
    run((k, a, c) -> (e[k] = sqrt(sum(sum(abs2, a[l] .- zs[1][l]) + sum(abs2, c[l] .- zs[2][l]) for l in eachindex(a)))))
    return e
end
slope(e, sel; dt = 0.5) = length(sel) < 10 ? NaN : -([ones(length(sel)) (sel .* dt)] \ log.(e[sel]))[2]

rows = NamedTuple[]; lk = ReentrantLock(); t0 = time()
row!(state, arm, r, metric, v; eps = 0.0, omega = 0.0) = lock(lk) do
    push!(rows, (item = "LIEP-REGEN", arm = arm, model = String(state), seed = r, eps = Float64(eps), omega_p = Float64(omega), n_cyc = 4, metric = metric, value = Float64(v)))
end
for (st, (ps, draws, epss)) in pairs(STATES)
    refs = Vector{Any}(undef, REPS)
    Threads.@threads :dynamic for r in 1:REPS
        x, cost = draws[r]
        ge, info = exact_gradient(ps, x, cost)
        chain = first(paper_net(7))
        gc = first(ep_gradient(StaticEP(β = 0.005f0, T_free = 400, T_nudge = 200, dt = 0.5f0, centered = true), chain, ps, first(Lux.setup(Xoshiro(0), chain)[2:2]), x, cost))
        refs[r] = (exact = flatg(ge), centred = flatg(gc))
        e = trace(ps, x)
        row!(st, "kappa", r, "kappa_tail", slope(e, findall(v -> 1e-9 < v < 1e-3, e))); row!(st, "exact", r, "settle_res", info.settle_res)
        st === :synthetic && row!(st, "kappa", r, "kappa_legacy_window", slope(e, leg.fit_lo[r]:leg.fit_hi[r]))
    end
    @printf("%s references done (%.0f s)\n", st, time() - t0); flush(stdout)
    jobs = vec([(ε, ω, a, r) for ε in epss, ω in OMEGA, a in (:corrected, :prefix, :prefix_nodc), r in 1:REPS])
    sort!(jobs, by = j -> j[2])
    Threads.@threads :dynamic for (ε, ω, a, r) in jobs
        x, cost = draws[r]; p = Probe(ε = ε, ω_p = ω, n_cyc = 4, warm_c = 1, T_free = 200)
        g = flatg(a === :corrected ? est_lockin(ps, x, cost, p) : est_lockin_prefix(ps, x, cost, p; dc_complex = a === :prefix_nodc))
        for (tn, tg) in pairs(refs[r])
            row!(st, String(a), r, "cos_vs_$tn", cosine(g, tg); eps = ε, omega = ω)
            for l in 1:2; row!(st, String(a), r, "cos_l$(l)_vs_$tn", cosine(layer_w(ps, g, l), layer_w(ps, tg, l)); eps = ε, omega = ω); end
        end
        row!(st, String(a), r, "norm_ratio_exact", norm(Float64.(g)) / norm(Float64.(refs[r].exact)); eps = ε, omega = ω)
    end
    @printf("%s done (%.0f s)\n", st, time() - t0); flush(stdout)
    CSV.write(joinpath(OUT, "grid_ext.csv"), DataFrame(rows))
end
write_provenance(joinpath(OUT, "grid_ext"); seeds = collect(1:REPS), config = Dict("omega_p" => string(OMEGA), "eps" => string(EPS)))
println("DONE grid_ext ($(round(Int, time() - t0)) s)")
