# LIEP revision harness: the paper's network, four EP estimators sharing one step, impairment hooks, and an exact
# gradient reference. Card LIEP-PORT (experiments/port/README.md); used by LIEP-GF/RDR/NT/UPD/REGEN.
#
# Why a private step (as T2's t2_lib.jl): the library's `phasor_settle` / `ep_gradient(::LockinEP)` have no hook for
# noise inside the dynamics or on the measured Hebbian. `step` below is the library's `_phasor_step` for PhasorDense
# chains (K_mode = :zero, hard projection, co-rotating frame, Jacobi), and `est_lockin` mirrors the corrected
# `ep_gradient(::LockinEP)` of library main (real reference cos(ω_p t), d35fc07): same probe and demodulator phases,
# DC subtraction and −2Re(H)/(T·ε) normalisation. At zero impairment both are checked against the library (port/gates.jl).
#
# Arrays are generic: pass `to = cu` (CUDA) to `Ctx` and the same code runs on the GPU.
#
# Impairments (all off by default; the cards that use them define their levels):
#   Dyn (inside the settle, every arm sees the same statistics per step)
#     ξ        phase diffusion, rad/√time: z ← z·cis(ξ√dt·N(0,1)) per step
#     σw       synaptic read noise, multiplicative, redrawn every step: W_t = W∘(1 + σw·N)
#     δω, trim static per-neuron detuning and its trim, rad/time, per layer: z ← z·cis((δω + trim)·dt) per step
#     gain     per-unit emission gain a_i (amplitude variation): other layers, and the Hebbians, see a∘z
#     iso      non-isochrony β: z ← z·cis(β·(|g| − r_ref)·dt) per step (amplitude-dependent frequency; |g| = drive);
#              iso and r_ref are numbers or per-layer vectors
#     readout_δ, readout_jitter  spike-time quantiser of every state read (library `_readout`), turns
#     emit_c   per-unit complex emission factor (a trim: `trim_factors`); other layers and the Hebbians see c∘z
#     wdrift, gdrift  slow linear drift during the estimate: W(t) = W∘(1 + wdrift·t·N), a(t) = a∘(1 + gdrift·t·N),
#              t the time since the estimate began (the Ctx's own clock), N a fixed draw (drift_seed)
#     fdrift, fpink  time-varying detuning (oscillator frequency wander), per unit, restarted per estimate, on top of δω:
#              z ← z·cis(ω_i(t)·dt) per step with ω_i(t) = fdrift·t·N_i (N per unit, fixed by drift_seed; rad/time²) plus
#              fpink·x_i(t), x a unit-RMS 1/f process (fpink_n Ornstein–Uhlenbeck processes, τ = fpink_τ·4^j, drawn
#              stationary at the estimate's first step from the dynamics rng). Not trimmed: a trim corrects δω only
#   Reader (on each Hebbian sample, before accumulation; forces the per-step form of every layer)
#     σh       additive noise on the sampled Hebbian (white unless `noise` supplies a process: (rng, H, t) -> array
#              like H, t the absolute time, so a coloured process advances between samples; see `PinkNoise`)
#     off_in   offset added to the Hebbian before the reference multiply: a number, or l -> array (per synapse)
#     off_prod offset added after the multiply (a multiplier offset): a number, or l -> array
#     phase    reference phase error φ(t) (a function of the sample index), lock-in only
#     bits     ADC on the sampled Hebbian (0 = exact), full scale ±fs
#     hp_τ     first-order high-pass (AC coupling) on the Hebbian, time constant in time units (0 = off)
using PhasorNetworks, Lux, LinearAlgebra, Random, Statistics
using Random: Xoshiro
const PN = PhasorNetworks

# ------------------------------------------------------------------ the paper's network and data
"784→256→64 PhasorDense chain as the paper (e6_basin_training.jl build_chain): unit circle, complex bias, ×0.4."
function paper_net(seed::Integer; dims = (784, 256, 64), scale = 0.4f0)
    rng = Xoshiro(seed)
    layers = [PhasorDense(dims[i] => dims[i + 1], normalize_to_unit_circle, use_bias = true) for i in 1:length(dims) - 1]
    chain = Chain(layers...)
    ps, st = Lux.setup(rng, chain)
    ps = NamedTuple(k => merge(ps[k], (weight = scale .* ps[k].weight,)) for k in keys(ps))
    return chain, ps, st
end

"Paper encoding: Phase(0.5·tanh(per-image standardised pixels)) (half-plane range)."
function encode_phase(imgs::AbstractArray{Float32,3})
    flat = reshape(imgs, :, size(imgs, 3))
    mu = mean(flat; dims = 1); sd = std(flat; dims = 1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((flat .- mu) ./ sd))
end

"FashionMNIST split, encoded; labels 1..10."
function fmnist(split::Symbol, n::Int = typemax(Int))
    d = fashion_mnist_data(split)
    n = min(n, length(d.targets))
    return encode_phase(Float32.(d.features[:, :, 1:n])), Int.(d.targets[1:n]) .+ 1
end

paper_codes(seed::Integer; dout = 64) = ComplexF32.(angle_to_complex(orthogonal_codes(Xoshiro(seed), dout, 10)))

# ------------------------------------------------------------------ impairment models
Base.@kwdef struct Dyn
    ξ::Float32 = 0f0
    σw::Float32 = 0f0
    δω::Union{Nothing, Vector{Vector{Float32}}} = nothing
    trim::Union{Nothing, Vector{Vector{Float32}}} = nothing
    gain::Union{Nothing, Vector{Vector{Float32}}} = nothing
    iso::Any = 0f0
    r_ref::Any = 1f0
    readout_δ::Float32 = 0f0
    readout_jitter::Float32 = 0f0
    emit_c::Union{Nothing, Vector{Vector{ComplexF32}}} = nothing
    wdrift::Float32 = 0f0
    gdrift::Float32 = 0f0
    drift_seed::Int = 0
    fdrift::Float32 = 0f0
    fpink::Float32 = 0f0
    fpink_τ::Float64 = 0.5
    fpink_n::Int = 7
    fpink_carry::Any = nothing          # a Vector{Any} (one entry per layer) holding the OU state: shared by every Ctx built from this Dyn,
                                        # so the 1/f process persists across estimates and evaluation (one process per device); nothing = fresh per Ctx
end
const IDEAL = Dyn()

Base.@kwdef struct Reader
    σh::Float32 = 0f0
    noise::Any = nothing            # (rng, H, t) -> array like H; overrides white noise of size σh (t = absolute time)
    off_in::Any = 0f0               # number, or l -> array of H's size (static per-synapse offsets)
    off_prod::Any = 0f0
    phase::Any = nothing            # k -> reference phase error (rad), lock-in
    bits::Int = 0
    fs::Float32 = 1f0
    hp_τ::Float32 = 0f0
    force_full::Bool = false        # per-step layer-1 product even when the shortcut is exact (gate check)
    stats::Any = nothing            # a `ReadStats`: RMS and peak of the multiplier input, per layer
end

"Running RMS and peak of |multiplier input| (real and imaginary parts pooled), per layer."
mutable struct ReadStats
    ss::Vector{Float64}; n::Vector{Int}; peak::Vector{Float64}
end
ReadStats(nl::Int = 2) = ReadStats(zeros(nl), zeros(Int, nl), zeros(nl))
function _stat!(st::ReadStats, H, l)
    st.ss[l] += sum(abs2, real.(H)) + sum(abs2, imag.(H)); st.n[l] += 2length(H)
    st.peak[l] = max(st.peak[l], maximum(abs, real.(H)), maximum(abs, imag.(H)))
end
stat_rms(st::ReadStats, l) = sqrt(st.ss[l] / max(st.n[l], 1))
const EXACT = Reader()
_zero_off(o) = o isa Real && o == 0
is_exact(r::Reader) = r.σh == 0 && r.noise === nothing && _zero_off(r.off_in) && _zero_off(r.off_prod) &&
                      r.phase === nothing && r.bits == 0 && r.hp_τ == 0 && r.stats === nothing
is_ideal_dyn(d::Dyn) = d.ξ == 0 && d.σw == 0 && d.δω === nothing && d.trim === nothing && d.gain === nothing &&
                       !_has_iso(d) && d.readout_δ == 0 && d.readout_jitter == 0 && d.emit_c === nothing &&
                       d.wdrift == 0 && d.gdrift == 0 && d.fdrift == 0 && d.fpink == 0

_lv(x, l) = x isa Real ? Float32(x) : Float32(x[l])
_has_iso(d::Dyn) = d.iso isa Real ? d.iso != 0 : any(!=(0), d.iso)

"Drift directions (N(0,1), fixed by drift_seed): per-weight for wdrift, per-unit for gdrift; nothing when off."
function drift_dirs(dyn::Dyn, W)
    dr = Xoshiro(90_000 + dyn.drift_seed)
    Wdir = Any[dyn.wdrift == 0 ? nothing : randn(dr, Float32, size(w)) for w in W]
    gdir = Any[dyn.gdrift == 0 ? nothing : randn(dr, Float32, size(w, 1)) for w in W]
    return Wdir, gdir
end

"Per-neuron N(0, σ²) draw per layer (detuning, trims, gains), one realisation per device seed."
draw_per_neuron(dims, σ::Real, seed::Int; mean = 0f0) =
    [Float32(mean) .+ Float32(σ) .* randn(Xoshiro(50_000 + seed + 7919l), Float32, dims[l + 1]) for l in 1:length(dims) - 1]

# ------------------------------------------------------------------ context and step
struct Ctx{M, MC, V, Z}
    W::Vector{M}                    # real weights (out, in)
    Wc::Vector{MC}                  # complex copies (cgemm), drive
    Wt::Vector{MC}                  # complex transposes, feedback
    b::Vector{V}                    # complex biases
    z0::Z                           # input on the unit circle (in, B)
    drive0::Z                       # W₁z₀ + b₁ (constant unless σw > 0)
    cost::Any
    dyn::Dyn
    rot::Vector{Any}                # per-layer cis((δω + trim)·dt) or nothing
    gain::Vector{Any}
    ec::Vector{Any}                 # per-layer complex emission factor or nothing
    Wdir::Vector{Any}               # drift directions (or nothing)
    gdir::Vector{Any}
    fdir::Vector{Any}               # per-unit frequency-drift directions (or nothing)
    fou::Vector{Any}                # per layer: the fpink OU states (a vector of per-unit arrays), set at the first step
    clock::Base.RefValue{Int}       # steps taken since the estimate began
    dt::Float32
    to::Any                         # array constructor (identity or cu)
end

_dev(to, x) = to === identity ? x : to(x)

function Ctx(ps, x, cost, dyn::Dyn, dt::Real; to = identity)
    ks = collect(keys(ps)); n = length(ks); dt = Float32(dt)
    W  = [_dev(to, Matrix{Float32}(ps[k].weight)) for k in ks]
    Wc = [ComplexF32.(w) for w in W]
    Wt = [ComplexF32.(transpose(w)) for w in W]
    b  = [_dev(to, ComplexF32.(Vector(ps[k].bias_real) .+ 1f0im .* Vector(ps[k].bias_imag))) for k in ks]
    z0 = _dev(to, Matrix{ComplexF32}(PN._phase_input_to_complex(x isa AbstractVector ? reshape(x, :, 1) : x)))
    rot = Any[nothing for _ in 1:n]
    if dyn.δω !== nothing || dyn.trim !== nothing
        for l in 1:n
            w = (dyn.δω === nothing ? 0f0 : dyn.δω[l]) .+ (dyn.trim === nothing ? 0f0 : dyn.trim[l])
            rot[l] = _dev(to, ComplexF32.(cis.(Float64.(w) .* dt)))
        end
    end
    gain = Any[dyn.gain === nothing ? nothing : _dev(to, dyn.gain[l]) for l in 1:n]
    ec = Any[dyn.emit_c === nothing ? nothing : _dev(to, dyn.emit_c[l]) for l in 1:n]
    Wdir, gdir = drift_dirs(dyn, [Array(w) for w in W])
    Wdir = Any[w === nothing ? nothing : _dev(to, w) for w in Wdir]; gdir = Any[g === nothing ? nothing : _dev(to, g) for g in gdir]
    fdir = Any[nothing for _ in 1:n]
    if dyn.fdrift != 0
        fr = Xoshiro(95_000 + dyn.drift_seed)
        fdir = Any[_dev(to, randn(fr, Float32, size(w, 1))) for w in W]
    end
    d0 = Wc[1] * z0 .+ b[1]
    return Ctx(W, Wc, Wt, b, z0, d0, cost, dyn, rot, gain, ec, Wdir, gdir, fdir, dyn.fpink_carry === nothing ? Any[nothing for _ in 1:n] : dyn.fpink_carry, Ref(0), dt, to)
end

init_states(c::Ctx) = [fill!(similar(c.z0, size(w, 1), size(c.z0, 2)), zero(ComplexF32)) for w in c.W]

_randn(rng, A) = A isa Array ? randn(rng, eltype(A), size(A)) : randn!(similar(A))   # GPU: device RNG
elapsed(c::Ctx) = c.clock[] * c.dt
"Time-varying detuning of layer l at this step (per-unit vector, rad/time), advancing the fpink state; nothing when off."
function fwander!(c::Ctx, l, rng)
    d = c.dyn; (d.fdrift == 0f0 && d.fpink == 0f0) && return nothing
    w = d.fdrift == 0f0 ? nothing : (d.fdrift * Float32(elapsed(c))) .* c.fdir[l]
    if d.fpink != 0f0
        tmpl = real.(c.b[l])
        if c.fou[l] === nothing
            c.fou[l] = Any[_randn(rng, tmpl) for _ in 1:d.fpink_n]          # stationary start, fresh per estimate
        else
            for (j, x) in enumerate(c.fou[l])
                a = Float32(exp(-c.dt / (d.fpink_τ * 4.0^(j - 1))))
                x .= a .* x .+ sqrt(1f0 - a^2) .* _randn(rng, tmpl)
            end
        end
        p = (d.fpink / sqrt(Float32(d.fpink_n))) .* sum(c.fou[l])
        w = w === nothing ? p : w .+ p
    end
    return w
end
"Per-unit emission factor of layer l (gain, its drift, the complex trim), or nothing when there is none."
function efac(c::Ctx, l)
    c.gain[l] === nothing && c.gdir[l] === nothing && c.ec[l] === nothing && return nothing
    f = c.gain[l] === nothing ? one(ComplexF32) : ComplexF32.(c.gain[l])
    c.gdir[l] === nothing || (f = f .* (1f0 .+ (c.dyn.gdrift * Float32(elapsed(c))) .* c.gdir[l]))
    c.ec[l] === nothing || (f = f .* c.ec[l])
    return f
end
emit(c::Ctx, Z, l) = (f = efac(c, l); f === nothing ? Z[l] : f .* Z[l])
# The cost reads the emitted output e = f∘z, and the nudge is a signal injected into the output units' drive: g + F(e).
# (Not conj(f)∘F: a trimmed unit's fixed point satisfies e = u(g) in the emitted frame, so the injected force must
# enter unrotated for the nudged fixed points to be the nominal ones.)
function nudge(c::Ctx, Z, β)
    n = length(Z); f = efac(c, n)
    f === nothing && return PN.nudge_force(c.cost, Z[n], β)
    return PN.nudge_force(c.cost, f .* Z[n], β)
end

"One Jacobi step of the projected damped dynamics with nudge β: library `_phasor_step` plus the Dyn hooks."
function step(c::Ctx, Z, β::Float32, rng)
    n = length(Z); dt = c.dt; d = c.dyn
    fresh = d.σw > 0f0 || d.wdrift != 0f0
    if fresh
        τ = Float32(elapsed(c))
        Wn = [ComplexF32.(w .* (d.wdrift == 0f0 ? 1f0 : 1f0 .+ (d.wdrift * τ) .* c.Wdir[l]) .*
                          (d.σw > 0f0 ? 1f0 .+ d.σw .* _randn(rng, w) : 1f0)) for (l, w) in enumerate(c.W)]
        Wd, Wf = Wn, [transpose(w) for w in Wn]
    else
        Wd, Wf = c.Wc, c.Wt
    end
    out = map(1:n) do l
        g = l == 1 ? (fresh ? Wd[1] * c.z0 .+ c.b[1] : c.drive0) : Wd[l] * emit(c, Z, l - 1) .+ c.b[l]
        l < n && (g = g .+ Wf[l + 1] * emit(c, Z, l + 1))
        (l == n && β != 0f0) && (g = g .+ nudge(c, Z, β))
        zn = PN._project_damp.(Z[l], g, dt, 1f-10)
        c.rot[l] === nothing || (zn = zn .* c.rot[l])
        fw = fwander!(c, l, rng); fw === nothing || (zn = zn .* cis.(fw .* dt))
        _has_iso(d) && (zn = zn .* cis.(_lv(d.iso, l) .* (abs.(g) .- _lv(d.r_ref, l)) .* dt))
        d.ξ > 0f0 && (zn = zn .* cis.(d.ξ * sqrt(dt) .* _randn(rng, real.(zn))))
        zn
    end
    c.clock[] += 1
    return out
end

function settle(c::Ctx, β::Real, T::Int, rng; init = nothing)
    Z = init === nothing ? init_states(c) : [copy(z) for z in init]
    for _ in 1:T
        Z = step(c, Z, Float32(β), rng)
    end
    return Z
end

# state read through the spike-time quantiser (library `_readout`)
_ro(c::Ctx, z, rng) = (c.dyn.readout_δ <= 0f0 && c.dyn.readout_jitter <= 0f0) ? z :
    PN._readout(z, c.dyn.readout_δ, c.dyn.readout_jitter,
                c.dyn.readout_jitter > 0f0 ? c.dyn.readout_jitter .* _randn(rng, real.(z)) : nothing)
observe(c::Ctx, Z, rng) = [_ro(c, emit(c, Z, l), rng) for l in eachindex(Z)]
input_obs(c::Ctx, rng) = _ro(c, c.z0, rng)

# ------------------------------------------------------------------ Hebbians and gradients
_B(z) = size(z, 2)
_sumb(z) = vec(sum(z; dims = 2))

"Library convention (ep_hebbian): weight = z_self z_inᴴ / B (complex; Re is the Hebbian), bias = Σ_b z_self / B."
hebbians(obs0, obs) = [(W = (obs[l] * adjoint(l == 1 ? obs0 : obs[l - 1])) ./ _B(obs[l]),
                        b = _sumb(obs[l]) ./ _B(obs[l])) for l in eachindex(obs)]

# dynamics parameters (log_neg_lambda) are not EP-trained: zero slot, as the library's `_pad_dynamics_zeros`
_pad(p, e) = haskey(p, :log_neg_lambda) ? merge(e, (log_neg_lambda = zero(p.log_neg_lambda),)) : e

"Gradient shaped like ps (on the CPU) from complex Hebbians: weight = s·Re(HW), bias_real/imag = s·Re/Im(Hb)."
function assemble(ps, HW, Hb, s::Real)
    ks = collect(keys(ps))
    NamedTuple(k => _pad(ps[k], (weight = Float32(s) .* Array(real.(HW[l])),
                                 bias_real = Float32(s) .* Array(real.(Hb[l])),
                                 bias_imag = Float32(s) .* Array(imag.(Hb[l])))) for (l, k) in enumerate(ks))
end

flatg(g) = reduce(vcat, [vec(Float32.(g[k][p])) for k in keys(g) for p in (:weight, :bias_real, :bias_imag)])
cosine(a, b) = (a = Float64.(a); b = Float64.(b); dot(a, b) / (norm(a) * norm(b) + 1e-300))
relerr(a, b) = (a = Float64.(a); b = Float64.(b); norm(a .- b) / (norm(b) + 1e-300))

# Weighted accumulator: S_l = Σ_k w_k·obs_l(k), M_l = Σ_k w_k·obs_l(k)·obs_{l-1}(k)ᴴ, through the Reader.
# Exact reader + exact input: layer 1's M is (S_1)·z₀ᴴ at the end (z₀ is clamped), not a product per step.
mutable struct Acc
    S::Vector{Any}
    M::Vector{Any}
    hp::Vector{Any}                 # high-pass state per layer (M-shaped) or nothing
    wsum::ComplexF32
    k::Int
    fast1::Bool
end
function Acc(c::Ctx, Z, reader::Reader; T = ComplexF32)
    # the input read is the same every step unless the readout jitters; layer 1's own read may vary (it is in S_1)
    fast1 = is_exact(reader) && c.dyn.readout_jitter <= 0f0 && !reader.force_full
    S = Any[fill!(similar(z, T), 0) for z in Z]
    M = Any[(l == 1 && fast1) ? nothing : fill!(similar(c.Wc[l], T), 0) for l in eachindex(Z)]
    Acc(S, M, Any[nothing for _ in Z], zero(ComplexF32), 0, fast1)
end

_quant(x, bits, fs) = (q = 2fs / (2^bits); clamp.(round.(x ./ q) .* q, -fs, fs))
_off(o, l) = o isa Real ? Float32(o) : o(l)
# AC coupling: first-order high-pass, state started at the first sample (no turn-on transient)
function _hp(r::Reader, H, hp, dts)
    hp === nothing && return zero(H), copy(H)
    hp .+= (dts / r.hp_τ) .* (H .- hp)
    return H .- hp, hp
end
function _read(r::Reader, H, ref::Float32, t, rng, hp, dts, l)
    r.hp_τ > 0f0 && ((H, hp) = _hp(r, H, hp, dts))
    _zero_off(r.off_in) || (H = H .+ _off(r.off_in, l))
    if r.noise !== nothing
        H = H .+ r.noise(rng, H, t)                                  # H: a template (size and device) for the process
    elseif r.σh > 0f0
        H = H .+ r.σh .* (_randn(rng, real.(H)) .+ 1f0im .* _randn(rng, real.(H)))
    end
    r.bits > 0 && (H = complex.(_quant(real.(H), r.bits, r.fs), _quant(imag.(H), r.bits, r.fs)))
    r.stats === nothing || _stat!(r.stats, H, l)
    P = ref .* H
    _zero_off(r.off_prod) || (P = P .+ _off(r.off_prod, l))
    return P, hp
end

function accumulate!(a::Acc, c::Ctx, obs0, obs, w::Float32, reader::Reader, rng; dts = c.dt)
    a.wsum += w; a.k += 1
    exact = is_exact(reader)
    for l in eachindex(obs)
        a.S[l] .+= obs[l] .* w
        (l == 1 && a.fast1) && continue
        zin = l == 1 ? obs0 : obs[l - 1]
        if exact
            if eltype(a.M[l]) === ComplexF32
                mul!(a.M[l], obs[l], adjoint(zin), ComplexF32(w), one(ComplexF32))
            else
                a.M[l] .+= w .* (ComplexF64.(obs[l]) * adjoint(ComplexF64.(zin)))
            end
        else
            H = (obs[l] * adjoint(zin)) ./ _B(obs[l])                   # the synapse's sample (batch mean)
            P, a.hp[l] = _read(reader, H, w, elapsed(c), rng, a.hp[l], dts, l)
            a.M[l] .+= P .* _B(obs[l])                                  # undo /B: closed out with the exact form
        end
    end
    return a
end

"Advance the AC-coupling state on a sample that is not accumulated (static_sq's discarded transients)."
function hp_only!(a::Acc, c::Ctx, obs0, obs, reader::Reader; dts = c.dt)
    reader.hp_τ > 0f0 || return a
    for l in eachindex(obs)
        H = (obs[l] * adjoint(l == 1 ? obs0 : obs[l - 1])) ./ _B(obs[l])
        _, a.hp[l] = _hp(reader, H, a.hp[l], dts)
    end
    return a
end

"Close an accumulator into per-layer complex sums (Σ w·H) per batch element: HW_l, Hb_l (both /B)."
function close_acc(a::Acc, c::Ctx, obs0)
    T = eltype(a.S[1])
    HW = [((l == 1 && a.fast1) ? a.S[1] * adjoint(T.(obs0)) : a.M[l]) ./ _B(a.S[l]) for l in eachindex(a.S)]
    Hb = [_sumb(a.S[l]) ./ _B(a.S[l]) for l in eachindex(a.S)]
    return HW, Hb
end

# ------------------------------------------------------------------ estimators
Base.@kwdef struct Probe
    ε::Float32 = 0.03f0             # probe amplitude (lock-in, static_sq) or β (centred, classic)
    ω_p::Float32 = 0.02f0
    dt::Float32 = 0.5f0
    n_cyc::Int = 4
    warm_c::Int = 1
    T_free::Int = 200
    T_nudge::Int = 100              # centred / classic: steps of each nudged settle
    se::Int = 1                     # sample every se steps
end
period_steps(p::Probe) = round(Int, 2π / (p.ω_p * p.dt))
lockin_steps(p::Probe) = p.T_free + (p.warm_c + p.n_cyc) * period_steps(p)

"Corrected lock-in EP (library main `ep_gradient(::LockinEP)`), with the Dyn and Reader hooks."
# `clock`: t -> phase error (rad) of the probe generator at step t (the reference's own error is `reader.phase`)
function est_lockin(ps, x, cost, p::Probe; dyn = IDEAL, reader = EXACT, rng = Xoshiro(0), to = identity,
                    acc = ComplexF64,   # Float64 accumulators: the library's Float32 sum is off by ~1e-3 (port FINDINGS)
                    clock = nothing)
    c = Ctx(ps, x, cost, dyn, p.dt; to = to)
    Z = settle(c, 0f0, p.T_free, rng)
    o0 = input_obs(c, rng)
    # the free-state (DC) Hebbian is measured through the same reader as the demodulated sum, as hardware would
    hdc = is_exact(reader) ? hebbians(o0, observe(c, Z, rng)) : _snap(c, Z, reader, rng)
    P = period_steps(p)
    # the probe phase restarts after the warm-up, as the library; the clock error runs on absolute steps
    nud(t, tabs) = p.ε * cos(p.ω_p * t * p.dt + (clock === nothing ? 0f0 : Float32(clock(tabs))))
    for t in 1:(p.warm_c * P)
        Z = step(c, Z, nud(t, t), rng)
    end
    a = Acc(c, Z, reader; T = acc)
    T_eff = 0
    for t in 1:(p.n_cyc * P)
        Z = step(c, Z, nud(t, p.warm_c * P + t), rng)
        (t - 1) % p.se == 0 || continue
        te = div(t - 1, p.se)
        φ = reader.phase === nothing ? 0f0 : Float32(reader.phase(te))
        w = cos(p.ω_p * Float32(te * p.se + 1) * p.dt + φ)
        accumulate!(a, c, input_obs(c, rng), observe(c, Z, rng), w, reader, rng; dts = p.se * p.dt)
        T_eff += 1
    end
    HW, Hb = close_acc(a, c, o0)
    cw = a.wsum
    HWn = [HW[l] .- cw .* real.(hdc[l].W) for l in eachindex(HW)]
    Hbn = [Hb[l] .- cw .* hdc[l].b for l in eachindex(Hb)]
    return assemble(ps, HWn, Hbn, -2f0 / (T_eff * p.ε))
end

"""The pre-fix lock-in estimator (library before d35fc07), for controls only: demodulates the complex Hebbian by
e^{-iω_p t} and keeps its real part, so Im H·sin(ω_p t) leaks into each gradient. Float32 sums, as the library then.
`dc_complex = true` also subtracts the free state's Im H, removing the static part of the leak (Im H_dc·Σsin, scaled by
1/ε) and leaving only the dynamic one."""
function est_lockin_prefix(ps, x, cost, p::Probe; rng = Xoshiro(0), to = identity, T = ComplexF32, dc_complex = false)
    c = Ctx(ps, x, cost, IDEAL, p.dt; to = to)
    Z = settle(c, 0f0, p.T_free, rng)
    o0 = c.z0; hdc = hebbians(o0, Z); P = period_steps(p)
    for t in 1:(p.warm_c * P)
        Z = step(c, Z, p.ε * cos(p.ω_p * t * p.dt), rng)
    end
    S = [fill!(similar(z, T), 0) for z in Z]; M = [fill!(similar(c.Wc[l], T), 0) for l in eachindex(Z)]; w = zero(T); n = p.n_cyc * P
    for t in 1:n
        Z = step(c, Z, p.ε * cos(p.ω_p * t * p.dt), rng)
        d = T(cis(-p.ω_p * Float32(t) * p.dt)); w += d
        for l in eachindex(Z)
            S[l] .+= Z[l] .* d
            l > 1 && mul!(M[l], Z[l], adjoint(Z[l - 1]), d, one(T))
        end
    end
    B = _B(Z[1]); HW = [((l == 1 ? S[1] * adjoint(T.(o0)) : M[l]) ./ B) .- w .* (dc_complex ? hdc[l].W : real.(hdc[l].W)) for l in eachindex(Z)]
    Hb = [(_sumb(S[l]) ./ B) .- w .* hdc[l].b for l in eachindex(Z)]
    return assemble(ps, HW, Hb, -2f0 / (n * p.ε))
end

"Snapshot Hebbians of a settled state, through the Reader (reference +1)."
function _snap(c::Ctx, Z, reader::Reader, rng)
    o0 = input_obs(c, rng); obs = observe(c, Z, rng)
    is_exact(reader) && return hebbians(o0, obs)
    a = Acc(c, Z, reader); accumulate!(a, c, o0, obs, 1f0, reader, rng)
    HW, Hb = close_acc(a, c, o0)
    return [(W = HW[l], b = Hb[l]) for l in eachindex(HW)]
end
_re(h) = [(W = ComplexF32.(real.(x.W)), b = x.b) for x in h]     # library: weight Hebbian is real

"""Settle T steps at nudge β from `init`; return the final state and the Hebbians through the reader: a snapshot of
the final state (window = 0, the library) or the average over the last `window` fraction of the steps, sampled
every p.se steps (a hardware integrator)."""
function _settle_read(c::Ctx, β, T::Int, rng, init, reader::Reader, window, se)
    window <= 0 && (Z = settle(c, β, T, rng; init = init); return Z, _re(_snap(c, Z, reader, rng)))
    reader.hp_τ > 0f0 && error("AC coupling removes a static phase's signal; not defined for centred/classic")
    Z = [copy(z) for z in init]
    t0 = T - max(1, round(Int, window * T))
    a = nothing
    for t in 1:T
        Z = step(c, Z, Float32(β), rng)
        (t > t0 && (t - t0 - 1) % se == 0) || continue
        a === nothing && (a = Acc(c, Z, reader; T = ComplexF64))
        accumulate!(a, c, input_obs(c, rng), observe(c, Z, rng), 1f0, reader, rng; dts = se * c.dt)
    end
    HW, Hb = close_acc(a, c, input_obs(c, rng))
    return Z, [(W = HW[l] ./ a.k, b = Hb[l] ./ a.k) for l in eachindex(HW)]
end

"""Centred static EP (library StaticEP(centered = true)): −(H₊ − H₋)/(2β), each nudged settle from the free state
(in time order free, +β, −β on the Ctx's clock). `window` > 0 averages each nudged Hebbian over the end of its settle."""
function est_centred(ps, x, cost, p::Probe; dyn = IDEAL, reader = EXACT, rng = Xoshiro(0), to = identity, window = 0.0)
    c = Ctx(ps, x, cost, dyn, p.dt; to = to)
    Zf = settle(c, 0f0, p.T_free, rng)
    _, hp = _settle_read(c, p.ε, p.T_nudge, rng, Zf, reader, window, p.se)
    _, hm = _settle_read(c, -p.ε, p.T_nudge, rng, Zf, reader, window, p.se)
    return assemble(ps, [hp[l].W .- hm[l].W for l in eachindex(hp)], [hp[l].b .- hm[l].b for l in eachindex(hp)],
                    -1f0 / (2f0 * p.ε))
end

"Classic one-sided EP (library StaticEP()): −(H_β − H_free)/β; `window` > 0 averages both phases' Hebbians."
function est_classic(ps, x, cost, p::Probe; dyn = IDEAL, reader = EXACT, rng = Xoshiro(0), to = identity, window = 0.0)
    c = Ctx(ps, x, cost, dyn, p.dt; to = to)
    Zf, hf = _settle_read(c, 0f0, p.T_free, rng, init_states(c), reader, window, p.se)
    _, hp = _settle_read(c, p.ε, p.T_nudge, rng, Zf, reader, window, p.se)
    return assemble(ps, [hp[l].W .- hf[l].W for l in eachindex(hp)], [hp[l].b .- hf[l].b for l in eachindex(hp)],
                    -1f0 / p.ε)
end

"""Square-wave averaged static EP (T2's static_sq): free settle T_free + warm-up, then n_cyc × (+ε, −ε) blocks of P/2
steps, the first `discard` of each block dropped, Hebbians averaged per sign. Same total steps as the lock-in.
With AC coupling one high-pass runs on the continuous Hebbian stream (both signs, discarded samples included)."""
function est_static_sq(ps, x, cost, p::Probe; dyn = IDEAL, reader = EXACT, rng = Xoshiro(0), to = identity,
                       discard = 0.5, acc = ComplexF64)
    c = Ctx(ps, x, cost, dyn, p.dt; to = to)
    P = period_steps(p)
    Z = settle(c, 0f0, p.T_free + p.warm_c * P, rng)
    hp = Any[nothing for _ in Z]
    accs = Dict(s => Acc(c, Z, reader; T = acc) for s in (1, -1))
    for s in (1, -1); accs[s].hp = hp; end                          # one AC-coupled line
    half = P ÷ 2; tr = round(Int, discard * half)
    for _ in 1:p.n_cyc, s in (1, -1)
        for t in 1:half
            Z = step(c, Z, Float32(s) * p.ε, rng)
            (t - 1) % p.se == 0 || continue
            if t > tr
                accumulate!(accs[s], c, input_obs(c, rng), observe(c, Z, rng), 1f0, reader, rng; dts = p.se * p.dt)
            elseif reader.hp_τ > 0f0
                hp_only!(accs[s], c, input_obs(c, rng), observe(c, Z, rng), reader; dts = p.se * p.dt)
            end
        end
    end
    o0 = input_obs(c, rng)
    (HWp, Hbp), (HWm, Hbm) = close_acc(accs[1], c, o0), close_acc(accs[-1], c, o0)
    kp, km = accs[1].k, accs[-1].k
    return assemble(ps, [HWp[l] ./ kp .- HWm[l] ./ km for l in eachindex(HWp)],
                    [Hbp[l] ./ kp .- Hbm[l] ./ km for l in eachindex(Hbp)], -1f0 / (2f0 * p.ε))
end

"""Inference on the (impaired) device: settle T steps with `dyn` (noise included) and return the codebook logits
Re⟨code_c, e⟩/d of the emitted output e (classes × batch, on the CPU)."""
function predict(ps, x, codes, dyn::Dyn = IDEAL; T = 200, dt = 0.5f0, to = identity, rng = Xoshiro(0))
    cost = CodebookCost(to === identity ? codes : to(codes), ones(Int, size(x, 2)))     # labels unused at β = 0
    c = Ctx(ps, x, cost, dyn, dt; to = to)
    Z = settle(c, 0f0, T, rng); e = emit(c, Z, length(Z))
    C = to === identity ? codes : to(codes)
    return Array(real.(adjoint(C) * e)) ./ size(e, 1)
end

const ESTIMATORS = Dict(:lockin => est_lockin, :centred => est_centred, :classic => est_classic,
                        :static_sq => est_static_sq)

"Steps per gradient for each estimator (matched time = equal totals)."
steps_per_grad(e::Symbol, p::Probe) =
    e === :lockin || e === :static_sq ? lockin_steps(p) :
    e === :centred ? p.T_free + 2p.T_nudge : p.T_free + p.T_nudge

"A Probe for centred/classic EP with the same total steps as a lock-in probe (β given; nudges share the rest)."
function matched_static(p::Probe, β::Real, e::Symbol)
    rest = lockin_steps(p) - p.T_free
    Probe(p; ε = Float32(β), T_nudge = e === :centred ? rest ÷ 2 : rest)
end
Probe(p::Probe; kw...) = Probe(; (k => get(kw, k, getfield(p, k)) for k in fieldnames(Probe))...)

# ------------------------------------------------------------------ trims and coloured noise
"""Emission trim for static detuning (T2c, derived): with z ← R[(1 − dt)z + dt·u], R = cis(δ·dt), the fixed point is
z = R·dt·u/(1 − (1 − dt)R), a lag φ(δ) and a magnitude m(δ) that depend on δ alone (not on the input). `:phase`
returns cis(−φ) (what a phase-comparison loop measures, T2c); `:full` returns (1 − (1 − dt)R)/(R·dt), which makes the
emitted fixed point the undetuned one exactly."""
function trim_factors(δω::Vector{Vector{Float32}}, dt::Real; kind::Symbol = :phase)
    map(δω) do d
        R = cis.(Float64.(d) .* dt)
        f = (1 .- (1 - dt) .* R) ./ (R .* dt)
        ComplexF32.(kind === :full ? f : f ./ abs.(f))
    end
end

"""1/f noise on the Hebbian samples: per element a sum of `n` Ornstein–Uhlenbeck processes with time constants
τ_min·4^j, each of unit variance, scaled to total RMS σ (complex: σ per real and imaginary part, as the white σh).
Exact OU updates over the elapsed time, so the process runs between samples. Call as reader.noise(rng, size, t)."""
mutable struct PinkNoise
    σ::Float32
    τs::Vector{Float64}
    state::Dict{Tuple, Vector{Any}}                  # per array size: n complex OU states (on H's device)
    buf::Dict{Tuple, Tuple{Any, Any}}                # real innovation buffers (in place: no allocation per sample)
    out::Dict{Tuple, Any}
    tlast::Dict{Tuple, Float64}
    rng::Xoshiro                                     # CPU arrays; device arrays use the device's default RNG
end
PinkNoise(σ; τ_min = 0.5, n = 7, seed = 0) =
    PinkNoise(Float32(σ), [τ_min * 4.0^j for j in 0:n - 1], Dict(), Dict(), Dict(), Dict(), Xoshiro(70_000 + seed))
_fillrandn!(r, A) = A isa Array ? randn!(r, A) : randn!(A)
function (p::PinkNoise)(rng, H, t)
    sz = size(H); n = length(p.τs); r = p.rng
    if !haskey(p.state, sz)
        rr, ri = similar(real.(H)), similar(real.(H))
        p.buf[sz] = (rr, ri)
        p.state[sz] = Any[(_fillrandn!(r, rr); _fillrandn!(r, ri); complex.(rr, ri)) for _ in 1:n]
        p.out[sz] = similar(H, ComplexF32); p.tlast[sz] = t
    else
        Δ = t - p.tlast[sz]; p.tlast[sz] = t; rr, ri = p.buf[sz]
        for (j, τ) in enumerate(p.τs)
            a = Float32(exp(-Δ / τ)); q = sqrt(1f0 - a^2)
            _fillrandn!(r, rr); _fillrandn!(r, ri); x = p.state[sz][j]
            x .= a .* x .+ q .* complex.(rr, ri)
        end
    end
    o = p.out[sz]; fill!(o, 0); for x in p.state[sz]; o .+= x; end
    o .*= p.σ / sqrt(Float32(n))
    return o
end

# ------------------------------------------------------------------ exact gradient (implicit differentiation)
# The free equilibrium s* solves s = Φ(s; θ), Φ the damped projected step (any dt gives the same fixed points; dt is
# the settle's). For L(s*) the gradient is g = (∂Φ/∂θ)ᵀλ with λ = ∂L/∂s + (∂Φ/∂s)ᵀλ, solved by fixed-point
# iteration (it converges at the settle's own rate). Real arithmetic in Float64 (re/im split) so Zygote is plain.
# Static impairments (δω, trim, gain, iso) are part of Φ: then g is the gradient of what the device computes.
using Zygote

_Φparts(ps) = [(W = Float64.(ps[k].weight), br = Float64.(ps[k].bias_real), bi = Float64.(ps[k].bias_imag))
               for k in keys(ps)]

"Emitted state of layer l in re/im parts: gain, then the complex emission factor (as `emit`, without drift)."
function _em64(r, i, l, dyn::Dyn)
    dyn.gain === nothing || (a = Float64.(dyn.gain[l]); r = a .* r; i = a .* i)
    dyn.emit_c === nothing && return r, i
    cr = Float64.(real.(dyn.emit_c[l])); ci = Float64.(imag.(dyn.emit_c[l]))
    return cr .* r .- ci .* i, cr .* i .+ ci .* r
end

function _Φ(zr, zi, θ, x0r, x0i, dyn::Dyn, dt)
    n = length(θ)
    em(l) = _em64(zr[l], zi[l], l, dyn)
    map(1:n) do l
        inr, ini = l == 1 ? (x0r, x0i) : em(l - 1)
        gr = θ[l].W * inr .+ θ[l].br; gi = θ[l].W * ini .+ θ[l].bi
        if l < n
            er, ei = em(l + 1)
            gr = gr .+ transpose(θ[l + 1].W) * er
            gi = gi .+ transpose(θ[l + 1].W) * ei
        end
        r = sqrt.(gr .^ 2 .+ gi .^ 2)
        ur = gr ./ max.(r, 1e-10); ui = gi ./ max.(r, 1e-10)
        nr = (1 - dt) .* zr[l] .+ dt .* ur; ni = (1 - dt) .* zi[l] .+ dt .* ui
        ang = zero(nr)
        if dyn.δω !== nothing || dyn.trim !== nothing
            ang = ang .+ ((dyn.δω === nothing ? 0.0 : Float64.(dyn.δω[l])) .+
                          (dyn.trim === nothing ? 0.0 : Float64.(dyn.trim[l]))) .* dt
        end
        _has_iso(dyn) && (ang = ang .+ Float64(_lv(dyn.iso, l)) .* (r .- Float64(_lv(dyn.r_ref, l))) .* dt)
        (nr .* cos.(ang) .- ni .* sin.(ang), nr .* sin.(ang) .+ ni .* cos.(ang))
    end
end

function _loss64(cost::CodebookCost, zr, zi)
    C = cost.codes; d = size(zr, 1)
    s = (transpose(real.(C)) * zr .+ transpose(imag.(C)) * zi) ./ d        # Re(Cᴴ z)/d
    s = s .- maximum(s; dims = 1)
    lp = s .- log.(sum(exp.(s); dims = 1))
    return -sum(Float64.(cost.y_onehot) .* lp) / size(zr, 2)
end

"Similarity cost (library ep_loss): 1 − mean_b Re⟨y, z_b⟩/d."
function _loss64(cost::SimilarityCost, zr, zi)
    y = cost.y; d = size(zr, 1)
    yr = Float64.(real.(y)); yi = Float64.(imag.(y))
    s = sum(yr .* zr .+ yi .* zi; dims = 1) ./ d
    return 1 - sum(s) / size(zr, 2)
end

"""Exact gradient of the free-equilibrium loss (mean over the batch) w.r.t. weight, bias_real, bias_imag, shaped like
ps. `dyn` may carry static impairments (δω, trim, gain, emit_c, iso); temporal noise and drift are ignored. `init`: raw
starting states per layer (default zero); info.emitted is the emitted fixed point. Returns (grad, info)."""
function exact_gradient(ps, x, cost, dyn::Dyn = IDEAL; dt = 0.5, T = 4000, tol = 1e-12, Tadj = 8000, init = nothing)
    θ = _Φparts(ps)
    z0 = ComplexF64.(PN._phase_input_to_complex(x isa AbstractVector ? reshape(x, :, 1) : x))
    x0r, x0i = real.(z0), imag.(z0)
    B = size(z0, 2)
    zr = init === nothing ? [zeros(size(t.W, 1), B) for t in θ] : [Float64.(real.(z)) for z in init]
    zi = init === nothing ? [zeros(size(t.W, 1), B) for t in θ] : [Float64.(imag.(z)) for z in init]
    res = Inf; it = 0
    for k in 1:T
        out = _Φ(zr, zi, θ, x0r, x0i, dyn, dt)
        res = maximum(maximum(abs.(out[l][1] .- zr[l])) + maximum(abs.(out[l][2] .- zi[l])) for l in eachindex(out))
        zr = [o[1] for o in out]; zi = [o[2] for o in out]; it = k
        res < tol && break
    end
    n = length(θ)
    Lr, Li = Zygote.gradient((a, b) -> _loss64(cost, _em64(a, b, n, dyn)...), zr[n], zi[n])   # cost reads the emission
    _, back = Zygote.pullback((a, b, t) -> _Φ(a, b, t, x0r, x0i, dyn, dt), zr, zi, θ)
    λr = [zeros(size(z)) for z in zr]; λi = [zeros(size(z)) for z in zi]
    gθ = nothing; ares = Inf; ait = 0
    for k in 1:Tadj
        ct = [(λr[l], λi[l]) for l in 1:n]
        dzr, dzi, dθ = back(ct)
        nr = [dzr[l] .+ (l == n ? Lr : 0.0) for l in 1:n]; ni = [dzi[l] .+ (l == n ? Li : 0.0) for l in 1:n]
        ares = maximum(maximum(abs.(nr[l] .- λr[l])) + maximum(abs.(ni[l] .- λi[l])) for l in 1:n)
        λr, λi = nr, ni; gθ = dθ; ait = k
        ares < tol && break
    end
    _, _, gθ = back([(λr[l], λi[l]) for l in 1:n])
    ks = collect(keys(ps))
    g = NamedTuple(k => _pad(ps[k], (weight = Float32.(gθ[l].W), bias_real = Float32.(gθ[l].br),
                                     bias_imag = Float32.(gθ[l].bi))) for (l, k) in enumerate(ks))
    emitted = [complex.(_em64(zr[l], zi[l], l, dyn)...) for l in 1:n]          # the fixed point as other units see it
    return g, (settle_res = res, settle_steps = it, adjoint_res = ares, adjoint_steps = ait, emitted = emitted)
end
