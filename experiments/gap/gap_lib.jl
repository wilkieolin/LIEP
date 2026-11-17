# LIEP-GAP harness: LIEP-READOUT's 784→256→d phasor network run through PhasorDense's complex dispatch (y = W·x + b),
# with the projection applied here so that a soft projection's amplitude reaches the next layer (the Phase dispatch
# drops it at complex_to_angle). With `proj = hard` this is the library's Phase path (diag_gap.jl checks it, D6).
const DEV = get(ENV, "GAP_DEV", "gpu")
DEV == "gpu" && (using CUDA, LuxCUDA)
include(joinpath(@__DIR__, "..", "base", "base_lib.jl"))
import PhasorNetworks: soft_normalize_to_unit_circle, hill_charge
dev(x) = DEV == "gpu" ? gpu_device()(x) : x
host(x) = DEV == "gpu" ? cpu_device()(x) : x
const READOUT = joinpath(@__DIR__, "..", "readout")

"Arm name → (readout kind, output width d, logit scale τ), LIEP-READOUT's naming (cb<d>_t<τ>, q10_t<τ>, learned64)."
function readout_kind(arm)
    if (r = match(r"^cb(\d+)_t([\d.]+)", arm)) !== nothing
        (:codebook, parse(Int, r[1]), parse(Float32, r[2]))
    elseif (r = match(r"^q10_t([\d.]+)", arm)) !== nothing
        (:quadrature, 10, parse(Float32, r[1]))
    elseif startswith(arm, "learned64")
        (:learned, 64, 1f0)
    else
        error("unknown readout in arm $arm")
    end
end

"The network as LIEP-READOUT builds it (same rng use, so the same init for a seed)."
function build_net(seed, dout)
    rng = Xoshiro(seed)
    m = Chain(PhasorDense(784 => 256, normalize_to_unit_circle, use_bias = true), PhasorDense(256 => dout, normalize_to_unit_circle, use_bias = true))
    ps0, st = Lux.setup(rng, m); ps0 = NamedTuple(k => merge(ps0[k], (weight = 0.4f0 .* ps0[k].weight,)) for k in keys(ps0))
    m, ps0, st
end

# projections: hard z/|z| (the library default, ε 1e-8), soft amplitude z/√(|z|² + s²), soft phase (magnitude 1,
# phase blended toward 0 for |z| below the band [lo, hi])
hard(y) = normalize_to_unit_circle(y)
soft_amp(s) = y -> y ./ sqrt.(abs2.(real.(y)) .+ abs2.(imag.(y)) .+ Float32(s)^2)
soft_phase(lo, hi) = y -> soft_normalize_to_unit_circle(y; r_lo = Float32(lo), r_hi = Float32(hi))
"Hill emission (library rule, EM1; 2-D static form here because the library's 2-D Phase path does not emit):
z = q · y/|y|, q = hill_charge(|y|, log θ, n) = rⁿ/(rⁿ + θⁿ). `lt` is log θ, learned by the caller (a parameter)."
hill(n, lt, y) = (r = sqrt.(abs2.(real.(y)) .+ abs2.(imag.(y)) .+ 1f-12); hill_charge(r, lt, n) .* (y ./ r))

"Complex forward: returns (pre-projection hidden y1, hidden z1, pre-projection output y2, output z2)."
function forward(m, pnet, st, xz; p1 = hard, p2 = hard)
    y1 = first(Lux.apply(m.layers[1], xz, pnet.layer_1, st.layer_1)); z1 = p1(y1)
    y2 = first(Lux.apply(m.layers[2], z1, pnet.layer_2, st.layer_2)); z2 = p2(y2)
    y1, z1, y2, z2
end

"Logits of the readout `kind` on output states z (d × B): codebook τ·Re⟨code_c, z⟩/d, quadrature τ·Re z_c, learned."
function readout_logits(kind, τ, codes, p, z)
    kind === :codebook   ? τ .* real.(adjoint(codes) * z) ./ Float32(size(z, 1)) :
    kind === :quadrature ? τ .* real.(z) :
                           p.Wr * real.(z) .+ p.Wi * imag.(z)
end

to_complex(E) = ComplexF32.(angle_to_complex(E))
"Accuracy of a column-wise argmax of host logits against labels."
argmax_acc(L, y) = mean([argmax(view(L, :, j)) for j in 1:size(L, 2)] .== y)
