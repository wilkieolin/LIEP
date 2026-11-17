# LIEP-CLOSE: the settled network's exact gradient on the GPU (Float32), by implicit differentiation as liep_lib's
# `exact_gradient` (CPU, Float64): settle s = Φ(s; θ) by fixed-point iteration from zero, then the adjoint
# λ = ∂L/∂s + (∂Φ/∂s)ᵀλ by fixed-point iteration, g = (∂Φ/∂θ)ᵀλ. Φ is the damped projected step with top-down
# feedback (layer l receives W_{l+1}ᵀ times layer l+1's state), IDEAL dynamics, codebook cost. "BP through the
# settled network": the gradient EP estimates.

function _Φg(zr, zi, θ, xr, xi, dt)
    n = length(θ)
    map(1:n) do l
        inr, ini = l == 1 ? (xr, xi) : (zr[l - 1], zi[l - 1])
        gr = θ[l].W * inr .+ θ[l].br; gi = θ[l].W * ini .+ θ[l].bi
        if l < n
            gr = gr .+ transpose(θ[l + 1].W) * zr[l + 1]; gi = gi .+ transpose(θ[l + 1].W) * zi[l + 1]
        end
        r = sqrt.(gr .^ 2 .+ gi .^ 2); ur = gr ./ max.(r, 1f-10); ui = gi ./ max.(r, 1f-10)
        ((1 - dt) .* zr[l] .+ dt .* ur, (1 - dt) .* zi[l] .+ dt .* ui)
    end
end
function _lossg(Cr, Ci, Y, zr, zi)
    s = (transpose(Cr) * zr .+ transpose(Ci) * zi) ./ size(zr, 1)
    s = s .- maximum(s; dims = 1)
    -sum(Y .* (s .- log.(sum(exp.(s); dims = 1)))) / size(zr, 2)
end

"""exact_gradient_gpu(ps, X, codes, Y; dt, T, Tadj, tol) → (grad shaped like ps's layers, info). ps on the GPU,
X complex input (d × B, CuArray), codes complex (d_out × 10, CuArray), Y one-hot (10 × B, CuArray)."""
function exact_gradient_gpu(ps, X, codes, Y; dt = 0.5f0, T = 600, Tadj = 600, tol = 1f-5)
    ks = collect(keys(ps))
    θ = [(W = ps[k].weight, br = ps[k].bias_real, bi = ps[k].bias_imag) for k in ks]
    xr, xi = real.(X), imag.(X); B = size(X, 2)
    zr = [CUDA.zeros(Float32, size(t.W, 1), B) for t in θ]; zi = [CUDA.zeros(Float32, size(t.W, 1), B) for t in θ]
    res = Inf32; it = 0
    for k in 1:T
        out = _Φg(zr, zi, θ, xr, xi, dt)
        if k % 20 == 0
            res = maximum(Float32(maximum(abs.(out[l][1] .- zr[l])) + maximum(abs.(out[l][2] .- zi[l]))) for l in eachindex(out))
        end
        zr = [o[1] for o in out]; zi = [o[2] for o in out]; it = k
        res < tol && break
    end
    n = length(θ); Cr, Ci = real.(codes), imag.(codes)
    Lr, Li = Zygote.gradient((a, b) -> _lossg(Cr, Ci, Y, a, b), zr[n], zi[n])
    _, back = Zygote.pullback((a, b, t) -> _Φg(a, b, t, xr, xi, dt), zr, zi, θ)
    λr = [zero(z) for z in zr]; λi = [zero(z) for z in zi]; ares = Inf32; ait = 0
    for k in 1:Tadj
        dzr, dzi, _ = back([(λr[l], λi[l]) for l in 1:n])
        nr = [(dzr[l] === nothing ? zero(λr[l]) : dzr[l]) .+ (l == n ? Lr : 0f0) for l in 1:n]
        ni = [(dzi[l] === nothing ? zero(λi[l]) : dzi[l]) .+ (l == n ? Li : 0f0) for l in 1:n]
        if k % 20 == 0
            ares = maximum(Float32(maximum(abs.(nr[l] .- λr[l])) + maximum(abs.(ni[l] .- λi[l]))) for l in 1:n)
        end
        λr, λi = nr, ni; ait = k
        ares < tol && break
    end
    _, _, gθ = back([(λr[l], λi[l]) for l in 1:n])
    g = NamedTuple(k => (weight = gθ[l].W, bias_real = gθ[l].br, bias_imag = gθ[l].bi) for (l, k) in enumerate(ks))
    g, (settle_res = res, settle_steps = it, adjoint_res = ares, adjoint_steps = ait)
end
