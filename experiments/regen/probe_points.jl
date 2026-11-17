# LIEP-REGEN stage 3d: linear probe on the hidden layer of the networks trained at each (ε, ω_p) point, epoch 8: does the
# hidden layer still learn there? Same probe as LIEP-BASE. Usage (repo root): julia --project=. -t 4 …/regen/probe_points.jl <arm> <point> <seed> [ckpt suffix, e.g. _dt0.1]
# Writes results/probe/<tag>.csv (long format) + provenance. Settings and predictions: README.md.
include(joinpath(@__DIR__, "..", "base", "base_lib.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "BASE_BLAS", "4")))
const ARM = ARGS[1]; const POINT = ARGS[2]; const SEED = parse(Int, ARGS[3]); const R = joinpath(@__DIR__, "results"); const SUF = length(ARGS) > 3 ? ARGS[4] : ""
Ptr, ytr = pixels(:train); Pte, yte = pixels(:test); Etr, _ = fmnist(:train); Ete, _ = fmnist(:test); codes = paper_codes(SEED)
function hidden(ps, E)
    F = Matrix{Float32}(undef, 512, size(E, 2))
    for i in 1:2000:size(E, 2)
        j = i:min(i + 1999, size(E, 2)); c = Ctx(ps, E[:, j], CodebookCost(codes, ones(Int, length(j))), IDEAL, 0.5f0); z = settle(c, 0f0, 200, Xoshiro(0))[1]
        F[:, j] = vcat(real.(z), imag.(z))
    end
    F
end
t0 = time(); rows = NamedTuple[]; tag = "$(ARM)_$(POINT)_s$(SEED)$(SUF)"
ps = deserialize(joinpath(R, "ckpt", "$(ARM)_$(POINT)_lr3e-3_s$(SEED)$(SUF)_e8.jls")); Ftr, Fte = standardise(hidden(ps, Etr), hidden(ps, Ete))
a, = softmax_regression(Ftr, ytr, Fte, yte; lr = 1f-3, wd = 0f0, epochs = 30, seed = SEED)
push!(rows, (item = "LIEP-REGEN", arm = ARM, model = POINT, seed = SEED, epoch = 8, metric = "probe_test_acc", value = Float64(a)))
mkpath(joinpath(R, "probe")); CSV.write(joinpath(R, "probe", "$tag.csv"), DataFrame(rows))
write_provenance(joinpath(R, "probe", tag); seeds = [SEED], config = Dict("arm" => ARM, "point" => POINT, "ckpt_suffix" => SUF))
@printf("DONE %s test %.4f (%.0f s)\n", tag, a, time() - t0)
