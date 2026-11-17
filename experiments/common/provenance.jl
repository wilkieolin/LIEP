# Provenance for every run: library + experiment commits, Julia version, seeds, config, and a content-addressed run_id.
# Usage (from an experiment's run.jl):
#   include(joinpath(@__DIR__, "..", "..", "common", "provenance.jl"))
#   write_provenance(joinpath(@__DIR__, "results", "run_tag"); seeds=[1,2,3], config=cfg)

using TOML, Dates, SHA

const ARC_ROOT = normpath(joinpath(@__DIR__, ".."))

function _git(dir, args...)
    try
        return strip(read(pipeline(`git --no-optional-locks -C $dir $args`; stderr=devnull), String))
    catch
        return "unavailable"
    end
end

function git_state(dir)
    commit = _git(dir, "rev-parse", "HEAD")
    branch = _git(dir, "rev-parse", "--abbrev-ref", "HEAD")
    # code state only: result files written by (other) runs do not make a run irreproducible
    status = _git(dir, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)**/results/**")
    dirty = status != "" && status != "unavailable"
    d = Dict{String,Any}("path" => dir, "commit" => commit, "branch" => branch, "dirty" => dirty)
    if dirty
        diff = _git(dir, "diff", "HEAD")
        d["diff_sha256"] = bytes2hex(sha256(diff))
        d["dirty_files"] = split(status, '\n')
    end
    return d
end

# The library actually loaded (an env may point at a worktree branch); else the sibling checkout.
function phasor_networks_dir()
    if isdefined(Main, :PhasorNetworks) && Main.PhasorNetworks isa Module
        d = pkgdir(Main.PhasorNetworks)
        d === nothing || return normpath(d)
    end
    return normpath(joinpath(ARC_ROOT, "..", "PhasorNetworks.jl"))
end

"""
    provenance(; seeds, config) -> Dict

Everything needed to reproduce a run. `config` should be a Dict (or NamedTuple) of
primitive values; it is written verbatim.
"""
function provenance(; seeds=Int[], config=Dict{String,Any}())
    cfg = config isa NamedTuple ? Dict(string(k) => v for (k, v) in pairs(config)) : config
    d = Dict{String,Any}(
        "timestamp" => string(now()),
        "hostname" => gethostname(),
        "julia_version" => string(VERSION),
        "script" => abspath(PROGRAM_FILE),
        "args" => copy(ARGS),
        "seeds" => collect(seeds),
        "config" => Dict(string(k) => (v isa Symbol ? string(v) : v) for (k, v) in cfg),
        "env" => Dict(k => get(ENV, k, "") for k in
                      ("JULIA_CUDA_HARD_MEMORY_LIMIT", "JULIA_NUM_THREADS")),
        "phasor_networks" => git_state(phasor_networks_dir()),
        "arc_experiments" => git_state(ARC_ROOT),
    )
    d["run_id"] = run_id(d)
    return d
end

"""
    run_id(prov) -> String

Content address of a run (2026-10-26): sha256 of the canonical TOML of what determines the result — script path
relative to the tree, args (path-like tokens reduced to their basename so host paths do not matter), seeds, config,
and both repos' commits and dirty-diff hashes. Timestamp, hostname and env are excluded. Two hosts running the same
cell on the same code get the same id; a dirty tree changes it through the diff hash.
"""
function run_id(p::Dict)
    canon(x) = x isa Symbol ? string(x) : x
    args = [occursin('/', a) ? basename(a) : a for a in p["args"]]
    key = Dict{String,Any}(
        "script" => relpath(p["script"], ARC_ROOT),
        "args" => args,
        "seeds" => p["seeds"],
        "config" => Dict(k => canon(v) for (k, v) in p["config"]),
        "lib_commit" => p["phasor_networks"]["commit"],
        "lib_diff" => get(p["phasor_networks"], "diff_sha256", ""),
        "exp_commit" => p["arc_experiments"]["commit"],
        "exp_diff" => get(p["arc_experiments"], "diff_sha256", ""),
    )
    io = IOBuffer(); TOML.print(io, key; sorted=true)
    return bytes2hex(sha256(take!(io)))
end

"""
    write_provenance(dir; seeds, config) -> path

Writes `provenance.toml` into `dir` (created if needed) and warns on dirty trees.
"""
function write_provenance(dir; kw...)
    mkpath(dir)
    p = provenance(; kw...)
    for repo in ("phasor_networks", "arc_experiments")
        p[repo]["dirty"] && @warn "$repo tree is dirty; result is not exactly reproducible" p[repo]["commit"]
    end
    path = joinpath(dir, "provenance.toml")
    open(io -> TOML.print(io, p; sorted=true), path, "w")
    return path
end
