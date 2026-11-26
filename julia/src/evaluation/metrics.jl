# Metrics of the experiments: errors against the truth or a reference, predictive scores, sampler accuracy (W2),
# time, and the experiment's single CSV table (as metrics.py)

using LinearAlgebra, Random, Statistics

export Z95, rmse, max_error, nlpd, coverage, predictive, predictive_w2, covariance_error, sampler,
       factor_wasserstein2, wasserstein2, draws_wasserstein2, timed, write_csv, read_csv, rand_beta

const Z95 = 1.959963984540054
_erf(x::Real) = ccall(:erf, Float64, (Float64,), Float64(x))

rmse(a, b) = sqrt(mean(abs2, vec(a) .- vec(b)))
max_error(a, b) = maximum(abs, vec(a) .- vec(b))

"Mean negative log predictive density of y under N(mean, var)."
nlpd(y, mu, v) = mean(0.5 .* log.(2π .* v) .+ 0.5 .* (y .- mu) .^ 2 ./ v)

"Fraction of y inside mean +- z sd."
coverage(y, mu, v; z=Z95) = mean(abs.(y .- mu) .<= z .* sqrt.(max.(v, 0)))

"Every predictive metric of the test targets y under N(mean, var) (variances floored at 1e-12)."
function predictive(y, mu, v, y_train)
    yv, muv, vv = Float64.(y), Float64.(mu), max.(Float64.(v), 1e-12)
    sd = sqrt.(vv)
    r = yv .- muv
    z = r ./ sd
    crps = sd .* (z .* _erf.(z ./ sqrt(2)) .+ sqrt(2 / π) .* exp.(-z .^ 2 ./ 2) .- 1 / sqrt(π))
    lo, hi = muv .- Z95 .* sd, muv .+ Z95 .* sd
    score = (hi .- lo) .+ (2 / 0.05) .* ((lo .- yv) .* (yv .< lo) .+ (yv .- hi) .* (yv .> hi))
    mt, vt = mean(y_train), var(y_train; corrected=false)
    cov_v = coverage(yv, muv, vv)
    (rmse=rmse(muv, yv), mae=mean(abs, r), smse=mean(abs2, r) / var(yv; corrected=false),
     nlpd=nlpd(yv, muv, vv), msll=nlpd(yv, muv, vv) - nlpd(yv, fill(mt, length(yv)), fill(vt, length(yv))),
     crps=mean(crps), interval_score=mean(score), coverage=cov_v, coverage_error=abs(cov_v - 0.95),
     zscore_ms=mean(abs2, z))
end

"Per-point 2-Wasserstein distance W2(N(mean, diag(var)), delta_y) / sqrt(m)."
predictive_w2(y, mu, v) = sqrt(mean((Float64.(mu) .- Float64.(y)) .^ 2 .+ max.(Float64.(v), 0.0)))

"max |L L' - K| of a sampler's implied factor L (its draws from the unit vectors)."
covariance_error(L, K) = max_error(L * L', K)

"W2(N(0, L L'), N(0, K)) = ||K^{1/2} U - L V||_F where K^{1/2} L = U S V' (no trace cancellation)."
function factor_wasserstein2(L, K)
    S = psd_sqrt(K)
    F = svd(S * L)
    norm(S * F.U - L * F.V)
end

"cov_error, whitened_error, kl_gaussian and wasserstein2 of the implied factor L against the exact K."
function sampler(L, K)
    C = cholesky(Symmetric(K)).L
    R = C \ L
    lam = eigvals(Symmetric(R * R' - I))
    (cov_error=covariance_error(L, K), whitened_error=maximum(abs, lam),
     kl_gaussian=0.5 * sum(lam .- log1p.(max.(lam, -1 + 1e-300))), wasserstein2=factor_wasserstein2(L, K))
end

psd_sqrt(M) = (E = eigen(Symmetric(M)); E.vectors * Diagonal(sqrt.(max.(E.values, 0))) * E.vectors')

"""W2(N(m + dm, Kh), N(m, K)) = sqrt(|dm|^2 + tr K + tr Kh - 2 tr (K^{1/2} Kh K^{1/2})^{1/2}), the error measure of
samplers (Wilson et al. 2021)."""
function wasserstein2(Kh, K, dm=0.0)
    S = psd_sqrt(K)
    sqrt(max(sum(abs2, dm) + tr(K) + tr(Kh) - 2 * tr(psd_sqrt(S * Kh * S)), 0))
end

"W2 between the Gaussian of the draws F (m x p) (sample mean and covariance) and N(mean, cov)."
draws_wasserstein2(F, mu, C) = wasserstein2(cov(F; dims=2), C, vec(mean(F; dims=2)) .- mu)

"(result, seconds): the mean of `repeat` calls after a first one that compiles (repeat = 0: the first call)."
function timed(f; repeat=1)
    t = @elapsed out = f()
    repeat == 0 && return out, t
    t = @elapsed for _ in 1:repeat
        out = f()
    end
    out, t / repeat
end

field(v::AbstractFloat) = string(round(v; sigdigits=6))
field(v) = (s = string(v); occursin(',', s) ? "\"$s\"" : s)

function fields(line)
    out, cur, quoted = String[], IOBuffer(), false
    for c in line
        if c == '"'
            quoted = !quoted
        elseif c == ',' && !quoted
            push!(out, String(take!(cur)))
        else
            print(cur, c)
        end
    end
    push!(out, String(take!(cur)))
end

"""Rows (NamedTuples) of one experiment part into the experiment's single table at path: a `part` column, the union of
the columns; earlier rows of the same part are replaced. Printed too."""
function write_csv(rows, path, part)
    old = Dict{String,String}[]
    old_head = String[]
    if isfile(path)
        lines = readlines(path)
        if !isempty(lines)
            old_head = fields(lines[1])
            old = [Dict(zip(old_head, fields(l))) for l in lines[2:end]]
            filter!(r -> r["part"] != part, old)
        end
    end
    new = [merge(Dict("part" => part), Dict(string(k) => field(v) for (k, v) in pairs(r))) for r in rows]
    cols = unique(["part"; old_head; [string(k) for r in rows for k in keys(r)]])
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in [old; new]
            println(io, join((get(r, c, "") for c in cols), ","))
        end
    end
    for r in rows
        println(join(("$k=$(v isa AbstractFloat ? round(v; sigdigits=4) : v)" for (k, v) in pairs(r)), ", "))
    end
end

"The rows of one part of the experiment's table, numbers as Float64 (empty cells as NaN, other text kept)."
function read_csv(path, part)
    lines = readlines(path)
    isempty(lines) && return Dict{String,Any}[]
    head = fields(lines[1])
    out = Dict{String,Any}[]
    for l in lines[2:end]
        vals = fields(l)
        row = Dict{String,Any}()
        for (k, v) in zip(head, vals)
            if v == ""
                row[k] = NaN
            else
                parsed = tryparse(Float64, v)
                row[k] = parsed === nothing ? v : parsed
            end
        end
        get(row, "part", "") == part && push!(out, row)
    end
    out
end

"Gamma(alpha, 1) draw via Marsaglia-Tsang (supports alpha > 0)."
function _rand_gamma(rng, a::Real)
    if a < 1
        return _rand_gamma(rng, a + 1) * rand(rng)^(1 / a)
    end
    d = a - 1 / 3
    c = 1 / sqrt(9d)
    while true
        x = randn(rng)
        v = 1 + c * x
        v <= 0 && continue
        v = v^3
        u = rand(rng)
        if u < 1 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1 - v + log(v))
            return d * v
        end
    end
end

"Beta(a, b) draw(s) without external dependencies."
rand_beta(rng::AbstractRNG, a::Real, b::Real) = (x = _rand_gamma(rng, a); x / (x + _rand_gamma(rng, b)))
rand_beta(rng::AbstractRNG, a::Real, b::Real, dims...) = [rand_beta(rng, a, b) for _ in CartesianIndices(dims)]
rand_beta(rng::AbstractRNG, a::AbstractArray, b::AbstractArray) = [rand_beta(rng, ai, bi) for (ai, bi) in zip(a, b)]
