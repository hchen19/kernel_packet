# Regression on toy data: exact kernel packets in 1-D, sparse-grid inducing points in d dimensions, against the exact
# GP by Cholesky; random training and test sets from different distributions, metrics on the test set over SEEDS
# replicates (mean and sd).
#   scripts/run_regression.sh julia toy [parts] [rho=.. sigma2=.. tau2=..]
#   scripts/run_regression.sh julia toy replot [parts]
using KernelPackets, KernelPackets.Benchmarks, KernelPackets.Evaluation, LinearAlgebra, Plots, Printf, Random,
      Serialization, Statistics

include(joinpath(@__DIR__, "data.jl"))

const ACCURACY_1D = [("rmse", "test RMSE", true), ("nlpd", "test NLPD", false), ("w2", "test W_2", true)]
const ACCURACY_SG = [("rmse", "test RMSE", true), ("nlpd", "test NLPD", false), ("w2", "W_2 to the exact posterior", true)]
const COST = [("time", "time of fit and prediction (s)"), ("storage_mb", "storage of the fitted model (MB)")]

function mean_sd(rows, key, value; match...)
    sel = [r for r in rows if all(r[string(a)] == b for (a, b) in match)]
    keys = unique(r[key] for r in sel)
    vals = [[r[value] for r in sel if r[key] == x] for x in keys]
    keys, ([mean(v) for v in vals], [std(v; corrected=false) for v in vals])
end

function padded(rows, key, value; methods=("KP", "Cholesky"), match...)
    keys = unique(r[key] for r in rows if all(r[string(a)] == b for (a, b) in match))
    out = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
    valid = [r for r in rows if !isnan(get(r, value, NaN))]
    for method in methods
        ks, (mu, sd) = mean_sd(valid, key, value; method=method, match...)
        at = Dict(k => i for (i, k) in enumerate(ks))
        out[method] = ([haskey(at, k) ? mu[at[k]] : NaN for k in keys],
                       [haskey(at, k) ? sd[at[k]] : NaN for k in keys])
    end
    keys, out
end

function fit_1d(X, y; k=K, noisy=false)
    noisy = noisy && METHOD === :stable
    m = KernelPacket(X, k, get(FIXED, "rho", 0.1); tau2=noisy ? get(FIXED, "tau2", 0.0) : 0.0, method=METHOD)
    train = noisy && !haskey(FIXED, "tau2")
    t_train = if !haskey(FIXED, "rho")
        _, t = timed(() -> mle(m, y; bounds=(1e-3, 1.0), tau2_bounds=train ? (1e-8, 10.0) : nothing); repeat=0)
        t
    else
        0.0
    end
    _, beta, s2 = loglike(m, y)
    m, beta[1], get(FIXED, "sigma2", s2), t_train
end

function part_posterior_1d()
    X, y, _, _ = data_1d(N_1D, MersenneTwister(0))
    grid = collect(range(0, 1, 1000))
    curves = Dict{String,Any}("X" => X, "y" => y, "grid" => grid, "truth" => f(grid))
    for k in KS
        m, b, s2, _ = fit_1d(X, y; k=k)
        mu, v = posterior(m, grid, y .- b; var=true)
        ref, ref_v = predict(DenseGP(X, y .- b, k, m.rho; sigma2=s2), grid)
        curves["rho_$k"] = m.rho
        curves["KP_mean_$k"] = mu .+ b
        curves["KP_var_$k"] = s2 .* v
        curves["Cholesky_mean_$k"] = ref .+ b
        curves["Cholesky_var_$k"] = ref_v
    end
    serialize(joinpath(OUT, "posterior_1d.jls"), curves)
    plot_posterior_1d()
end

function plot_posterior_1d()
    c = deserialize(joinpath(OUT, "posterior_1d.jls"))
    panels = [posterior_plot!(plot(), c["grid"], c["truth"], c["X"], c["y"],
                              ["KP" => (c["KP_mean_$k"], c["KP_var_$k"]),
                               "Cholesky" => (c["Cholesky_mean_$k"], c["Cholesky_var_$k"])],
                              @sprintf("Matérn-%d/2, n = %d, ρ = %.3f", k - 2, length(c["X"]), c["rho_$k"]))
              for k in KS]
    save_figure(plot(panels...; layout=(1, length(KS)), size=(500 * length(KS), 380), link=:y),
                joinpath(OUT, "posterior_1d.png"))
end

function part_accuracy_1d()
    rows = NamedTuple[]
    for n in N_ACC, seed in 0:((n <= BIG ? SEEDS : 1) - 1)
        X, y, xs, ys = data_1d(n, MersenneTwister(seed); noise=NOISE)
        m, b, s2, t_train = fit_1d(X, y; noisy=true)
        (mu, v), t = timed(() -> posterior(m, xs, y .- b; var=true))
        mu, v = mu .+ b, s2 .* v
        w2_kp = predictive_w2(ys, mu, v)
        base = (seed=seed, n=n, m=M, rho=m.rho, sigma2=s2, tau2=m.tau2, train_time=t_train)
        if n > N_DENSE
            push!(rows, merge(base, (method="KP", test_time=t, w2=w2_kp), predictive(ys, mu, v, y)))
            continue
        end
        exact, t_fit = timed(() -> DenseGP(X, y .- b, K, m.rho; sigma2=s2, tau2=s2 * m.tau2))
        (ref, ref_v), t_ref = timed(() -> predict(exact, xs))
        ref = ref .+ b
        xw = xs[1:M_W2]
        mu_w, cov_w = posterior(m, xw, y .- b; cov=true)
        ref_w, ref_cov = predict(exact, xw; cov=true)
        w2_exact = wasserstein2(s2 .* cov_w, ref_cov, mu_w .- ref_w)
        for (method, mm, vv, tm, w, we) in (("KP", mu, v, t, w2_kp, w2_exact),
                                            ("Cholesky", ref, ref_v, t_fit + t_ref, predictive_w2(ys, ref, ref_v), 0.0))
            push!(rows, merge(base, (method=method, test_time=tm, w2=w, w2_exact=we,
                                     mean_vs_cholesky=max_error(mm, ref), var_vs_cholesky=max_error(vv, ref_v)),
                              predictive(ys, mm, vv, y)))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "accuracy_1d")
    plot_accuracy_1d()
end

function plot_accuracy_1d()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "accuracy_1d")
    panels = [begin
                  ns, series = padded(rows, "n", col)
                  metric_plot!(plot(), ns, series, "training size n", title; logy=logy)
              end for (col, title, logy) in ACCURACY_1D]
    save_figure(plot(panels...; layout=(1, 3), size=(1500, 380)), joinpath(OUT, "accuracy_1d.png"))
end

function part_scaling_1d()
    rows = NamedTuple[]
    for n in SIZES, seed in 0:((n <= BIG ? min(SEEDS, 3) : 1) - 1)
        X, y, xs, ys = scaling_design(n, MersenneTwister(seed))
        m, t = timed(() -> begin
            kp = KernelPacket(X, K, get(FIXED, "rho", SCALING_RHO); method=METHOD)
            haskey(FIXED, "rho") || mle(kp, y; bounds=(1.0, 1e3))
            kp
        end; repeat=0)
        push!(rows, (seed=seed, method="KP", n=n, m=M, rho=m.rho, train_time=t,
                     storage_mb=nbytes(m) / 2^20, rmse=rmse(posterior(m, xs, y), ys)))
        if n <= N_DENSE
            (exact, beta), t = timed(() -> haskey(FIXED, "rho") ? (DenseGP(X, y, K, FIXED["rho"]), 0.0) :
                                     mle(DenseGP, X, y, K; bounds=(1.0, 1e3)); repeat=n >= 5000 ? 0 : 1)
            push!(rows, (seed=seed, method="Cholesky", n=n, m=M, rho=exact.rho[1], train_time=t,
                         storage_mb=nbytes(exact) / 2^20, rmse=rmse(predict(exact, xs)[1] .+ beta, ys)))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "scaling_1d")
    plot_scaling_1d()
end

function plot_scaling_1d()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "scaling_1d")
    ns, s_train = padded(rows, "n", "train_time")
    _, s_mem = padded(rows, "n", "storage_mb")
    p1 = metric_plot!(plot(), ns, s_train, "training size n", "training time: maximum likelihood of ρ and fit (s)"; slopes=(1, 3))
    p2 = metric_plot!(plot(), ns, s_mem, "training size n", "storage of the fitted model (MB)"; slopes=(1, 2))
    save_figure(plot(p1, p2; layout=(1, 2), size=(1100, 380)), joinpath(OUT, "scaling_1d.png"))
end

function part_sparse_grid()
    rows, p0 = NamedTuple[], merge(INIT, FIXED)
    for d in SG_DIMS, seed in 0:(SEEDS - 1)
        X, y, xs, ys = data_nd(d, SG_N, MersenneTwister(seed))
        ys = ys .+ NOISE .* randn(MersenneTwister(100 + seed), M)
        exact = DenseGP(X, y, K, p0["rho"]; sigma2=p0["sigma2"], tau2=p0["tau2"])
        _, t_train = isempty(FIXED) ? timed(() -> train!(exact); repeat=0) : (nothing, 0.0)
        _, t_fit = timed(() -> fit!(exact, exact.rho, exact.sigma2, exact.tau2))
        (ref, ref_v), t = timed(() -> predict(exact, xs))
        ref_w, ref_cov = predict(exact, xs[1:M_W2, :]; cov=true)
        push!(rows, merge((seed=seed, method="Cholesky", d=d, level="exact", inducing=NaN, n=SG_N, rho=exact.rho[1],
                           sigma2=exact.sigma2, tau2=exact.tau2, train_time=t_train, fit_time=t_fit, test_time=t,
                           time=t_fit + t, storage_mb=nbytes(exact) / 2^20, w2=0.0),
                          predictive(ys, ref, ref_v .+ exact.tau2, y)))
        auto = for_data(SparseGrid, X; k=K, rho=p0["rho"], tau2=p0["tau2"])
        for lev in Any[SG_LEVELS[d]...; "auto"]
            g = lev == "auto" ? auto : SparseGrid(d, d + lev; k=K, rho=p0["rho"])
            m = SparseGridRegression(X, y, g; sigma2=p0["sigma2"], tau2=p0["tau2"])
            _, t_train = isempty(FIXED) ? timed(() -> train!(m); repeat=0) : (nothing, 0.0)
            _, t_fit = timed(() -> (m.cache = nothing; fit!(m)))
            (mu, v), t = timed(() -> predict(m, xs; var=true))
            mu_w, cov_w = predict(m, xs[1:M_W2, :]; cov=true)
            push!(rows, merge((seed=seed, method="KP", d=d, level=(lev == "auto" ? "auto" : string(g.eta)),
                               inducing=Float64(length(g)), n=SG_N, rho=g.rho[1], sigma2=m.sigma2, tau2=m.tau2,
                               train_time=t_train, fit_time=t_fit, test_time=t, time=t_fit + t,
                               storage_mb=nbytes(m) / 2^20, w2=wasserstein2(cov_w, ref_cov, mu_w .- ref_w)),
                              predictive(ys, mu, v .+ m.tau2, y)))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "sparse_grid")
    plot_sparse_grid()
end

function plot_sparse_grid()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "sparse_grid")
    swept = [r for r in rows if string(r["level"]) ∉ ("auto", "exact")]
    for (name, panels) in (("accuracy", ACCURACY_SG), ("scaling_sparse_grid", [(c, t, true) for (c, t) in COST]))
        for d in SG_DIMS
            ps = [begin
                      nz, kps = mean_sd(swept, "inducing", col; method="KP", d=Float64(d))
                      series = Dict{String,Any}("KP" => kps)
                      if col != "w2"
                          _, (cm, cs) = mean_sd(rows, "d", col; method="Cholesky", d=Float64(d))
                          series["Cholesky"] = (fill(cm[1], length(nz)), fill(cs[1], length(nz)))
                      end
                      metric_plot!(plot(), nz, series, "inducing points n_Z (sparse-grid levels)",
                                   "$title, d = $d, n = $SG_N"; logy=logy)
                  end for (col, title, logy) in panels]
            save_figure(plot(ps...; layout=(1, length(panels)), size=(500 * length(panels), 380)),
                        joinpath(OUT, "$(name)_$(d)d.png"))
        end
    end
end

function part_scaling_nd()
    rows, p0 = NamedTuple[], merge(INIT, FIXED)
    for d in SG_DIMS, n in ND_SIZES, seed in 0:(min(SEEDS, 3) - 1)
        X, y, xs, ys = data_nd(d, n, MersenneTwister(seed))
        m, t = timed(() -> begin
            g = for_data(SparseGrid, X; k=K, rho=p0["rho"], tau2=p0["tau2"])
            SparseGridRegression(X, y, g; sigma2=p0["sigma2"], tau2=p0["tau2"])
        end; repeat=0)
        push!(rows, (seed=seed, method="KP", d=d, n=n, inducing=Float64(length(m.sg)), train_time=t,
                     storage_mb=nbytes(m) / 2^20, rmse=rmse(predict(m, xs), ys)))
        if n <= N_DENSE
            exact, t = timed(() -> DenseGP(X, y, K, p0["rho"]; sigma2=p0["sigma2"], tau2=p0["tau2"]);
                             repeat=n >= 5000 ? 0 : 1)
            push!(rows, (seed=seed, method="Cholesky", d=d, n=n, inducing=NaN, train_time=t,
                         storage_mb=nbytes(exact) / 2^20, rmse=rmse(predict(exact, xs)[1], ys)))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "scaling_nd")
    plot_scaling_nd()
end

function plot_scaling_nd()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "scaling_nd")
    for d in SG_DIMS
        ns = unique(r["n"] for r in rows if r["d"] == Float64(d))
        ps = [begin
                  series = Dict{String,Any}()
                  for method in ("KP", "Cholesky")
                      ks, (mu, sd) = mean_sd(rows, "n", col; method=method, d=Float64(d))
                      pad = length(ns) - length(ks)
                      series[method] = ([mu; fill(NaN, pad)], [sd; fill(NaN, pad)])
                  end
                  metric_plot!(plot(), ns, series, "training size n", "$title, d = $d"; slopes=slopes)
              end for (col, title, slopes) in (("train_time", "training time: fit at given hyperparameters (s)", (1, 3)),
                                               ("storage_mb", "storage of the fitted model (MB)", (1, 2)))]
        save_figure(plot(ps...; layout=(1, 2), size=(1100, 380)), joinpath(OUT, "scaling_$(d)d.png"))
    end
end

named = Dict(split(a, "="; limit=2)[1] => split(a, "="; limit=2)[2] for a in ARGS if occursin("=", a))
const METHOD = Symbol(pop!(named, "method", "stable"))
const FIXED = Dict(k => parse(Float64, v) for (k, v) in named)
const OUT = METHOD === :stable ? joinpath(@__DIR__, "results") : joinpath(@__DIR__, "results", string(METHOD))
mkpath(OUT)
replot = "replot" in ARGS
parts = [a for a in ARGS if !occursin("=", a) && a != "replot"]
for part in (isempty(parts) ? PARTS : parts)
    println("== $part")
    getfield(Main, Symbol(replot ? "plot_" : "part_", part))()
end
