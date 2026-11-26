# Prior and posterior sampling on synthetic designs: exact kernel packets in 1-D and sparse-grid inducing points in d
# dimensions (KP), against the exact GP by Cholesky; random inputs, SEEDS replicates (mean and sd).
#   scripts/run_sampling.sh julia synthetic [parts] [rho=.. sigma2=.. tau2=..]
#   scripts/run_sampling.sh julia synthetic replot [parts]
using KernelPackets, KernelPackets.Benchmarks, KernelPackets.Evaluation, LinearAlgebra, Plots, Printf, Random,
      Serialization, Statistics

include(joinpath(@__DIR__, "data.jl"))

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

function part_prior_draws()
    X = nodes(500, MersenneTwister(0))
    curves = Dict{String,Any}("X" => X)
    for k in KS
        curves["KP_$k"] = prior_sample(KernelPacket(X, k, PRIOR_RHO; method=METHOD), 3; rng=MersenneTwister(0))
        curves["Cholesky_$k"] = prior_sample(DenseGP, X, k, PRIOR_RHO, 3; rng=MersenneTwister(1))
    end
    serialize(joinpath(OUT, "prior_draws_1d.jls"), curves)
    plot_prior_draws()
end

function plot_prior_draws()
    c = deserialize(joinpath(OUT, "prior_draws_1d.jls"))
    X = c["X"]
    panels = [draws_plot!(plot(), X, ["KP" => c["KP_$k"], "Cholesky" => c["Cholesky_$k"]],
                          "Matérn-$(k - 2)/2, ρ = $PRIOR_RHO, n = $(length(X))";
                          band=(zeros(length(X)), ones(length(X)), "prior mean and 95% interval"))
              for k in KS]
    save_figure(plot(panels...; layout=(1, length(KS)), size=(500 * length(KS), 380), link=:y),
                joinpath(OUT, "prior_draws_1d.png"))
end

function part_prior_accuracy()
    rows = NamedTuple[]
    for k in SENS_KS, n in PRIOR_ACC_N, seed in 0:(SEEDS - 1)
        X = scaling_design(n, MersenneTwister(seed))
        Kd = matern(X, X, k, SCALING_RHO)
        c = cond(Kd)
        for (method, L) in (("KP", sample_(KernelPacket(X, k, SCALING_RHO; method=METHOD), Matrix(1.0I, n, n))),
                            ("Cholesky", Matrix(cholesky(Symmetric(Kd)).L)))
            push!(rows, merge((seed=seed, k=k, n=n, rho=SCALING_RHO, cond=c, method=method), sampler(L, Kd)))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "prior_accuracy")
    plot_prior_accuracy()
end

function plot_prior_accuracy()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "prior_accuracy")
    ns = unique(r["n"] for r in rows)
    panels = [begin
                  series = Dict(m => begin
                                    _, (mu, sd) = mean_sd(rows, "n", "wasserstein2"; method=m, k=Float64(k))
                                    (max.(mu, 1e-17), sd)
                                end for m in ("KP", "Cholesky"))
                  metric_plot!(plot(), ns, series, "number of points n", "W_2 to the exact prior, Matérn-$(k - 2)/2")
              end for k in SENS_KS]
    save_figure(plot(panels...; layout=(1, length(SENS_KS)), size=(1500, 380), link=:y),
                joinpath(OUT, "prior_accuracy_1d.png"))
end

function part_prior_sensitivity()
    rows = NamedTuple[]
    for k in SENS_KS, r in SENS_RHOS, seed in 0:(SEEDS - 1)
        X = nodes(SENS_N, MersenneTwister(seed))
        Kd = matern(X, X, k, r)
        c = cond(Kd)
        for (method, L) in (("KP", sample_(KernelPacket(X, k, r; method=METHOD), Matrix(1.0I, SENS_N, SENS_N))),
                            ("Cholesky", Matrix(cholesky(Symmetric(Kd)).L)))
            push!(rows, merge((seed=seed, k=k, rho=r, n=SENS_N, cond=c, method=method), sampler(L, Kd)))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "prior_sensitivity")
    plot_prior_sensitivity()
end

function plot_prior_sensitivity()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "prior_sensitivity")
    rhos = unique(r["rho"] for r in rows)
    panels = [begin
                  series = Dict(m => begin
                                    _, (mu, sd) = mean_sd(rows, "rho", "wasserstein2"; method=m, k=Float64(k))
                                    (max.(mu, 1e-17), sd)
                                end for m in ("KP", "Cholesky"))
                  metric_plot!(plot(), rhos, series, "lengthscale ρ",
                               "W_2 to the exact prior, Matérn-$(k - 2)/2, n = $SENS_N")
              end for k in SENS_KS]
    save_figure(plot(panels...; layout=(1, length(SENS_KS)), size=(1500, 380), link=:y),
                joinpath(OUT, "prior_sensitivity_1d.png"))
end

function part_prior_scaling()
    rows = NamedTuple[]
    for n in SIZES, seed in 0:((n <= BIG ? min(SEEDS, 3) : 1) - 1)
        X = scaling_design(n, MersenneTwister(seed))
        m, t = timed(() -> begin
            kp = KernelPacket(X, K, SCALING_RHO; method=METHOD)
            prior_sample(kp, P_TIME; rng=MersenneTwister(seed))
            kp
        end; repeat=0)
        push!(rows, (seed=seed, method="KP", n=n, draws=P_TIME, time=t, storage_mb=nbytes(m) / 2^20))
        if n <= N_DENSE
            _, t = timed(() -> prior_sample(DenseGP, X, K, SCALING_RHO, P_TIME; rng=MersenneTwister(seed));
                         repeat=n >= 5000 ? 0 : 1)
            push!(rows, (seed=seed, method="Cholesky", n=n, draws=P_TIME, time=t, storage_mb=8 * n * n / 2^20))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "prior_scaling")
    plot_prior_scaling()
end

function plot_prior_scaling()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "prior_scaling")
    sizes = unique(r["n"] for r in rows)
    panels = [begin
                  series = Dict{String,Any}()
                  for method in ("KP", "Cholesky")
                      ns, (mu, sd) = mean_sd(rows, "n", col; method=method)
                      pad = length(sizes) - length(ns)
                      series[method] = ([mu; fill(NaN, pad)], [sd; fill(NaN, pad)])
                  end
                  metric_plot!(plot(), sizes, series, "number of points n", title; slopes=slopes)
              end for (col, title, slopes) in (("time", "time of setup and $P_TIME prior draws (s)", (1, 3)),
                                               ("storage_mb", "storage of the sampler (MB)", (1, 2)))]
    save_figure(plot(panels...; layout=(1, 2), size=(1100, 380)), joinpath(OUT, "prior_scaling_1d.png"))
end

function part_posterior_draws()
    rows, curves = NamedTuple[], Dict{String,Any}()
    for seed in 0:(SEEDS - 1)
        rng = MersenneTwister(seed)
        X = sort(train_inputs(N_POST, 1, rng)[:, 1])
        xs, y = sort(test_inputs(M_POST, 1, rng)[:, 1]), f(X)
        for k in KS
            m = KernelPacket(X, k, get(FIXED, "rho", 0.1); method=METHOD)
            t_train = haskey(FIXED, "rho") ? 0.0 : timed(() -> mle(m, y; bounds=(1e-2, 1.0)); repeat=0)[2]
            _, beta, s2 = loglike(m, y)
            b, s2 = beta[1], get(FIXED, "sigma2", s2)
            exact = DenseGP(X, y .- b, k, m.rho; sigma2=s2)
            mu, C = predict(exact, xs; cov=true)
            draws = Dict{String,Matrix{Float64}}()
            for (method, fn) in (("KP", () -> b .+ sqrt(s2) .* posterior_sample(m, xs, (y .- b) ./ sqrt(s2), P_POST; rng=MersenneTwister(seed))),
                                 ("Cholesky", () -> b .+ sample(exact, xs, P_POST; rng=MersenneTwister(seed))))
                F, t = timed(fn)
                draws[method] = F
                push!(rows, (seed=seed, k=k, method=method, n=N_POST, m=M_POST, draws=P_POST, rho=m.rho, sigma2=s2,
                             train_time=t_train, test_time=t, w2=draws_wasserstein2(F, mu .+ b, C),
                             mean_error=max_error(vec(mean(F; dims=2)), mu .+ b),
                             var_error=max_error(vec(var(F; dims=2, corrected=false)), diag(C)),
                             rmse=rmse(vec(mean(F; dims=2)), f(xs))))
            end
            if seed == 0
                curves["rho_$k"] = m.rho
                curves["mean_$k"] = mu .+ b
                curves["var_$k"] = diag(C)
                curves["KP_$k"] = draws["KP"][:, 1:3]
                curves["Cholesky_$k"] = draws["Cholesky"][:, 1:3]
                curves["X"] = X; curves["y"] = y; curves["xs"] = xs; curves["truth"] = f(xs)
            end
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "posterior_draws")
    serialize(joinpath(OUT, "posterior_draws_1d.jls"), curves)
    plot_posterior_draws()
end

function plot_posterior_draws()
    c = deserialize(joinpath(OUT, "posterior_draws_1d.jls"))
    panels = [draws_plot!(plot(), c["xs"], ["KP" => c["KP_$k"], "Cholesky" => c["Cholesky_$k"]],
                          @sprintf("Matérn-%d/2, n = %d, ρ = %.3f", k - 2, length(c["X"]), c["rho_$k"]);
                          X=c["X"], y=c["y"], truth=c["truth"],
                          band=(c["mean_$k"], c["var_$k"], "exact posterior mean and 95% interval"))
              for k in KS]
    save_figure(plot(panels...; layout=(1, length(KS)), size=(500 * length(KS), 380), link=:y),
                joinpath(OUT, "posterior_draws_1d.png"))
end

function part_posterior_accuracy()
    rows = NamedTuple[]
    for n in POST_ACC_N, seed in 0:((n <= 10^5 ? min(SEEDS, 3) : 1) - 1)
        rng = MersenneTwister(seed)
        X = sort(train_inputs(n, 1, rng)[:, 1])
        xs = sort(test_inputs(M_POST, 1, rng)[:, 1])
        y, truth = f(X) .+ NOISE .* randn(rng, n), f(xs)
        m = KernelPacket(X, K, PRIOR_RHO; tau2=NOISE^2, method=METHOD)
        Fk = posterior_sample(m, xs, y, P_ACC; rng=MersenneTwister(seed))
        w2_kp = predictive_w2(truth, vec(mean(Fk; dims=2)), vec(var(Fk; dims=2, corrected=false)))
        if n > N_DENSE
            push!(rows, (seed=seed, method="KP", n=n, m=M_POST, draws=P_ACC, rho=PRIOR_RHO, tau2=NOISE^2,
                         w2=w2_kp, rmse=rmse(vec(mean(Fk; dims=2)), truth)))
            continue
        end
        exact = DenseGP(X, y, K, PRIOR_RHO; sigma2=1.0, tau2=NOISE^2)
        mu, C = predict(exact, xs; cov=true)
        Fc = sample(exact, xs, P_ACC; rng=MersenneTwister(seed))
        for (method, F, w2_t) in (("KP", Fk, w2_kp),
                                  ("Cholesky", Fc, predictive_w2(truth, vec(mean(Fc; dims=2)), vec(var(Fc; dims=2, corrected=false)))))
            push!(rows, (seed=seed, method=method, n=n, m=M_POST, draws=P_ACC, rho=PRIOR_RHO, tau2=NOISE^2,
                         w2=w2_t, w2_exact=draws_wasserstein2(F, mu, C), rmse=rmse(vec(mean(F; dims=2)), truth)))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "posterior_accuracy")
    plot_posterior_accuracy()
end

function plot_posterior_accuracy()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "posterior_accuracy")
    panels = [begin
                  ns, series = padded(rows, "n", col)
                  metric_plot!(plot(), ns, series, "training size n", title)
              end for (col, title) in (("w2", "W_2 to the ground truth"), ("w2_exact", "W_2 to the exact posterior"))]
    save_figure(plot(panels...; layout=(1, 2), size=(1100, 380)), joinpath(OUT, "posterior_accuracy_1d.png"))
end

function part_posterior_sensitivity()
    rows = NamedTuple[]
    for (sweep, vals) in (("rho", POST_SENS_RHOS), ("tau2", POST_SENS_TAU2S))
        for val in vals
            r, t2 = sweep == "rho" ? (val, NOISE^2) : (PRIOR_RHO, val)
            for seed in 0:(SEEDS - 1)
                rng = MersenneTwister(seed)
                X = sort(train_inputs(SENS_N, 1, rng)[:, 1])
                xs = sort(test_inputs(M_POST, 1, rng)[:, 1])
                y = f(X) .+ sqrt(t2) .* randn(rng, SENS_N)
                m = KernelPacket(X, K, r; tau2=t2, method=METHOD)
                exact = DenseGP(X, y, K, r; sigma2=1.0, tau2=t2)
                mu, C = predict(exact, xs; cov=true)
                for (method, fn) in (("KP", () -> posterior_sample(m, xs, y, P_POST; rng=MersenneTwister(seed))),
                                     ("Cholesky", () -> sample(exact, xs, P_POST; rng=MersenneTwister(seed))))
                    F = fn()
                    push!(rows, (seed=seed, sweep=sweep, method=method, n=SENS_N, m=M_POST, draws=P_POST,
                                 rho=r, tau2=t2, w2=draws_wasserstein2(F, mu, C),
                                 rmse=rmse(vec(mean(F; dims=2)), f(xs))))
                end
            end
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "posterior_sensitivity")
    plot_posterior_sensitivity()
end

function plot_posterior_sensitivity()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "posterior_sensitivity")
    panels = [begin
                  ks, series = padded(rows, key, "w2"; sweep=sweep)
                  metric_plot!(plot(), ks, series, xlabel, title)
              end for (sweep, key, xlabel, title) in (
                  ("rho", "rho", "lengthscale ρ", "W_2 to the exact posterior (τ² = $(NOISE^2))"),
                  ("tau2", "tau2", "noise ratio τ²", "W_2 to the exact posterior (ρ = $PRIOR_RHO)"))]
    save_figure(plot(panels...; layout=(1, 2), size=(1100, 380)), joinpath(OUT, "posterior_sensitivity_1d.png"))
end

function part_scaling_1d()
    rows = NamedTuple[]
    for n in SIZES, seed in 0:((n <= BIG ? min(SEEDS, 3) : 1) - 1)
        X, y, _ = scaling_data(n, MersenneTwister(seed))
        m, t = timed(() -> begin
            kp = KernelPacket(X, K, get(FIXED, "rho", SCALING_RHO); method=METHOD)
            haskey(FIXED, "rho") || mle(kp, y; bounds=(1.0, 1e3))
            kp
        end; repeat=0)
        push!(rows, (seed=seed, method="KP", n=n, rho=m.rho, train_time=t, storage_mb=nbytes(m) / 2^20))
        if n <= N_DENSE
            exact, t = timed(() -> haskey(FIXED, "rho") ? DenseGP(X, y, K, FIXED["rho"]) :
                               mle(DenseGP, X, y, K; bounds=(1.0, 1e3))[1]; repeat=n >= 5000 ? 0 : 1)
            push!(rows, (seed=seed, method="Cholesky", n=n, rho=exact.rho[1], train_time=t,
                         storage_mb=nbytes(exact) / 2^20))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "scaling_1d")
    plot_scaling_1d()
end

function plot_scaling_1d()
    rows = read_csv(joinpath(OUT, "metrics.csv"), "scaling_1d")
    sizes = unique(r["n"] for r in rows)
    panels = [begin
                  series = Dict{String,Any}()
                  for method in ("KP", "Cholesky")
                      ns, (mu, sd) = mean_sd(rows, "n", col; method=method)
                      pad = length(sizes) - length(ns)
                      series[method] = ([mu; fill(NaN, pad)], [sd; fill(NaN, pad)])
                  end
                  metric_plot!(plot(), sizes, series, "training size n", title; slopes=slopes)
              end for (col, title, slopes) in (("train_time", "training time: maximum likelihood of ρ (s)", (1, 3)),
                                               ("storage_mb", "storage of the fitted model (MB)", (1, 2)))]
    save_figure(plot(panels...; layout=(1, 2), size=(1100, 380)), joinpath(OUT, "posterior_scaling_1d.png"))
end

function part_posterior_sparse_grid()
    rows = NamedTuple[]
    for d in SG_DIMS, seed in 0:(min(SEEDS, 3) - 1)
        rng = MersenneTwister(seed)
        X, xs = train_inputs(SG_N, d, rng), test_inputs(SG_M, d, rng)
        y = griewank(X) .+ NOISE .* randn(rng, SG_N)
        p = merge(Dict("rho" => rho(d), "sigma2" => 1.0, "tau2" => NOISE^2), FIXED)
        g = for_data(SparseGrid, X; k=K, rho=p["rho"], tau2=p["tau2"])
        m = SparseGridRegression(X, y, g; sigma2=p["sigma2"], tau2=p["tau2"])
        _, t_train = isempty(FIXED) ? timed(() -> train!(m); repeat=0) : (nothing, 0.0)
        exact = DenseGP(X, y, K, g.rho[1]; sigma2=m.sigma2, tau2=m.tau2)
        mu, C = predict(exact, xs; cov=true)
        for (method, fn, nb) in (("KP", () -> posterior_sample(m, xs, P_POST; rng=MersenneTwister(seed)), nbytes(m)),
                                 ("Cholesky", () -> sample(exact, xs, P_POST; rng=MersenneTwister(seed)), nbytes(exact)))
            F, t = timed(fn)
            push!(rows, (seed=seed, d=d, method=method, level=g.eta, inducing=length(g), n=SG_N, m=SG_M, draws=P_POST,
                         rho=g.rho[1], sigma2=m.sigma2, tau2=m.tau2, train_time=t_train, test_time=t,
                         storage_mb=nb / 2^20, w2=draws_wasserstein2(F, mu, C),
                         rmse=rmse(vec(mean(F; dims=2)), griewank(xs))))
        end
    end
    write_csv(rows, joinpath(OUT, "metrics.csv"), "posterior_sparse_grid")
end

named = Dict(split(a, "="; limit=2)[1] => split(a, "="; limit=2)[2] for a in ARGS if occursin("=", a))
const METHOD = Symbol(pop!(named, "method", "stable"))
const FIXED = Dict(k => parse(Float64, v) for (k, v) in named)
const OUT = METHOD === :stable ? joinpath(@__DIR__, "results") : joinpath(@__DIR__, "results", string(METHOD))
mkpath(OUT)
replot = "replot" in ARGS
parts = [a for a in ARGS if !occursin("=", a) && a != "replot"]
for part in (isempty(parts) ? PARTS : parts)
    sym = Symbol(replot ? "plot_" : "part_", part)
    isdefined(Main, sym) || continue
    println("== $part")
    getfield(Main, sym)()
end
