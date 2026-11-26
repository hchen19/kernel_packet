# Figures of the experiments: the quantity of each panel is its title, no y-axis label; every
# method has one color, marker and line in every figure, KP drawn thick and on top; posterior mean and 95% interval with
# observations and ground truth, draws, and metrics against a parameter (log axes, reference slopes) (as plotting.py)

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Plots

export STYLE, posterior_plot!, draws_plot!, metric_plot!, save_figure

const INK, INK2, GRID = colorant"#0b0b0b", colorant"#52514e", colorant"#e4e3df"
const STYLE = Dict("KP" => (color=colorant"#1c5cab", marker=:circle, ls=:solid, lw=2.0),
                   "Cholesky" => (color=colorant"#e07b00", marker=:rect, ls=:dash, lw=1.3))
const GUIDES = Dict(1 => :dot, 2 => :dash, 3 => :dashdot)

default(fontfamily="sans-serif", titlefontsize=10, guidefontsize=9, tickfontsize=8, legendfontsize=7, grid=true,
        gridcolor=GRID, gridalpha=1, framestyle=:axes, foreground_color_legend=nothing, background_color=:white, dpi=200)

"fits: method => (mean, var): KP mean and 95% interval, Cholesky mean, ground truth and observations."
function posterior_plot!(p, xs, truth, X, y, fits, title)
    plot!(p, xs, truth; color=INK, ls=:dot, lw=1.2, label="ground truth")
    for (name, (mu, v)) in fits
        st, sd = STYLE[name], sqrt.(max.(v, 0))
        if name == "KP"
            plot!(p, xs, mu; ribbon=1.96 .* sd, fillalpha=0.18, color=st.color, lw=st.lw,
                  label="KP mean and 95% interval")
        else
            plot!(p, xs, mu; color=st.color, ls=st.ls, lw=st.lw, label="$name mean")
        end
    end
    scatter!(p, X, y; color=colorant"#d62728", ms=3, msw=0, label="observations")
    plot!(p; xlabel="x", title=title)
    p
end

"draws: method => F (m x p), one legend entry per method; band: optional (mean, var, label) of target distribution."
function draws_plot!(p, xs, draws, title; X=nothing, y=nothing, truth=nothing, band=nothing)
    if band !== nothing
        bmu, bv, blab = band
        bsd = sqrt.(max.(bv, 0))
        plot!(p, xs, bmu; ribbon=1.96 .* bsd, fillcolor=GRID, fillalpha=0.9, color=INK2, lw=0.8, label=blab)
    end
    truth === nothing || plot!(p, xs, truth; color=INK, ls=:dot, lw=1.2, label="ground truth")
    for (name, F) in draws
        st = STYLE[name]
        plot!(p, xs, F; color=st.color, ls=st.ls, lw=0.9, alpha=0.8,
              label=["$name draws" fill("", 1, size(F, 2) - 1)])
    end
    X === nothing || scatter!(p, X, y; color=colorant"#d62728", ms=3, msw=0, label="observations")
    plot!(p; xlabel="x", title=title)
    p
end

"series: method => values or (mean, sd) at x pairs, the quantity named by the title; slopes: reference powers."
function metric_plot!(p, x, series, xlabel, title; slopes=(), logx=true, logy=true)
    first_vals = nothing
    ordered = [k for k in ("KP", "Cholesky") if haskey(series, k)]
    append!(ordered, [k for k in keys(series) if !(k in ordered)])
    for name in ordered
        v = series[name]
        st = STYLE[name]
        mu, sd = v isa Tuple ? (Float64.(v[1]), Float64.(v[2])) : (Float64.(v), nothing)
        first_vals === nothing && (first_vals = mu)
        mask = isfinite.(mu) .& (logy ? (mu .> 0) : trues(length(mu)))
        any(mask) || continue
        kw = if sd === nothing
            (;)
        elseif logy
            s = sd[mask]; m = mu[mask]
            (; yerror=(min.(s, 0.99 .* m), s))
        else
            (; yerror=sd[mask])
        end
        plot!(p, x[mask], mu[mask]; color=st.color, marker=st.marker, ls=st.ls, lw=st.lw, ms=4, label=name, kw...)
    end
    if first_vals !== nothing
        idx = findfirst(v -> isfinite(v) && (!logy || v > 0), first_vals)
        if idx !== nothing
            x0, y0 = Float64(x[idx]), Float64(first_vals[idx])
            for s in slopes
                plot!(p, x, y0 .* (Float64.(x) ./ x0) .^ s; color=INK2, ls=get(GUIDES, s, :dot), lw=0.9,
                      label=s == 1 ? "O(n)" : "O(n^$s)")
            end
        end
    end
    plot!(p; xlabel=xlabel, title=title, xscale=logx ? :log10 : :identity, yscale=logy ? :log10 : :identity)
    p
end

save_figure(p, path) = savefig(p, path)
