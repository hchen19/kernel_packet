# Pure-Julia derivative-free optimizers (Brent 1-D and box-clamped Nelder-Mead) so KernelPackets needs no Optim dep.

"Brent's 1-D minimization on [a, b]; returns (minimizer, minimum)."
function brent(f, a::Real, b::Real; tol::Real=1e-3, maxiter::Int=200)
    cgold = 0.5 * (3.0 - sqrt(5.0))
    lo, hi = Float64(min(a, b)), Float64(max(a, b))
    x = w = v = lo + cgold * (hi - lo)
    fx = fw = fv = Float64(f(x))
    d = e = 0.0
    for _ in 1:maxiter
        xm = 0.5 * (lo + hi)
        tol1 = sqrt(eps(Float64)) * abs(x) + tol / 3.0
        tol2 = 2.0 * tol1
        abs(x - xm) <= tol2 - 0.5 * (hi - lo) && break
        parab = false
        if abs(e) > tol1
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0
                p = -p
            else
                q = -q
            end
            if isfinite(p) && abs(p) < abs(0.5 * q * e) && p > q * (lo - x) && p < q * (hi - x)
                e = d
                d = p / q
                u = x + d
                if (u - lo < tol2) || (hi - u < tol2)
                    d = x < xm ? tol1 : -tol1
                end
                parab = true
            end
        end
        if !parab
            e = x < xm ? hi - x : lo - x
            d = cgold * e
        end
        u = abs(d) >= tol1 ? x + d : x + (d >= 0.0 ? tol1 : -tol1)
        fu = Float64(f(u))
        if fu <= fx
            u < x ? (hi = x) : (lo = x)
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else
            u < x ? (lo = u) : (hi = u)
            if fu <= fw || w == x
                v, fv = w, fw
                w, fw = u, fu
            elseif fu <= fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
    (minimizer=x, minimum=fx)
end

"Box-clamped Nelder-Mead simplex minimization on [lo, hi]; returns (minimizer, minimum)."
function nelder_mead(f, x0::AbstractVector, lo::AbstractVector, hi::AbstractVector;
                     xtol::Real=1e-3, ftol::Real=1e-3, maxiter::Int=500)
    n = length(x0)
    lov, hiv = Float64.(lo), Float64.(hi)
    evalf(z) = Float64(f(clamp.(z, lov, hiv)))
    v0 = clamp.(Float64.(x0), lov, hiv)
    V = Vector{Vector{Float64}}(undef, n + 1)
    V[1] = v0
    for i in 1:n
        vi = copy(v0)
        step = max(0.05 * abs(v0[i]), 0.05)
        vi[i] = v0[i] + step <= hiv[i] ? v0[i] + step : max(lov[i], v0[i] - step)
        V[i + 1] = clamp.(vi, lov, hiv)
    end
    fv = [evalf(v) for v in V]
    for _ in 1:maxiter
        ord = sortperm(fv)
        V .= V[ord]
        fv .= fv[ord]
        dx = maximum(i -> maximum(abs, V[i] .- V[1]), 2:(n + 1))
        (dx <= xtol && (fv[end] - fv[1]) <= ftol) && break
        xc = sum(@view V[1:n]) ./ n
        xr = clamp.(xc .+ (xc .- V[end]), lov, hiv)
        fr = evalf(xr)
        if fv[1] <= fr < fv[n]
            V[end], fv[end] = xr, fr
        elseif fr < fv[1]
            xe = clamp.(xc .+ 2.0 .* (xr .- xc), lov, hiv)
            fe = evalf(xe)
            if fe < fr
                V[end], fv[end] = xe, fe
            else
                V[end], fv[end] = xr, fr
            end
        elseif fr < fv[end]
            xc_out = clamp.(xc .+ 0.5 .* (xr .- xc), lov, hiv)
            fc = evalf(xc_out)
            if fc <= fr
                V[end], fv[end] = xc_out, fc
            else
                for i in 2:(n + 1)
                    V[i] = clamp.(V[1] .+ 0.5 .* (V[i] .- V[1]), lov, hiv)
                    fv[i] = evalf(V[i])
                end
            end
        else
            xc_in = clamp.(xc .- 0.5 .* (xc .- V[end]), lov, hiv)
            fc = evalf(xc_in)
            if fc < fv[end]
                V[end], fv[end] = xc_in, fc
            else
                for i in 2:(n + 1)
                    V[i] = clamp.(V[1] .+ 0.5 .* (V[i] .- V[1]), lov, hiv)
                    fv[i] = evalf(V[i])
                end
            end
        end
    end
    idx = argmin(fv)
    (minimizer=clamp.(V[idx], lov, hiv), minimum=fv[idx])
end
