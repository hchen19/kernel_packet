# Kernel packets against dense Gaussian process algebra: backward error of the solves at any conditioning, values
# against dense solves where K is well conditioned (as python/tests/test_kernel_packets.py)
#   julia --project=julia julia/tests/runtests.jl
using KernelPackets, KernelPackets.Evaluation, LinearAlgebra, Random, Statistics, Test

backward_error(K, W, Y) = maximum(abs, K * W - Y) / (maximum(sum(abs, K, dims=2)) * maximum(abs, W))

@testset "KernelPackets" begin
    @testset "solve k=$k n=$n rho=$rho method=$method" for k in (3, 5, 7, 9), n in (4, 30, 300),
            rho in (0.03, 1.0, 20.0), method in (:stable, :original)
        rng = MersenneTwister(k * n)
        x = sort(rand(rng, n)); K = matern(x, x, k, rho); Y = randn(rng, n, 3)
        m = KernelPacket(x, k, rho; method=method)
        @test backward_error(K, solve(m, Y), Y) < 1e-13
        cond(K) < 1e8 && @test abs(logdet(m) - logdet(cholesky(Symmetric(K)))) < 1e-14 * n * cond(K)
    end

    @testset "posterior k=$k n=$n method=$method" for k in (3, 5, 7), n in (4, 30), method in (:stable, :original)
        rng = MersenneTwister(k * n)
        x, xs, y = sort(rand(rng, n)), 1.4 .* rand(rng, 50) .- 0.2, randn(rng, n)
        rho = 0.01 * (k - 2)
        K, Ks = matern(x, x, k, rho), matern(xs, x, k, rho)
        ck = cond(K)
        mu, v, C = posterior(KernelPacket(x, k, rho; method=method), xs, y; var=true, cov=true)
        @test maximum(abs, mu - Ks * (K \ y)) < max(1e-10, 1e-14 * ck)
        @test maximum(abs, C - (matern(xs, xs, k, rho) - Ks * (K \ Ks'))) < max(1e-10, 1e-14 * ck)
        @test maximum(abs, v - diag(C)) < max(1e-13, 1e-15 * ck)
    end

    @testset "posterior noise k=$k rho=$rho" for k in (3, 5, 7), rho in (0.05, 1.0)
        rng = MersenneTwister(k)
        n, tau2 = 40, 1e-6
        x, xs, y = sort(rand(rng, n)), 1.4 .* rand(rng, 30) .- 0.2, randn(rng, n)
        K, Ks = matern(x, x, k, rho) + tau2 * I, matern(xs, x, k, rho)
        mu, v, C = posterior(KernelPacket(x, k, rho; tau2=tau2), xs, y; var=true, cov=true)
        ref_mu = Ks * (K \ y)
        ref_C = matern(xs, xs, k, rho) - Ks * (K \ Ks')
        @test maximum(abs, mu - ref_mu) < max(1e-9 * maximum(abs, ref_mu) + 1e-12, 1e-14 * cond(K))
        @test maximum(abs, C - ref_C) < 1e-9
        @test maximum(abs, v - diag(ref_C)) < 1e-9
    end

    @testset "smoother route k=$k" for k in (3, 5, 7)
        rng = MersenneTwister(10 + k)
        n, m_pts, rho, tau2 = 200, 25, 0.2, 1e-3
        x = sort(rand_beta(rng, 5.0, 15.0, n))
        xs, y = 1.2 .* rand(rng, m_pts) .- 0.1, randn(rng, n)
        kp_m = KernelPacket(x, k, 0.5; tau2=1e-2)
        set_params!(kp_m, rho, tau2)
        K, Ks = matern(x, x, k, rho) + tau2 * I, matern(xs, x, k, rho)
        mu, v, C = posterior(kp_m, xs, y; var=true, cov=true)
        ref_C = matern(xs, xs, k, rho) - Ks * (K \ Ks')
        @test maximum(abs, mu - Ks * (K \ y)) < 1e-10
        @test maximum(abs, C - ref_C) < 1e-10
        @test maximum(abs, v - diag(ref_C)) < 1e-10
        @test backward_error(K, solve(kp_m, y), y) < 1e-12
        F_ones = ones(n, 1)
        beta = (F_ones' * (K \ F_ones)) \ (F_ones' * (K \ y))
        r = y - F_ones * beta
        s2 = (r' * (K \ r))[1] / n
        L, b, s = loglike(kp_m, y)
        @test L ≈ -0.5 * (n * log(s2) + logdet(cholesky(Symmetric(K))) + n) rtol = 1e-9
        @test b[1] ≈ beta[1] rtol = 1e-9
        @test s ≈ s2 rtol = 1e-9
        draws = posterior_sample(kp_m, xs[1:5], y, 10000; rng=MersenneTwister(0))
        @test maximum(abs, vec(mean(draws, dims=2)) - mu[1:5]) < 5e-2 * sqrt(maximum(v[1:5]))
        @test size(sample_(KernelPacket(x, k, rho; tau2=tau2), Matrix(1.0I, n, n))) == (n, n)
        m_prior = KernelPacket(x, k, rho; tau2=tau2)
        prior_sample(m_prior, 2; rng=MersenneTwister(0))
        @test nbytes(m_prior) == 8 * n * (1 + (k - 1))
    end

    @testset "sparse grid d=$d" for (d, eta) in ((1, 4), (2, 5), (3, 6), (4, 7))
        g = SparseGrid(d, eta; k=5, rho=[0.5, 0.9, 1.3, 0.7][1:d])
        K, Y = matern(g.X, g.X, 5, g.rho), randn(MersenneTwister(d), length(g), 2)
        @test backward_error(K, solve(g, Y), Y) < 1e-13
        @test logdet(g) ≈ logdet(cholesky(Symmetric(K))) rtol = 1e-9
        @test nbytes(g) > 0
    end

    @testset "sparse grid for_data" begin
        rng = MersenneTwister(42)
        X = rand(rng, 200, 2)
        g = for_data(SparseGrid, X; k=5, rho=0.6, tau2=1e-2, n_max=100)
        @test 1 <= length(g) <= 100
        @test g.eta >= 2
    end

    @testset "sparse-grid regression d=$d" for d in (2, 3)
        rng = MersenneTwister(d)
        X, s2, t2 = rand(rng, 300, d), 1.3, 1e-2
        y, xs = vec(sum(sin.(4X), dims=2)) .+ 0.1 .* randn(rng, 300), rand(rng, 40, d)
        g = SparseGrid(d, d + 3; k=5, rho=0.6)
        m = SparseGridRegression(X, y, g; sigma2=s2, tau2=t2)
        Kuu, Kux, Ms, Mss = matern(g.X, g.X, 5, g.rho), matern(g.X, X, 5, g.rho), matern(xs, g.X, 5, g.rho), matern(xs, xs, 5, g.rho)
        A = Kuu + s2 / t2 * Kux * Kux'
        A += size(A, 1) * eps(Float64) * mean(diag(A)) * I   # the method's jitter
        Q = Kux' * (Kuu \ Kux); C = s2 * Q + t2 * I
        elbo = -0.5 * (y' * (C \ y) + logdet(cholesky(Symmetric(C))) + 300 * log(2π)) - 0.5 * s2 * (300 - tr(Q)) / t2
        mu, v, cov_m = predict(m, xs; var=true, cov=true)
        ca = cond(A)
        @test maximum(abs, mu - s2 * Ms * (A \ (Kux * y)) / t2) < max(1e-9, 1e-14 * ca)
        @test maximum(abs, v - s2 .* (1 .- vec(sum(Ms .* (Kuu \ Ms')', dims=2)) .+ vec(sum(Ms .* (A \ Ms')', dims=2)))) < max(1e-9, 1e-14 * ca)
        @test maximum(abs, cov_m - s2 .* (Mss - Ms * (Kuu \ Ms') + Ms * (A \ Ms'))) < max(1e-9, 1e-14 * ca)
        @test m.elbo ≈ elbo rtol = 1e-7
        @test nbytes(m) > nbytes(g)
    end

    @testset "prior sample k=$k n=$n method=$method" for k in (3, 5, 7), n in (4, 40), method in (:stable, :original)
        x, rho = sort(rand(MersenneTwister(n), n)), 0.01 * (k - 2)
        K = matern(x, x, k, rho)
        L = sample_(KernelPacket(x, k, rho; method=method), Matrix(1.0I, n, n))
        tol = method === :stable ? 1e-12 : max(1e-12, 1e-13 * cond(K))
        @test maximum(abs, L * L' - K) < tol
    end

    @testset "posterior sample method=$method tau2=$tau2" for (method, tau2) in ((:stable, 0.0), (:stable, 1e-2), (:original, 0.0))
        rng = MersenneTwister(3)
        x, xs, y = sort(rand(rng, 30)), rand(rng, 5), randn(rng, 30)
        m = KernelPacket(x, 5, 0.2; tau2=tau2, method=method)
        F = posterior_sample(m, xs, y, 20000; rng=MersenneTwister(0))
        mu, C = posterior(m, xs, y; cov=true)
        @test maximum(abs, vec(mean(F, dims=2)) - mu) < 5e-2 * sqrt(maximum(diag(C)))
        @test maximum(abs, cov(F; dims=2) - C) < 5e-2 * maximum(abs, C)
    end

    @testset "predict, loglike, and mle" begin
        rng = MersenneTwister(5)
        X, U, w = rand(rng, 40, 3), rand(rng, 50, 3), randn(rng, 50, 2)
        @test maximum(abs, predict(X, U, w, 5, [0.3, 0.5, 0.7]) - matern(X, U, 5, [0.3, 0.5, 0.7]) * w) < 1e-12
        x = collect(range(0, 1, 50)); y = sin.(6x); F = ones(50, 1)
        for (method, tau2) in ((:stable, 0.0), (:stable, 1e-3), (:original, 0.0))
            m = KernelPacket(x, 5, 0.2; tau2=tau2, method=method)
            K = matern(x, x, 5, 0.2) + tau2 * I
            beta = (F' * (K \ F)) \ (F' * (K \ y)); r = y - F * beta; s2 = r' * (K \ r) / 50
            L, b, s = loglike(m, y)
            @test L ≈ -0.5 * (50 * log(s2[1]) + logdet(cholesky(Symmetric(K))) + 50) rtol = 1e-8
            @test b ≈ vec(beta) rtol = 1e-8
            @test s ≈ s2[1] rtol = 1e-8
        end
        m_mle = KernelPacket(x, 5, 0.1; tau2=1e-4)
        theta, L_hat, L_init = mle(m_mle, y; bounds=(1e-2, 5.0), tau2_bounds=(1e-8, 1.0))
        @test L_hat >= L_init
        @test 1e-2 <= theta.rho <= 5.0
        @test 1e-8 <= theta.tau2 <= 1.0
    end
end
