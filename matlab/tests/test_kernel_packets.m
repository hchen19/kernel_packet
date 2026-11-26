function tests = test_kernel_packets
% Kernel packets against dense Gaussian process algebra: backward error of the solves at any conditioning, values
% against dense solves where K is well conditioned (as python/tests/test_kernel_packets.py)
% (run: runtests('matlab/tests'))
tests = functiontests(localfunctions);
end

function setupOnce(~)
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));
kp_build;
end

function e = backward_error(K, W, Y)
e = max(abs(K*W - Y), [], 'all') / (max(sum(abs(K), 2)) * max(abs(W), [], 'all'));
end

function test_solve(tc)
for k = [3, 5, 7, 9]
    for n = [4, 30, 300]
        for rho = [0.03, 1, 20]
            for method = {'stable', 'original'}
                rng(k*n);
                x = sort(rand(n, 1)); K = matern(x, x, k, rho); Y = randn(n, 3);
                m = kernel_packet(x, k, rho, 0, method{1});
                tc.verifyLessThan(backward_error(K, m.solve(Y), Y), 1e-13);
                if cond(K) < 1e8
                    tc.verifyLessThan(abs(m.logdet() - 2*sum(log(diag(chol(K))))), 1e-14 * n * cond(K));
                end
            end
        end
    end
end
end

function test_posterior(tc)
for k = [3, 5, 7]
    for n = [4, 30]
        for method = {'stable', 'original'}
            rng(k*n);
            x = sort(rand(n, 1)); xnew = 1.4*rand(50, 1) - 0.2; y = randn(n, 1); rho = 0.01 * (k-2);
            K = matern(x, x, k, rho); Ks = matern(xnew, x, k, rho); ck = cond(K);
            m = kernel_packet(x, k, rho, 0, method{1});
            [mu, C] = posterior(m, xnew, y, true);
            [~, v] = posterior(m, xnew, y);
            tc.verifyLessThan(max(abs(mu - Ks * (K \ y))), max(1e-10, 1e-14 * ck));
            tc.verifyLessThan(max(abs(C - (matern(xnew, xnew, k, rho) - Ks * (K \ Ks'))), [], 'all'), max(1e-10, 1e-14 * ck));
            tc.verifyLessThan(max(abs(v - diag(C))), max(1e-13, 1e-15 * ck));
        end
    end
end
end

function test_posterior_noise(tc)
for k = [3, 5, 7]
    for rho = [0.05, 1.0]
        rng(k);
        n = 40; tau2 = 1e-6;
        x = sort(rand(n, 1)); xnew = 1.4*rand(30, 1) - 0.2; y = randn(n, 1);
        K = matern(x, x, k, rho) + tau2 * eye(n); Ks = matern(xnew, x, k, rho);
        m = kernel_packet(x, k, rho, tau2);
        [mu, C] = posterior(m, xnew, y, true);
        [~, v] = posterior(m, xnew, y);
        ref_mu = Ks * (K \ y);
        ref_C = matern(xnew, xnew, k, rho) - Ks * (K \ Ks');
        tc.verifyLessThan(max(abs(mu - ref_mu)), max(1e-9 * max(abs(ref_mu)) + 1e-12, 1e-14 * cond(K)));
        tc.verifyLessThan(max(abs(C - ref_C), [], 'all'), 1e-9);
        tc.verifyLessThan(max(abs(v - diag(ref_C))), 1e-9);
    end
end
end

function test_smoother_route(tc)
for k = [3, 5, 7]
    rng(10 + k);
    n = 200; m_pts = 25; rho = 0.2; tau2 = 1e-3;
    x = sort(rand_beta(5.0, 15.0, [n, 1]));
    xnew = 1.2 * rand(m_pts, 1) - 0.1; y = randn(n, 1);
    kp_m = kernel_packet(x, k, 0.5, 1e-2);
    kp_m.set_params(rho, tau2);
    K = matern(x, x, k, rho) + tau2 * eye(n); Ks = matern(xnew, x, k, rho);
    [mu, C] = posterior(kp_m, xnew, y, true);
    [~, v] = posterior(kp_m, xnew, y);
    ref_C = matern(xnew, xnew, k, rho) - Ks * (K \ Ks');
    tc.verifyLessThan(max(abs(mu - Ks * (K \ y))), 1e-10);
    tc.verifyLessThan(max(abs(C - ref_C), [], 'all'), 1e-10);
    tc.verifyLessThan(max(abs(v - diag(ref_C))), 1e-10);
    tc.verifyLessThan(backward_error(K, kp_m.solve(y), y), 1e-12);
    F_ones = ones(n, 1);
    beta = (F_ones' * (K \ F_ones)) \ (F_ones' * (K \ y));
    r = y - F_ones * beta;
    s2 = r' * (K \ r) / n;
    [L, b, s] = loglike(kp_m, y);
    tc.verifyLessThan(abs(L / (-0.5 * (n * log(s2) + 2*sum(log(diag(chol(K)))) + n)) - 1), 1e-9);
    tc.verifyLessThan(abs(b - beta), 1e-9 * abs(beta) + 1e-12);
    tc.verifyLessThan(abs(s / s2 - 1), 1e-9);
    draws = posterior_sample(kp_m, xnew(1:5), y, 10000);
    tc.verifyLessThan(max(abs(mean(draws, 2) - mu(1:5))), 5e-2 * sqrt(max(v(1:5))));
    tc.verifyEqual(size(kernel_packet(x, k, rho, tau2).sample_(eye(n))), [n, n]);
    m_prior = kernel_packet(x, k, rho, tau2);
    prior_sample(m_prior, 2);
    tc.verifyEqual(m_prior.nbytes(), 8 * n * (1 + (k - 1)));
end
end

function test_sparse_grid(tc)
rhos = [0.5, 0.9, 1.3, 0.7];
for de = [1 4; 2 5; 3 6; 4 7]'
    d = de(1); g = sparse_grid(d, de(2), 5, rhos(1:d));
    K = matern(g.X, g.X, 5, g.rho); Y = randn(size(g.X, 1), 2);
    tc.verifyLessThan(backward_error(K, g.solve(Y), Y), 1e-13);
    tc.verifyLessThan(abs(g.logdet() / (2*sum(log(diag(chol(K))))) - 1), 1e-9);
    tc.verifyGreaterThan(g.nbytes(), 0);
end
rng(42);
g_auto = sparse_grid.for_data(rand(200, 2), 5, 0.6, 1e-2, 100);
tc.verifyGreaterThanOrEqual(size(g_auto.X, 1), 1);
tc.verifyLessThanOrEqual(size(g_auto.X, 1), 100);
end

function test_sparse_grid_regression(tc)
for d = [2, 3]
    rng(d);
    X = rand(300, d); s2 = 1.3; t2 = 1e-2; y = sum(sin(4*X), 2) + 0.1 * randn(300, 1); xs = rand(40, d);
    g = sparse_grid(d, d + 3, 5, 0.6);
    m = sparse_grid_regression(X, y, g, s2, t2);
    Kuu = matern(g.X, g.X, 5, g.rho); Kux = matern(g.X, X, 5, g.rho);
    Ms = matern(xs, g.X, 5, g.rho); Mss = matern(xs, xs, 5, g.rho);
    A = Kuu + s2 / t2 * (Kux * Kux');
    A = A + size(A, 1) * eps * mean(diag(A)) * eye(size(A, 1));   % the method's jitter
    Q = Kux' * (Kuu \ Kux); C = s2 * Q + t2 * eye(300);
    elbo = -0.5 * (y' * (C \ y) + 2*sum(log(diag(chol(C)))) + 300 * log(2*pi)) - 0.5 * s2 * (300 - trace(Q)) / t2;
    [mu, v] = m.predict(xs);
    [mu_c, cov_m] = m.predict(xs, true);
    ca = cond(A);
    tc.verifyLessThan(max(abs(mu - s2 * Ms * (A \ (Kux * y)) / t2)), max(1e-9, 1e-14 * ca));
    tc.verifyLessThan(max(abs(mu_c - mu)), 1e-13);
    tc.verifyLessThan(max(abs(v - s2 * (1 - sum(Ms .* (Kuu \ Ms')', 2) + sum(Ms .* (A \ Ms')', 2)))), max(1e-9, 1e-14 * ca));
    tc.verifyLessThan(max(abs(cov_m - s2 * (Mss - Ms * (Kuu \ Ms') + Ms * (A \ Ms'))), [], 'all'), max(1e-9, 1e-14 * ca));
    tc.verifyLessThan(abs(m.elbo / elbo - 1), 1e-7);
    tc.verifyGreaterThan(m.nbytes(), g.nbytes());
end
end

function test_prior_sample(tc)
% the draws are linear in the normals: their implied factor L (L*L' = K) from unit vectors, K well conditioned
for k = [3, 5, 7]
    for n = [4, 40]
        for method = {'stable', 'original'}
            rng(n); x = sort(rand(n, 1)); rho = 0.01 * (k-2);
            K = matern(x, x, k, rho);
            m = kernel_packet(x, k, rho, 0, method{1});
            L = m.sample_(eye(n));
            tol = 1e-12; if strcmp(method{1}, 'original'), tol = max(1e-12, 1e-13 * cond(K)); end
            tc.verifyLessThan(max(abs(L*L' - K), [], 'all'), tol);
        end
    end
end
end

function test_posterior_sample(tc)
cases = {{'stable', 0}, {'stable', 1e-2}, {'original', 0}};
for c = cases
    method = c{1}{1}; tau2 = c{1}{2};
    rng(3);
    x = sort(rand(30, 1)); xnew = rand(5, 1); y = randn(30, 1);
    m = kernel_packet(x, 5, 0.2, tau2, method);
    F = posterior_sample(m, xnew, y, 20000);
    [mu, C] = posterior(m, xnew, y, true);
    tc.verifyLessThan(max(abs(mean(F, 2) - mu)), 5e-2 * sqrt(max(diag(C))));
    tc.verifyLessThan(max(abs(cov(F') - C), [], 'all'), 5e-2 * max(abs(C), [], 'all'));
end
end

function test_predict_loglike(tc)
rng(5);
X = rand(40, 3); U = rand(50, 3); w = randn(50, 2);
tc.verifyLessThan(max(abs(predict(X, U, w, 5, [0.3, 0.5, 0.7]) - matern(X, U, 5, [0.3, 0.5, 0.7]) * w), [], 'all'), 1e-12);
x = linspace(0, 1, 50)'; y = sin(6 * x); F = ones(50, 1);
cases = {{'stable', 0}, {'stable', 1e-3}, {'original', 0}};
for c = cases
    method = c{1}{1}; tau2 = c{1}{2};
    m = kernel_packet(x, 5, 0.2, tau2, method);
    K = matern(x, x, 5, 0.2) + tau2 * eye(50);
    beta = (F' * (K \ F)) \ (F' * (K \ y)); r = y - F * beta; s2 = r' * (K \ r) / 50;
    [L, b, s] = loglike(m, y);
    tc.verifyLessThan(abs(L / (-0.5 * (50 * log(s2) + 2*sum(log(diag(chol(K)))) + 50)) - 1), 1e-8);
    tc.verifyLessThan(abs(b - beta), 1e-8); tc.verifyLessThan(abs(s / s2 - 1), 1e-8);
end
m_mle = kernel_packet(x, 5, 0.1, 1e-4);
[theta, L_hat, L_init] = mle(m_mle, y, [], [1e-2, 5], [1e-8, 1]);
tc.verifyGreaterThanOrEqual(L_hat, L_init);
tc.verifyGreaterThanOrEqual(theta.rho, 1e-2 - 1e-12);
tc.verifyLessThanOrEqual(theta.rho, 5 + 1e-12);
tc.verifyGreaterThanOrEqual(theta.tau2, 1e-8 - 1e-16);
tc.verifyLessThanOrEqual(theta.tau2, 1 + 1e-12);
end
