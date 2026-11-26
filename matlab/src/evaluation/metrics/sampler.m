function out = sampler(L, K)
% cov_error, whitened_error, kl_gaussian and wasserstein2 of the implied factor L against the exact K
C = chol(K, 'lower');
R = C \ L;
M = R * R' - eye(size(K, 1));
lam = eig((M + M') / 2);
out = struct('cov_error', covariance_error(L, K), 'whitened_error', max(abs(lam)), ...
    'kl_gaussian', 0.5 * sum(lam - log1p(max(lam, -1 + 1e-300))), 'wasserstein2', factor_wasserstein2(L, K));
end

function w = factor_wasserstein2(L, K)
% W2(N(0, L*L'), N(0, K)) = ||K^(1/2)*U - L*V||_F where K^(1/2)*L = U*S*V' (no trace cancellation)
[V, D] = eig((K + K') / 2);
S = V * diag(sqrt(max(diag(D), 0))) * V';
[U, ~, W] = svd(S * L, 'econ');
w = norm(S * U - L * W, 'fro');
end
