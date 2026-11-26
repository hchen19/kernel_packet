function w = wasserstein2(Kh, K, dm)
% W2(N(m + dm, Kh), N(m, K)) = sqrt(|dm|^2 + tr K + tr Kh - 2 tr (K^(1/2) Kh K^(1/2))^(1/2)), the error measure of
% samplers (Wilson et al. 2021)
if nargin < 3, dm = 0; end
S = psd_sqrt(K);
w = sqrt(max(sum(dm(:).^2) + trace(K) + trace(Kh) - 2 * trace(psd_sqrt(S * Kh * S)), 0));
end

function R = psd_sqrt(M)
[V, D] = eig((M + M') / 2);
R = V * diag(sqrt(max(diag(D), 0))) * V';
end
