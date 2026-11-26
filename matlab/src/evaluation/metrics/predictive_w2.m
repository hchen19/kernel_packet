function w = predictive_w2(y, mu, v)
% Per-point 2-Wasserstein distance W2(N(mu, diag(v)), delta_y) / sqrt(m)
w = sqrt(mean((double(mu(:)) - double(y(:))).^2 + max(double(v(:)), 0)));
end
