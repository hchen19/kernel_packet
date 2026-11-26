function out = predictive(y, mu, v, y_train)
% Every predictive metric of the test targets y under N(mu, v) (variances floored at 1e-12)
Z95 = 1.959963984540054;
y = double(y(:)); mu = double(mu(:)); v = max(double(v(:)), 1e-12); y_train = double(y_train(:));
sd = sqrt(v);
r = y - mu;
z = r ./ sd;
crps = sd .* (z .* erf(z / sqrt(2)) + sqrt(2 / pi) * exp(-z.^2 / 2) - 1 / sqrt(pi));
lo = mu - Z95 * sd; hi = mu + Z95 * sd;
score = (hi - lo) + (2 / 0.05) * ((lo - y) .* (y < lo) + (y - hi) .* (y > hi));
mt = mean(y_train); vt = var(y_train, 1);
cov_v = coverage(y, mu, v, Z95);
nlpd_v = nlpd(y, mu, v);
out = struct('rmse', rmse(mu, y), 'mae', mean(abs(r)), 'smse', mean(r.^2) / var(y, 1), ...
    'nlpd', nlpd_v, 'msll', nlpd_v - nlpd(y, repmat(mt, size(y)), repmat(vt, size(y))), ...
    'crps', mean(crps), 'interval_score', mean(score), 'coverage', cov_v, ...
    'coverage_error', abs(cov_v - 0.95), 'zscore_ms', mean(z.^2));
end
