function x = rand_beta(a, b, sz)
% Beta(a, b) draws in base MATLAB (no Statistics Toolbox) via Marsaglia-Tsang Gamma
if nargin < 3, sz = size(a); end
if isscalar(a), a = repmat(a, sz); end
if isscalar(b), b = repmat(b, sz); end
ga = rand_gamma(a);
gb = rand_gamma(b);
x = ga ./ (ga + gb);
end

function g = rand_gamma(a)
small = a < 1;
a_eff = a;
a_eff(small) = a(small) + 1;
d = a_eff - 1/3;
c = 1 ./ sqrt(9 * d);
g = zeros(size(a));
rem = true(size(a));
while any(rem, 'all')
    idx = find(rem);
    n = numel(idx);
    x = randn(n, 1);
    v = 1 + c(idx) .* x;
    ok_v = v > 0;
    idx = idx(ok_v); x = x(ok_v); v = v(ok_v).^3;
    if isempty(idx), continue; end
    u = rand(numel(idx), 1);
    acc = (u < 1 - 0.0331 * x.^4) | (log(u) < 0.5 * x.^2 + d(idx) .* (1 - v + log(v)));
    g(idx(acc)) = d(idx(acc)) .* v(acc);
    rem(idx(acc)) = false;
end
if any(small, 'all')
    g(small) = g(small) .* rand(nnz(small), 1).^(1 ./ a(small));
end
end
