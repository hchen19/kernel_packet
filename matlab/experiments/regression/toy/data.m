function s = data
% Toy regression data: standard test functions of GP regression (Surjanovic & Bingham, Virtual Library
% of Simulation Experiments, https://www.sfu.ca/~ssurjano/).
%   1-D: the Gramacy & Lee (2012) function sin(10 pi t)/(2t) + (t-1)^4, t = 0.5 + 2x on x in [0,1].
%   d-D: the Genz (1984) oscillatory function cos(2 pi w + sum_i a_i x_i) on [0,1]^d, a_i = 9/i^2, w = 0.5.
s.PARTS = {'posterior_1d', 'accuracy_1d', 'scaling_1d', 'sparse_grid', 'scaling_nd'};
s.K = 5;
s.KS = [3, 5, 7];
s.SEEDS = 5;
s.M = 1000;
s.N_1D = 60;
s.N_ACC = [100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8];
s.M_W2 = 300;
s.SIZES = s.N_ACC;
s.BIG = 1e6;
s.SCALING_RHO = 20; s.N_DENSE = 1e4;
s.NOISE = 0.05;
s.SG_N = 2000; s.SG_DIMS = [2, 4];
s.SG_LEVELS = struct('d2', 2:6, 'd4', 1:5);
s.INIT = struct('rho', 0.5, 'sigma2', 1.0, 'tau2', s.NOISE^2);
s.ND_SIZES = [100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6];

s.f = @(x) sin(10*pi*(0.5 + 2*x)) ./ (2*(0.5 + 2*x)) + (0.5 + 2*x - 1).^4;
s.train_inputs = @train_inputs;
s.test_inputs = @(m, d) rand_beta(0.7, 0.7, [m, d]);
s.data_1d = @(n, noise) data_1d(n, noise, s);
s.scaling_design = @(n, m) scaling_design(n, m);
s.f_nd = @(X) cos(pi + X * (9 ./ (1:size(X, 2))'.^2));
s.data_nd = @(d, n) data_nd(d, n, s);
end

function X = train_inputs(n, d)
a = 5 + 10 * (rand(n, d) > 0.5);
X = rand_beta(a, 20 - a, [n, d]);
end

function [X, y, xs, ys] = data_1d(n, noise, s)
if nargin < 2, noise = 0; end
X = sort(train_inputs(n, 1));
xs = rand_beta(0.7, 0.7, [s.M, 1]);
y = s.f(X) + noise * randn(n, 1);
ys = s.f(xs);
end

function [X, y, xs, ys] = scaling_design(n, m)
X = (0:n-1)' + 0.6 * rand(n, 1) - 0.3;
xs = (n - 1) * rand(m, 1);
y = sin(X / 50); ys = sin(xs / 50);
end

function [X, y, xs, ys] = data_nd(d, n, s)
X = train_inputs(n, d); xs = rand_beta(0.7, 0.7, [s.M, d]);
y = s.f_nd(X) + s.NOISE * randn(n, 1);
ys = s.f_nd(xs);
end
