function s = data
% Synthetic designs for prior and posterior sampling: Griewank function on [0,1]^d with noise s.d. NOISE.
s.PARTS = {'prior_draws', 'prior_accuracy', 'prior_sensitivity', 'prior_scaling', 'posterior_draws', ...
    'posterior_accuracy', 'posterior_sensitivity', 'scaling_1d', 'posterior_sparse_grid'};
s.K = 5;
s.SEEDS = 5;
s.NOISE = 0.1;
s.KS = [3, 5, 7];
s.PRIOR_RHO = 0.2;
s.PRIOR_ACC_N = [50, 100, 200, 500, 1000];
s.POST_ACC_N = [100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6];
s.P_ACC = 512;
s.SENS_N = 100; s.SENS_KS = [3, 5, 7];
s.SENS_RHOS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3];
s.POST_SENS_RHOS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
s.POST_SENS_TAU2S = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
s.SIZES = [100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8];
s.BIG = 1e6;
s.SCALING_RHO = 20; s.P_TIME = 16; s.N_DENSE = 1e4;
s.N_POST = 20; s.M_POST = 200; s.P_POST = 4096;
s.SG_DIMS = [2, 4, 8]; s.SG_N = 2^10; s.SG_M = 200;

s.griewank = @griewank;
s.f = @(x) griewank(x(:));
s.nodes = @(n) sort(rand_beta(0.7, 0.7, [n, 1]));
s.scaling_design = @(n) (0:n-1)' + 0.6 * rand(n, 1) - 0.3;
s.scaling_data = @(n) scaling_data(n);
s.train_inputs = @train_inputs;
s.test_inputs = @(m, d) rand_beta(0.7, 0.7, [m, d]);
s.rho = @(d) 0.25 * sqrt(d);
end

function y = griewank(X)
Z = 10 * X - 5;
y = sum(Z.^2, 2) / 4000 - prod(cos(Z ./ sqrt(1:size(Z, 2))), 2) + 1;
end

function [X, y, xs] = scaling_data(n)
X = (0:n-1)' + 0.6 * rand(n, 1) - 0.3;
y = sin(X / 50);
xs = sort((n - 1) * rand(200, 1));
end

function X = train_inputs(n, d)
a = 5 + 10 * (rand(n, d) > 0.5);
X = rand_beta(a, 20 - a, [n, d]);
end
