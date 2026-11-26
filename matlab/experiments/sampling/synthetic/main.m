function main(varargin)
% Prior and posterior sampling on synthetic designs: exact kernel packets in 1-D and sparse-grid inducing points in d
% dimensions (KP), against the exact GP by Cholesky; random inputs, SEEDS replicates (mean and sd).
%   scripts/run_sampling.sh matlab synthetic [parts] [rho=.. sigma2=.. tau2=..]
%   scripts/run_sampling.sh matlab synthetic replot [parts]
s = data;
method = 'stable';
fixed = struct();
replot = false;
parts = {};
for a = varargin
    arg = char(a{1});
    if contains(arg, '=')
        kv = split(string(arg), '=');
        if kv(1) == "method", method = char(kv(2));
        else, fixed.(char(kv(1))) = str2double(kv(2)); end
    elseif strcmp(arg, 'replot')
        replot = true;
    else
        parts{end+1} = arg; %#ok<AGROW>
    end
end
if isempty(parts), parts = s.PARTS; end
here = fileparts(mfilename('fullpath'));
if strcmp(method, 'stable'), out = fullfile(here, 'results');
else, out = fullfile(here, 'results', method); end
if ~isfolder(out), mkdir(out); end
for p = parts
    if replot && strcmp(p{1}, 'posterior_sparse_grid'), continue; end
    fprintf('== %s\n', p{1});
    if replot, feval(['plot_' p{1}], s, out);
    else, feval(['part_' p{1}], s, out, method, fixed); end
end
end

function part_prior_draws(s, out, method, ~) %#ok<DEFNU>
rng(0); X = s.nodes(500);
curves = struct('X', X);
for k = s.KS
    rng(0); curves.(sprintf('KP_%d', k)) = prior_sample(kernel_packet(X, k, s.PRIOR_RHO, 0, method), 3);
    rng(1); curves.(sprintf('Cholesky_%d', k)) = dense_gp.prior_sample(X, k, s.PRIOR_RHO, 3);
end
save(fullfile(out, 'prior_draws_1d.mat'), '-struct', 'curves');
plot_prior_draws(s, out);
end

function plot_prior_draws(s, out)
c = load(fullfile(out, 'prior_draws_1d.mat'));
X = c.X;
fig = figure('Visible', 'off', 'Position', [0, 0, 500 * numel(s.KS), 380]);
for i = 1:numel(s.KS)
    k = s.KS(i);
    draws = struct('KP', c.(sprintf('KP_%d', k)), 'Cholesky', c.(sprintf('Cholesky_%d', k)));
    draws_plot(subplot(1, numel(s.KS), i), X, draws, ...
        sprintf('Matern-%d/2, \\rho = %g, n = %d', k - 2, s.PRIOR_RHO, numel(X)), ...
        [], [], [], {zeros(size(X)), ones(size(X)), 'prior mean and 95% interval'});
end
save_figure(fig, fullfile(out, 'prior_draws_1d.png'));
end

function part_prior_accuracy(s, out, method, ~) %#ok<DEFNU>
rows = struct([]);
for k = s.SENS_KS
    for n = s.PRIOR_ACC_N
        for seed = 0:s.SEEDS-1
            rng(seed);
            X = s.scaling_design(n);
            Kd = matern(X, X, k, s.SCALING_RHO); c = cond(Kd);
            Ls = struct('KP', kernel_packet(X, k, s.SCALING_RHO, 0, method).sample_(eye(n)), ...
                'Cholesky', chol(Kd, 'lower'));
            for mth = ["KP", "Cholesky"]
                row = merge_struct(struct('seed', seed, 'k', k, 'n', n, 'rho', s.SCALING_RHO, ...
                    'cond', c, 'method', mth), sampler(Ls.(mth), Kd));
                rows = [rows, row]; %#ok<AGROW>
            end
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'prior_accuracy');
plot_prior_accuracy(s, out);
end

function plot_prior_accuracy(s, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'prior_accuracy');
ns = unique([rows.n], 'stable');
nk = numel(s.SENS_KS);
fig = figure('Visible', 'off', 'Position', [0, 0, 1500, 380]);
for i = 1:nk
    k = s.SENS_KS(i);
    series = struct();
    for mth = ["KP", "Cholesky"]
        [~, pair] = mean_sd(rows, 'n', 'wasserstein2', 'method', mth, 'k', k);
        series.(mth) = {max(pair{1}, 1e-17), pair{2}};
    end
    metric_plot(subplot(1, nk, i), ns, series, 'number of points n', ...
        sprintf('W_2 to the exact prior, Matern-%d/2', k - 2));
end
save_figure(fig, fullfile(out, 'prior_accuracy_1d.png'));
end

function part_prior_sensitivity(s, out, method, ~) %#ok<DEFNU>
rows = struct([]);
for k = s.SENS_KS
    for r = s.SENS_RHOS
        for seed = 0:s.SEEDS-1
            rng(seed);
            X = s.nodes(s.SENS_N);
            Kd = matern(X, X, k, r); c = cond(Kd);
            Ls = struct('KP', kernel_packet(X, k, r, 0, method).sample_(eye(s.SENS_N)), ...
                'Cholesky', chol(Kd, 'lower'));
            for mth = ["KP", "Cholesky"]
                row = merge_struct(struct('seed', seed, 'k', k, 'rho', r, 'n', s.SENS_N, ...
                    'cond', c, 'method', mth), sampler(Ls.(mth), Kd));
                rows = [rows, row]; %#ok<AGROW>
            end
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'prior_sensitivity');
plot_prior_sensitivity(s, out);
end

function plot_prior_sensitivity(s, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'prior_sensitivity');
rhos = unique([rows.rho], 'stable');
nk = numel(s.SENS_KS);
fig = figure('Visible', 'off', 'Position', [0, 0, 1500, 380]);
for i = 1:nk
    k = s.SENS_KS(i);
    series = struct();
    for mth = ["KP", "Cholesky"]
        [~, pair] = mean_sd(rows, 'rho', 'wasserstein2', 'method', mth, 'k', k);
        series.(mth) = {max(pair{1}, 1e-17), pair{2}};
    end
    metric_plot(subplot(1, nk, i), rhos, series, 'lengthscale \rho', ...
        sprintf('W_2 to the exact prior, Matern-%d/2, n = %d', k - 2, s.SENS_N));
end
save_figure(fig, fullfile(out, 'prior_sensitivity_1d.png'));
end

function part_prior_scaling(s, out, method, ~) %#ok<DEFNU>
rows = struct([]);
for n = s.SIZES
    nseeds = min(s.SEEDS, 3); if n > s.BIG, nseeds = 1; end
    for seed = 0:nseeds-1
        rng(seed);
        X = s.scaling_design(n);
        [t, m] = timed(@() run_kp_prior(X, s, method, seed), 0);
        rows = [rows, struct('seed', seed, 'method', "KP", 'n', n, 'draws', s.P_TIME, 'time', t, ...
            'storage_mb', m.nbytes() / 2^20)]; %#ok<AGROW>
        if n <= s.N_DENSE
            rep = 1; if n >= 5000, rep = 0; end
            rng(seed);
            t = timed(@() dense_gp.prior_sample(X, s.K, s.SCALING_RHO, s.P_TIME), rep);
            rows = [rows, struct('seed', seed, 'method', "Cholesky", 'n', n, 'draws', s.P_TIME, 'time', t, ...
                'storage_mb', 8 * n * n / 2^20)]; %#ok<AGROW>
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'prior_scaling');
plot_prior_scaling(s, out);
end

function m = run_kp_prior(X, s, method, seed)
rng(seed);
m = kernel_packet(X, s.K, s.SCALING_RHO, 0, method);
prior_sample(m, s.P_TIME);
end

function plot_prior_scaling(s, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'prior_scaling');
sizes = unique([rows.n], 'stable');
fig = figure('Visible', 'off', 'Position', [0, 0, 1100, 380]);
specs = {'time', sprintf('time of setup and %d prior draws (s)', s.P_TIME), [1, 3];
         'storage_mb', 'storage of the sampler (MB)', [1, 2]};
for i = 1:2
    col = specs{i, 1}; ttl = specs{i, 2}; slopes = specs{i, 3};
    series = struct();
    for mth = ["KP", "Cholesky"]
        [~, pair] = mean_sd(rows, 'n', col, 'method', mth);
        pad = numel(sizes) - numel(pair{1});
        series.(mth) = {[pair{1}, nan(1, pad)], [pair{2}, nan(1, pad)]};
    end
    metric_plot(subplot(1, 2, i), sizes, series, 'number of points n', ttl, slopes);
end
save_figure(fig, fullfile(out, 'prior_scaling_1d.png'));
end

function part_posterior_draws(s, out, method, fixed) %#ok<DEFNU>
rows = struct([]); curves = struct();
for seed = 0:s.SEEDS-1
    rng(seed);
    X = sort(s.train_inputs(s.N_POST, 1));
    xs = sort(s.test_inputs(s.M_POST, 1));
    y = s.f(X);
    for k = s.KS
        rho0 = 0.1; if isfield(fixed, 'rho'), rho0 = fixed.rho; end
        m = kernel_packet(X, k, rho0, 0, method);
        t_train = 0; if ~isfield(fixed, 'rho'), t_train = timed(@() mle(m, y, [], [1e-2, 1]), 0); end
        [~, beta, s2] = loglike(m, y);
        b = beta(1); if isfield(fixed, 'sigma2'), s2 = fixed.sigma2; end
        exact = dense_gp(X, y - b, k, m.rho, s2);
        [mu, C] = exact.predict(xs, true);
        draws = struct();
        rng(seed); [t_k, draws.KP] = timed(@() b + sqrt(s2) * posterior_sample(m, xs, (y - b) / sqrt(s2), s.P_POST));
        rng(seed); [t_c, draws.Cholesky] = timed(@() b + exact.sample(xs, s.P_POST));
        times = struct('KP', t_k, 'Cholesky', t_c);
        for mth = ["KP", "Cholesky"]
            F = draws.(mth);
            rows = [rows, struct('seed', seed, 'k', k, 'method', mth, 'n', s.N_POST, 'm', s.M_POST, ...
                'draws', s.P_POST, 'rho', m.rho, 'sigma2', s2, 'train_time', t_train, 'test_time', times.(mth), ...
                'w2', draws_wasserstein2(F, mu + b, C), 'mean_error', max_error(mean(F, 2), mu + b), ...
                'var_error', max_error(var(F, 1, 2), diag(C)), 'rmse', rmse(mean(F, 2), s.f(xs)))]; %#ok<AGROW>
        end
        if seed == 0
            curves.(sprintf('rho_%d', k)) = m.rho;
            curves.(sprintf('mean_%d', k)) = mu + b;
            curves.(sprintf('var_%d', k)) = diag(C);
            curves.(sprintf('KP_%d', k)) = draws.KP(:, 1:3);
            curves.(sprintf('Cholesky_%d', k)) = draws.Cholesky(:, 1:3);
            curves.X = X; curves.y = y; curves.xs = xs; curves.truth = s.f(xs);
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'posterior_draws');
save(fullfile(out, 'posterior_draws_1d.mat'), '-struct', 'curves');
plot_posterior_draws(s, out);
end

function plot_posterior_draws(s, out)
c = load(fullfile(out, 'posterior_draws_1d.mat'));
fig = figure('Visible', 'off', 'Position', [0, 0, 500 * numel(s.KS), 380]);
for i = 1:numel(s.KS)
    k = s.KS(i);
    draws = struct('KP', c.(sprintf('KP_%d', k)), 'Cholesky', c.(sprintf('Cholesky_%d', k)));
    draws_plot(subplot(1, numel(s.KS), i), c.xs, draws, ...
        sprintf('Matern-%d/2, n = %d, \\rho = %.3f', k - 2, numel(c.X), c.(sprintf('rho_%d', k))), ...
        c.X, c.y, c.truth, {c.(sprintf('mean_%d', k)), c.(sprintf('var_%d', k)), 'exact posterior mean and 95% interval'});
end
save_figure(fig, fullfile(out, 'posterior_draws_1d.png'));
end

function part_posterior_accuracy(s, out, method, ~) %#ok<DEFNU>
rows = struct([]);
for n = s.POST_ACC_N
    nseeds = 1; if n <= 1e5, nseeds = min(s.SEEDS, 3); end
    for seed = 0:nseeds-1
        rng(seed);
        X = sort(s.train_inputs(n, 1));
        xs = sort(s.test_inputs(s.M_POST, 1));
        y = s.f(X) + s.NOISE * randn(n, 1); truth = s.f(xs);
        m = kernel_packet(X, s.K, s.PRIOR_RHO, s.NOISE^2, method);
        rng(seed); Fk = posterior_sample(m, xs, y, s.P_ACC);
        w2_kp = predictive_w2(truth, mean(Fk, 2), var(Fk, 1, 2));
        if n > s.N_DENSE
            rows = [rows, struct('seed', seed, 'method', "KP", 'n', n, 'm', s.M_POST, 'draws', s.P_ACC, ...
                'rho', s.PRIOR_RHO, 'tau2', s.NOISE^2, 'w2', w2_kp, 'w2_exact', nan, ...
                'rmse', rmse(mean(Fk, 2), truth))]; %#ok<AGROW>
            continue
        end
        exact = dense_gp(X, y, s.K, s.PRIOR_RHO, 1.0, s.NOISE^2);
        [mu, C] = exact.predict(xs, true);
        rng(seed); Fc = exact.sample(xs, s.P_ACC);
        items = {{"KP", Fk, w2_kp}, {"Cholesky", Fc, predictive_w2(truth, mean(Fc, 2), var(Fc, 1, 2))}};
        for it = items
            [mth, F, w2_t] = it{1}{:};
            rows = [rows, struct('seed', seed, 'method', mth, 'n', n, 'm', s.M_POST, 'draws', s.P_ACC, ...
                'rho', s.PRIOR_RHO, 'tau2', s.NOISE^2, 'w2', w2_t, 'w2_exact', draws_wasserstein2(F, mu, C), ...
                'rmse', rmse(mean(F, 2), truth))]; %#ok<AGROW>
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'posterior_accuracy');
plot_posterior_accuracy(s, out);
end

function plot_posterior_accuracy(~, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'posterior_accuracy');
fig = figure('Visible', 'off', 'Position', [0, 0, 1100, 380]);
specs = {'w2', 'W_2 to the ground truth'; 'w2_exact', 'W_2 to the exact posterior'};
for i = 1:2
    [ns, series] = padded(rows, 'n', specs{i, 1});
    metric_plot(subplot(1, 2, i), ns, series, 'training size n', specs{i, 2});
end
save_figure(fig, fullfile(out, 'posterior_accuracy_1d.png'));
end

function part_posterior_sensitivity(s, out, method, ~) %#ok<DEFNU>
rows = struct([]);
sweeps = {{"rho", s.POST_SENS_RHOS}, {"tau2", s.POST_SENS_TAU2S}};
for sw = sweeps
    sweep = sw{1}{1}; vals = sw{1}{2};
    for val = vals
        if strcmp(sweep, 'rho'), r = val; t2 = s.NOISE^2;
        else, r = s.PRIOR_RHO; t2 = val; end
        for seed = 0:s.SEEDS-1
            rng(seed);
            X = sort(s.train_inputs(s.SENS_N, 1));
            xs = sort(s.test_inputs(s.M_POST, 1));
            y = s.f(X) + sqrt(t2) * randn(s.SENS_N, 1);
            m = kernel_packet(X, s.K, r, t2, method);
            exact = dense_gp(X, y, s.K, r, 1.0, t2);
            [mu, C] = exact.predict(xs, true);
            rng(seed); Fk = posterior_sample(m, xs, y, s.P_POST);
            rng(seed); Fc = exact.sample(xs, s.P_POST);
            items = {{"KP", Fk}, {"Cholesky", Fc}};
            for it = items
                [mth, F] = it{1}{:};
                rows = [rows, struct('seed', seed, 'sweep', string(sweep), 'method', mth, 'n', s.SENS_N, ...
                    'm', s.M_POST, 'draws', s.P_POST, 'rho', r, 'tau2', t2, ...
                    'w2', draws_wasserstein2(F, mu, C), 'rmse', rmse(mean(F, 2), s.f(xs)))]; %#ok<AGROW>
            end
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'posterior_sensitivity');
plot_posterior_sensitivity(s, out);
end

function plot_posterior_sensitivity(s, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'posterior_sensitivity');
fig = figure('Visible', 'off', 'Position', [0, 0, 1100, 380]);
specs = {"rho", 'rho', 'lengthscale \rho', sprintf('W_2 to the exact posterior (\\tau^2 = %g)', s.NOISE^2);
         "tau2", 'tau2', 'noise ratio \tau^2', sprintf('W_2 to the exact posterior (\\rho = %g)', s.PRIOR_RHO)};
for i = 1:2
    [ks, series] = padded(rows, specs{i, 2}, 'w2', 'sweep', specs{i, 1});
    metric_plot(subplot(1, 2, i), ks, series, specs{i, 3}, specs{i, 4});
end
save_figure(fig, fullfile(out, 'posterior_sensitivity_1d.png'));
end

function part_scaling_1d(s, out, method, fixed) %#ok<DEFNU>
rows = struct([]);
for n = s.SIZES
    nseeds = min(s.SEEDS, 3); if n > s.BIG, nseeds = 1; end
    for seed = 0:nseeds-1
        rng(seed);
        [X, y, ~] = s.scaling_data(n);
        [t, m] = timed(@() run_kp_scale1d(X, y, s, method, fixed), 0);
        rows = [rows, struct('seed', seed, 'method', "KP", 'n', n, 'rho', m.rho, 'train_time', t, ...
            'storage_mb', m.nbytes() / 2^20)]; %#ok<AGROW>
        if n <= s.N_DENSE
            rep = 1; if n >= 5000, rep = 0; end
            [t, exact] = timed(@() run_dense_scale1d(X, y, s, fixed), rep);
            rows = [rows, struct('seed', seed, 'method', "Cholesky", 'n', n, 'rho', exact.rho(1), ...
                'train_time', t, 'storage_mb', exact.nbytes() / 2^20)]; %#ok<AGROW>
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'scaling_1d');
plot_scaling_1d(s, out);
end

function m = run_kp_scale1d(X, y, s, method, fixed)
rho0 = s.SCALING_RHO; if isfield(fixed, 'rho'), rho0 = fixed.rho; end
m = kernel_packet(X, s.K, rho0, 0, method);
if ~isfield(fixed, 'rho'), mle(m, y, [], [1, 1e3]); end
end

function exact = run_dense_scale1d(X, y, s, fixed)
if isfield(fixed, 'rho'), exact = dense_gp(X, y, s.K, fixed.rho);
else, [exact, ~] = dense_gp.mle(X, y, s.K, [1, 1e3]); end
end

function plot_scaling_1d(~, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'scaling_1d');
sizes = unique([rows.n], 'stable');
fig = figure('Visible', 'off', 'Position', [0, 0, 1100, 380]);
specs = {'train_time', 'training time: maximum likelihood of \rho (s)', [1, 3];
         'storage_mb', 'storage of the fitted model (MB)', [1, 2]};
for i = 1:2
    col = specs{i, 1}; ttl = specs{i, 2}; slopes = specs{i, 3};
    series = struct();
    for mth = ["KP", "Cholesky"]
        [~, pair] = mean_sd(rows, 'n', col, 'method', mth);
        pad = numel(sizes) - numel(pair{1});
        series.(mth) = {[pair{1}, nan(1, pad)], [pair{2}, nan(1, pad)]};
    end
    metric_plot(subplot(1, 2, i), sizes, series, 'training size n', ttl, slopes);
end
save_figure(fig, fullfile(out, 'posterior_scaling_1d.png'));
end

function part_posterior_sparse_grid(s, out, ~, fixed) %#ok<DEFNU>
rows = struct([]);
has_fixed = ~isempty(fieldnames(fixed));
for d = s.SG_DIMS
    for seed = 0:min(s.SEEDS, 3)-1
        rng(seed);
        X = s.train_inputs(s.SG_N, d); xs = s.test_inputs(s.SG_M, d);
        y = s.griewank(X) + s.NOISE * randn(s.SG_N, 1);
        p = merge_struct(struct('rho', s.rho(d), 'sigma2', 1.0, 'tau2', s.NOISE^2), fixed);
        g = sparse_grid.for_data(X, s.K, p.rho, p.tau2);
        m = sparse_grid_regression(X, y, g, p.sigma2, p.tau2);
        t_train = 0; if ~has_fixed, t_train = timed(@() m.train(), 0); end
        exact = dense_gp(X, y, s.K, g.rho(1), m.sigma2, m.tau2);
        [mu, C] = exact.predict(xs, true);
        rng(seed); [t_k, F_k] = timed(@() posterior_sample(m, xs, [], s.P_POST));
        rng(seed); [t_c, F_c] = timed(@() exact.sample(xs, s.P_POST));
        items = {{"KP", F_k, t_k, m.nbytes()}, {"Cholesky", F_c, t_c, exact.nbytes()}};
        for it = items
            [mth, F, t, nb] = it{1}{:};
            rows = [rows, struct('seed', seed, 'd', d, 'method', mth, 'level', g.eta, 'inducing', size(g.X, 1), ...
                'n', s.SG_N, 'm', s.SG_M, 'draws', s.P_POST, 'rho', g.rho(1), 'sigma2', m.sigma2, ...
                'tau2', m.tau2, 'train_time', t_train, 'test_time', t, 'storage_mb', nb / 2^20, ...
                'w2', draws_wasserstein2(F, mu, C), 'rmse', rmse(mean(F, 2), s.griewank(xs)))]; %#ok<AGROW>
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'posterior_sparse_grid');
end

function [keys, pair] = mean_sd(rows, key, value, varargin)
mask = true(size(rows));
for i = 1:2:numel(varargin)
    f = varargin{i}; v = varargin{i+1};
    mask = mask & ([rows.(f)] == v);
end
sel = rows(mask);
keys = unique([sel.(key)], 'stable');
mu = zeros(size(keys)); sd = zeros(size(keys));
for i = 1:numel(keys)
    vals = [sel([sel.(key)] == keys(i)).(value)];
    mu(i) = mean(vals); sd(i) = std(vals, 1);
end
pair = {mu, sd};
end

function [keys, out] = padded(rows, key, value, varargin)
mask = true(size(rows));
for i = 1:2:numel(varargin)
    f = varargin{i}; v = varargin{i+1};
    mask = mask & ([rows.(f)] == v);
end
sel = rows(mask);
keys = unique([sel.(key)], 'stable');
out = struct();
valid = sel(~isnan([sel.(value)]));
for mth = ["KP", "Cholesky"]
    [ks, pair] = mean_sd(valid, key, value, 'method', mth);
    mu = nan(size(keys)); sd = nan(size(keys));
    [~, loc] = ismember(ks, keys);
    mu(loc) = pair{1}; sd(loc) = pair{2};
    out.(mth) = {mu, sd};
end
end

function s = merge_struct(a, b)
s = a;
for fn = fieldnames(b)'
    s.(fn{1}) = b.(fn{1});
end
end
