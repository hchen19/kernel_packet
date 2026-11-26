function main(varargin)
% Regression on toy data: exact kernel packets in 1-D, sparse-grid inducing points in d dimensions, against Cholesky.
%   scripts/run_regression.sh matlab toy [parts] [rho=.. sigma2=.. tau2=..]
%   scripts/run_regression.sh matlab toy replot [parts]
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
    fprintf('== %s\n', p{1});
    if replot, feval(['plot_' p{1}], s, out);
    else, feval(['part_' p{1}], s, out, method, fixed); end
end
end

function [m, b, s2, t_train] = fit_1d(X, y, k, noisy, method, fixed)
noisy = noisy && strcmp(method, 'stable');
rho0 = 0.1; if isfield(fixed, 'rho'), rho0 = fixed.rho; end
t20 = 0; if noisy && isfield(fixed, 'tau2'), t20 = fixed.tau2; end
m = kernel_packet(X, k, rho0, t20, method);
train_t2 = noisy && ~isfield(fixed, 'tau2');
if ~isfield(fixed, 'rho')
    if train_t2, [t_train, ~] = timed(@() mle(m, y, [], [1e-3, 1], [1e-8, 10]), 0);
    else, [t_train, ~] = timed(@() mle(m, y, [], [1e-3, 1]), 0); end
else
    t_train = 0;
end
[~, beta, s2] = loglike(m, y);
b = beta(1);
if isfield(fixed, 'sigma2'), s2 = fixed.sigma2; end
end

function part_posterior_1d(s, out, method, fixed) %#ok<DEFNU>
rng(0);
[X, y, ~, ~] = s.data_1d(s.N_1D, 0);
grid = linspace(0, 1, 1000)';
curves = struct('X', X, 'y', y, 'grid', grid, 'truth', s.f(grid));
for k = s.KS
    [m, b, s2, ~] = fit_1d(X, y, k, false, method, fixed);
    [mu, v] = posterior(m, grid, y - b);
    [ref, ref_v] = dense_gp(X, y - b, k, m.rho, s2).predict(grid);
    curves.(sprintf('rho_%d', k)) = m.rho;
    curves.(sprintf('KP_mean_%d', k)) = mu + b;
    curves.(sprintf('KP_var_%d', k)) = s2 * v;
    curves.(sprintf('Cholesky_mean_%d', k)) = ref + b;
    curves.(sprintf('Cholesky_var_%d', k)) = ref_v;
end
save(fullfile(out, 'posterior_1d.mat'), '-struct', 'curves');
plot_posterior_1d(s, out);
end

function plot_posterior_1d(s, out)
c = load(fullfile(out, 'posterior_1d.mat'));
fig = figure('Visible', 'off', 'Position', [0, 0, 500 * numel(s.KS), 380]);
for i = 1:numel(s.KS)
    k = s.KS(i);
    fits = struct('KP', {{c.(sprintf('KP_mean_%d', k)), c.(sprintf('KP_var_%d', k))}}, ...
        'Cholesky', {{c.(sprintf('Cholesky_mean_%d', k)), c.(sprintf('Cholesky_var_%d', k))}});
    posterior_plot(subplot(1, numel(s.KS), i), c.grid, c.truth, c.X, c.y, fits, ...
        sprintf('Matern-%d/2, n = %d, \\rho = %.3f', k - 2, numel(c.X), c.(sprintf('rho_%d', k))));
end
save_figure(fig, fullfile(out, 'posterior_1d.png'));
end

function part_accuracy_1d(s, out, method, fixed) %#ok<DEFNU>
rows = struct([]);
for n = s.N_ACC
    nseeds = s.SEEDS; if n > s.BIG, nseeds = 1; end
    for seed = 0:nseeds-1
        rng(seed);
        [X, y, xs, ys] = s.data_1d(n, s.NOISE);
        [m, b, s2, t_train] = fit_1d(X, y, s.K, true, method, fixed);
        [t, mu, v] = timed(@() posterior(m, xs, y - b));
        mu = mu + b; v = s2 * v;
        w2_kp = predictive_w2(ys, mu, v);
        base = struct('seed', seed, 'n', n, 'm', s.M, 'rho', m.rho, 'sigma2', s2, 'tau2', m.tau2, 'train_time', t_train);
        if n > s.N_DENSE
            r = merge_struct(base, struct('method', "KP", 'test_time', t, 'w2', w2_kp, 'w2_exact', nan, ...
                'mean_vs_cholesky', nan, 'var_vs_cholesky', nan));
            r = merge_struct(r, predictive(ys, mu, v, y));
            rows = [rows, r]; %#ok<AGROW>
            continue
        end
        [t_fit, exact] = timed(@() dense_gp(X, y - b, s.K, m.rho, s2, s2 * m.tau2));
        [t_ref, ref, ref_v] = timed(@() exact.predict(xs));
        ref = ref + b;
        xw = xs(1:s.M_W2);
        [mu_w, cov_w] = posterior(m, xw, y - b, true);
        [ref_w, ref_cov] = exact.predict(xw, true);
        w2_exact = wasserstein2(s2 * cov_w, ref_cov, mu_w - ref_w);
        items = {{"KP", mu, v, t, w2_kp, w2_exact}, {"Cholesky", ref, ref_v, t_fit + t_ref, predictive_w2(ys, ref, ref_v), 0}};
        for it = items
            [mth, mm, vv, tm, w, we] = it{1}{:};
            r = merge_struct(base, struct('method', mth, 'test_time', tm, 'w2', w, 'w2_exact', we, ...
                'mean_vs_cholesky', max_error(mm, ref), 'var_vs_cholesky', max_error(vv, ref_v)));
            r = merge_struct(r, predictive(ys, mm, vv, y));
            rows = [rows, r]; %#ok<AGROW>
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'accuracy_1d');
plot_accuracy_1d(s, out);
end

function plot_accuracy_1d(~, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'accuracy_1d');
cols = {'rmse', 'test RMSE', true; 'nlpd', 'test NLPD', false; 'w2', 'test W_2', true};
fig = figure('Visible', 'off', 'Position', [0, 0, 1500, 380]);
for i = 1:3
    [ns, series] = padded(rows, 'n', cols{i, 1});
    metric_plot(subplot(1, 3, i), ns, series, 'training size n', cols{i, 2}, [], true, cols{i, 3});
end
save_figure(fig, fullfile(out, 'accuracy_1d.png'));
end

function part_scaling_1d(s, out, method, fixed) %#ok<DEFNU>
rows = struct([]);
for n = s.SIZES
    nseeds = min(s.SEEDS, 3); if n > s.BIG, nseeds = 1; end
    for seed = 0:nseeds-1
        rng(seed);
        [X, y, xs, ys] = s.scaling_design(n, s.M);
        [t, m] = timed(@() run_kp_scale(X, y, s, method, fixed), 0);
        rows = [rows, struct('seed', seed, 'method', "KP", 'n', n, 'm', s.M, 'rho', m.rho, 'train_time', t, ...
            'storage_mb', m.nbytes() / 2^20, 'rmse', rmse(posterior(m, xs, y), ys))]; %#ok<AGROW>
        if n <= s.N_DENSE
            rep = 1; if n >= 5000, rep = 0; end
            [t, exact, beta] = timed(@() run_dense_scale(X, y, s, fixed), rep);
            [mu_e, ~] = exact.predict(xs);
            rows = [rows, struct('seed', seed, 'method', "Cholesky", 'n', n, 'm', s.M, 'rho', exact.rho(1), ...
                'train_time', t, 'storage_mb', exact.nbytes() / 2^20, 'rmse', rmse(mu_e + beta, ys))]; %#ok<AGROW>
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'scaling_1d');
plot_scaling_1d(s, out);
end

function m = run_kp_scale(X, y, s, method, fixed)
rho0 = s.SCALING_RHO; if isfield(fixed, 'rho'), rho0 = fixed.rho; end
m = kernel_packet(X, s.K, rho0, 0, method);
if ~isfield(fixed, 'rho'), mle(m, y, [], [1, 1e3]); end
end

function [exact, beta] = run_dense_scale(X, y, s, fixed)
if isfield(fixed, 'rho'), exact = dense_gp(X, y, s.K, fixed.rho); beta = 0;
else, [exact, beta] = dense_gp.mle(X, y, s.K, [1, 1e3]); end
end

function plot_scaling_1d(~, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'scaling_1d');
fig = figure('Visible', 'off', 'Position', [0, 0, 1100, 380]);
[~, s_train] = padded(rows, 'n', 'train_time');
[ns, s_mem] = padded(rows, 'n', 'storage_mb');
metric_plot(subplot(1, 2, 1), ns, s_train, 'training size n', 'training time: maximum likelihood of \rho and fit (s)', [1, 3]);
metric_plot(subplot(1, 2, 2), ns, s_mem, 'training size n', 'storage of the fitted model (MB)', [1, 2]);
save_figure(fig, fullfile(out, 'scaling_1d.png'));
end

function part_sparse_grid(s, out, ~, fixed) %#ok<DEFNU>
rows = struct([]); p0 = merge_struct(s.INIT, fixed);
has_fixed = ~isempty(fieldnames(fixed));
for d = s.SG_DIMS
    for seed = 0:s.SEEDS-1
        rng(seed);
        [X, y, xs, ys] = s.data_nd(d, s.SG_N);
        rng(100 + seed);
        ys = ys + s.NOISE * randn(s.M, 1);
        exact = dense_gp(X, y, s.K, p0.rho, p0.sigma2, p0.tau2);
        t_train = 0; if ~has_fixed, t_train = timed(@() exact.train(), 0); end
        t_fit = timed(@() exact.fit(exact.rho, exact.sigma2, exact.tau2));
        [t, ref, ref_v] = timed(@() exact.predict(xs));
        [ref_w, ref_cov] = exact.predict(xs(1:s.M_W2, :), true);
        r_c = merge_struct(struct('seed', seed, 'method', "Cholesky", 'd', d, 'level', "exact", 'inducing', nan, ...
            'n', s.SG_N, 'rho', exact.rho(1), 'sigma2', exact.sigma2, 'tau2', exact.tau2, 'train_time', t_train, ...
            'fit_time', t_fit, 'test_time', t, 'time', t_fit + t, 'storage_mb', exact.nbytes() / 2^20, 'w2', 0), ...
            predictive(ys, ref, ref_v + exact.tau2, y));
        rows = [rows, r_c]; %#ok<AGROW>
        auto = sparse_grid.for_data(X, s.K, p0.rho, p0.tau2);
        levs = num2cell(s.SG_LEVELS.(sprintf('d%d', d))); levs{end+1} = 'auto'; %#ok<AGROW>
        for lv = levs
            if ischar(lv{1}), g = auto; lvl_str = "auto";
            else, g = sparse_grid(d, d + lv{1}, s.K, p0.rho); lvl_str = string(g.eta); end
            m = sparse_grid_regression(X, y, g, p0.sigma2, p0.tau2);
            t_train = 0; if ~has_fixed, t_train = timed(@() m.train(), 0); end
            t_fit = timed(@() fit_fresh(m));
            [t, mu, v] = timed(@() m.predict(xs));
            [mu_w, cov_w] = m.predict(xs(1:s.M_W2, :), true);
            r_k = merge_struct(struct('seed', seed, 'method', "KP", 'd', d, 'level', lvl_str, ...
                'inducing', size(g.X, 1), 'n', s.SG_N, 'rho', g.rho(1), 'sigma2', m.sigma2, 'tau2', m.tau2, ...
                'train_time', t_train, 'fit_time', t_fit, 'test_time', t, 'time', t_fit + t, ...
                'storage_mb', m.nbytes() / 2^20, 'w2', wasserstein2(cov_w, ref_cov, mu_w - ref_w)), ...
                predictive(ys, mu, v + m.tau2, y));
            rows = [rows, r_k]; %#ok<AGROW>
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'sparse_grid');
plot_sparse_grid(s, out);
end

function fit_fresh(m)
m.clear_cache();
m.fit();
end

function plot_sparse_grid(s, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'sparse_grid');
lv = string({rows.level});
swept = rows(lv ~= "auto" & lv ~= "exact");
specs = {
    'accuracy', {'rmse', 'test RMSE', true; 'nlpd', 'test NLPD', false; 'w2', 'W_2 to the exact posterior', true};
    'scaling_sparse_grid', {'time', 'time of fit and prediction (s)', true; 'storage_mb', 'storage of the fitted model (MB)', true}
};
for sp = 1:2
    name = specs{sp, 1}; panels = specs{sp, 2}; np = size(panels, 1);
    for d = s.SG_DIMS
        fig = figure('Visible', 'off', 'Position', [0, 0, 500 * np, 380]);
        for i = 1:np
            col = panels{i, 1}; ttl = panels{i, 2}; logy = panels{i, 3};
            [nz, kps] = mean_sd(swept, 'inducing', col, 'method', "KP", 'd', d);
            series = struct('KP', {kps});
            if ~strcmp(col, 'w2')
                [~, chol_s] = mean_sd(rows, 'd', col, 'method', "Cholesky", 'd', d);
                series.Cholesky = {repmat(chol_s{1}(1), size(nz)), repmat(chol_s{2}(1), size(nz))};
            end
            metric_plot(subplot(1, np, i), nz, series, 'inducing points n_Z (sparse-grid levels)', ...
                sprintf('%s, d = %d, n = %d', ttl, d, s.SG_N), [], true, logy);
        end
        save_figure(fig, fullfile(out, sprintf('%s_%dd.png', name, d)));
    end
end
end

function part_scaling_nd(s, out, ~, fixed) %#ok<DEFNU>
rows = struct([]); p0 = merge_struct(s.INIT, fixed);
for d = s.SG_DIMS
    for n = s.ND_SIZES
        for seed = 0:min(s.SEEDS, 3)-1
            rng(seed);
            [X, y, xs, ys] = s.data_nd(d, n);
            [t, m] = timed(@() sparse_grid_regression(X, y, sparse_grid.for_data(X, s.K, p0.rho, p0.tau2), ...
                p0.sigma2, p0.tau2), 0);
            [mu_m, ~] = m.predict(xs);
            rows = [rows, struct('seed', seed, 'method', "KP", 'd', d, 'n', n, 'inducing', size(m.sg.X, 1), ...
                'train_time', t, 'storage_mb', m.nbytes() / 2^20, 'rmse', rmse(mu_m, ys))]; %#ok<AGROW>
            if n <= s.N_DENSE
                rep = 1; if n >= 5000, rep = 0; end
                [t, exact] = timed(@() dense_gp(X, y, s.K, p0.rho, p0.sigma2, p0.tau2), rep);
                [mu_e, ~] = exact.predict(xs);
                rows = [rows, struct('seed', seed, 'method', "Cholesky", 'd', d, 'n', n, 'inducing', nan, ...
                    'train_time', t, 'storage_mb', exact.nbytes() / 2^20, 'rmse', rmse(mu_e, ys))]; %#ok<AGROW>
            end
        end
    end
end
write_csv(rows, fullfile(out, 'metrics.csv'), 'scaling_nd');
plot_scaling_nd(s, out);
end

function plot_scaling_nd(s, out)
rows = read_csv(fullfile(out, 'metrics.csv'), 'scaling_nd');
specs = {'train_time', 'training time: fit at given hyperparameters (s)', [1, 3];
         'storage_mb', 'storage of the fitted model (MB)', [1, 2]};
for d = s.SG_DIMS
    fig = figure('Visible', 'off', 'Position', [0, 0, 1100, 380]);
    ns = unique([rows([rows.d] == d).n], 'stable');
    for i = 1:2
        col = specs{i, 1}; ttl = specs{i, 2}; slopes = specs{i, 3};
        series = struct();
        for mth = ["KP", "Cholesky"]
            [~, ms] = mean_sd(rows, 'n', col, 'method', mth, 'd', d);
            pad = numel(ns) - numel(ms{1});
            series.(mth) = {[ms{1}, nan(1, pad)], [ms{2}, nan(1, pad)]};
        end
        metric_plot(subplot(1, 2, i), ns, series, 'training size n', sprintf('%s, d = %d', ttl, d), slopes);
    end
    save_figure(fig, fullfile(out, sprintf('scaling_%dd.png', d)));
end
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

function [keys, out] = padded(rows, key, value)
keys = unique([rows.(key)], 'stable');
out = struct();
valid = rows(~isnan([rows.(value)]));
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
