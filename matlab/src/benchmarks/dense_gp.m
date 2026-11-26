classdef dense_gp < handle
    % Exact Gaussian process by dense Cholesky factorization: the reference of every experiment, O(n^3) time, O(n^2)
    % memory; kernel sigma2*matern(k, rho), noise tau2, data (X, y)
    properties
        X
        y
        k
        rho
        sigma2
        tau2
        L
        alpha
        loglike
    end

    methods
        function obj = dense_gp(X, y, k, rho, sigma2, tau2)
            if nargin < 3, k = 5; end
            if nargin < 4, rho = 1; end
            if nargin < 5, sigma2 = 1; end
            if nargin < 6, tau2 = 0; end
            obj.X = X; obj.y = y(:); obj.k = k;
            obj.fit(rho, sigma2, tau2);
        end

        function obj = fit(obj, rho, sigma2, tau2)
            obj.rho = rho; obj.sigma2 = sigma2; obj.tau2 = tau2;
            K = sigma2 * matern(obj.X, obj.X, obj.k, rho) + tau2 * eye(size(obj.X, 1));
            obj.L = chol(K, 'lower');
            obj.alpha = obj.L' \ (obj.L \ obj.y);
            n = numel(obj.y);
            obj.loglike = -0.5 * (obj.y' * obj.alpha + 2 * sum(log(diag(obj.L))) + n * log(2*pi));
        end

        function obj = train(obj, bounds)
            % Exact maximum likelihood over (rho, sigma2, tau2) in bounds (Nelder-Mead on their logarithms)
            if nargin < 2, bounds = [1e-2, 1e2; 1e-4, 1e4; 1e-8, 1e2]; end
            lb = log(bounds(:, 1)); ub = log(bounds(:, 2));
            z0 = log([obj.rho(1); obj.sigma2; max(obj.tau2, 1e-6)]);
            z = fminsearch(@nll, z0, optimset('TolX', 1e-3, 'TolFun', 1e-3));
            p = exp(min(max(z(:), lb), ub));
            obj.fit(p(1), p(2), p(3));

            function v = nll(z)
                p = exp(min(max(z(:), lb), ub));
                K = p(2) * matern(obj.X, obj.X, obj.k, p(1)) + p(3) * eye(size(obj.X, 1));
                [L_, fail] = chol(K, 'lower');
                if fail > 0, v = 1e30; return; end
                a = L_' \ (L_ \ obj.y);
                v = 0.5 * (obj.y' * a + 2 * sum(log(diag(L_))) + numel(obj.y) * log(2*pi));
                if ~isreal(v) || ~isfinite(v), v = 1e30; end
            end
        end

        function b = nbytes(obj)
            b = 8 * (numel(obj.L) + numel(obj.X) + numel(obj.alpha));
        end

        function [mu, v] = predict(obj, xnew, cov)
            % posterior mean and variance (or covariance if cov) of f at xnew; variance in batches
            if nargin > 2 && cov
                Ks = obj.sigma2 * matern(xnew, obj.X, obj.k, obj.rho);
                V = obj.L \ Ks';
                mu = Ks * obj.alpha;
                v = obj.sigma2 * matern(xnew, xnew, obj.k, obj.rho) - V' * V;
                return
            end
            m = size(xnew, 1);
            mu = zeros(m, 1); v = zeros(m, 1);
            bs = max(1, floor(2^24 / size(obj.X, 1)));
            for i = 1:bs:m
                r = i:min(i+bs-1, m);
                Ks = obj.sigma2 * matern(xnew(r, :), obj.X, obj.k, obj.rho);
                V = obj.L \ Ks';
                mu(r) = Ks * obj.alpha;
                v(r) = obj.sigma2 - sum(V.^2, 1)';
            end
        end

        function F = sample(obj, xnew, p)
            % p posterior draws at xnew (m*p)
            if nargin < 3, p = 1; end
            [mu, C] = obj.predict(xnew, true);
            F = mu + chol(C + 1e-12 * eye(numel(mu)), 'lower') * randn(numel(mu), p);
        end
    end

    methods (Static)
        function [g, beta] = mle(X, y, k, bounds)
            % Profile maximum likelihood of rho (fminbnd on log rho), constant mean and sigma2 profiled out
            if nargin < 4, bounds = [1e-3, 1]; end
            y = y(:); n = numel(y); ones_v = ones(n, 1);
            z = fminbnd(@nll, log(bounds(1)), log(bounds(2)), optimset('TolX', 1e-3));
            rho = exp(z);
            [beta, s2, ~] = profile(rho);
            g = dense_gp(X, y - beta, k, rho, s2);

            function [b, s2, out] = profile(rho_val)
                [L_, fail] = chol(matern(X, X, k, rho_val), 'lower');
                if fail > 0, b = 0; s2 = nan; out = 1e30; return; end
                a = L_ \ ones_v; c = L_ \ y;
                b = (a' * c) / (a' * a);
                r = c - b * a;
                s2 = (r' * r) / n;
                out = 0.5 * (n * log(s2) + 2 * sum(log(diag(L_))) + n);
            end

            function v = nll(z_val)
                [~, ~, v] = profile(exp(z_val));
                if ~isreal(v) || ~isfinite(v), v = 1e30; end
            end
        end

        function F = prior_sample(X, k, rho, p)
            % p prior draws at X by the Cholesky factor of K (NaN if K is numerically non-PD)
            if nargin < 4, p = 1; end
            [L, fail] = chol(matern(X, X, k, rho), 'lower');
            if fail > 0, F = nan(size(X, 1), p); else, F = L * randn(size(X, 1), p); end
        end
    end
end
