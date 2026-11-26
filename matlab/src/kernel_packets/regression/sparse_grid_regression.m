classdef sparse_grid_regression < handle
    % Scattered data (X, y) in d dimensions with the points Z of a sparse_grid as inducing points (Titsias VFE), kernel
    % sigma2*matern and noise tau2.  With K = matern and Sigma = K_ZZ + sigma2*K_ZX*K_XZ/tau2 = L*L':
    %   mean(x) = sigma2*K(x,Z)*alpha, alpha = Sigma\(K_ZX*y)/tau2,
    %   var(x) = sigma2*(1 - K(x,Z)*(K_ZZ\K(Z,x)) + |L\K(Z,x)|^2).
    % K_ZZ\ and log|K_ZZ| come from kernel packets (C++); K_ZX*K_XZ, K_ZX*y and the Cholesky factor of Sigma are dense
    % BLAS work, accumulated in chunks (memory of n_Z^2, not n*n_Z)
    properties
        X
        y
        sg
        sigma2
        tau2
        L
        alpha
        elbo
    end
    properties (Access = private)
        cache = struct('rho', [])   % G, b, K_ZZ, the trace and log|K_ZZ| depend on rho only: reused across sigma2, tau2
    end

    methods
        function obj = sparse_grid_regression(X, y, sg, sigma2, tau2)
            if nargin < 4, sigma2 = 1; end
            if nargin < 5, tau2 = 1e-4; end
            obj.X = X; obj.y = y(:); obj.sg = sg; obj.sigma2 = sigma2; obj.tau2 = tau2;
            obj.fit();
        end

        function clear_cache(obj)
            obj.cache = struct('rho', []);
        end

        function b = nbytes(obj)
            c = 0;
            if ~isempty(obj.cache.rho), c = numel(obj.cache.G) + numel(obj.cache.Kzz); end
            b = obj.sg.nbytes() + 8 * (numel(obj.L) + numel(obj.alpha) + c + numel(obj.sg.X));
        end

        function obj = fit(obj)
            % factor Sigma and the variational lower bound elbo of the log marginal likelihood at the current parameters
            Z = obj.sg.X; k = obj.sg.k; rho = obj.sg.rho; s = obj.sigma2; t = obj.tau2; y = obj.y;
            n = numel(y); n_z = size(Z, 1);
            if ~isequal(obj.cache.rho, rho)
                G = zeros(n_z); b = zeros(n_z, 1);
                c = max(1, floor(2^24 / n_z));
                for i = 1:c:n
                    r = i:min(i+c-1, n);
                    Kc = matern(Z, obj.X(r, :), k, rho);
                    G = G + Kc * Kc'; b = b + Kc * y(r);
                end
                % tr(K_XZ*(K_ZZ\K_ZX)) = tr(K_ZZ\G): n_Z columns, not n (C++)
                obj.cache = struct('rho', rho, 'G', G, 'b', b, 'Kzz', matern(Z, Z, k, rho), ...
                    'trq', obj.sg.trace_solve(G), 'ld_zz', obj.sg.logdet());
            end
            G = obj.cache.G; b = obj.cache.b; trq = obj.cache.trq;
            A = obj.cache.Kzz + (s / t) * G;
            % A is positive definite but cond(A) can exceed 1/eps: a diagonal of n_Z*eps times its mean keeps the factor
            % defined in every BLAS (CPU or GPU) at a relative change of ~1e-13
            A = A + n_z * eps * mean(diag(A)) * eye(n_z);
            obj.L = chol(A, 'lower');
            obj.alpha = obj.L' \ (obj.L \ b) / t;
            quad = (y' * y - s * (b' * obj.alpha)) / t;
            logdet = 2 * sum(log(diag(obj.L))) - obj.cache.ld_zz + n * log(t);
            obj.elbo = -0.5 * (quad + logdet + n * log(2*pi)) - 0.5 * s * (n - trq) / t;
        end

        function [mu, v] = predict(obj, xnew, cov)
            % posterior mean (C++ kernel sum) and variance (or covariance if cov) at the rows of xnew
            if nargin < 3, cov = false; end
            mu = obj.sigma2 * predict(xnew, obj.sg.X, obj.alpha, obj.sg.k, obj.sg.rho);
            if nargout > 1 || cov
                M = matern(xnew, obj.sg.X, obj.sg.k, obj.sg.rho);
                S = obj.sg.solve(M');
                V = obj.L \ M';
                if cov
                    v = obj.sigma2 * (matern(xnew, xnew, obj.sg.k, obj.sg.rho) - M * S + V' * V);
                else
                    v = obj.sigma2 * (1 - sum(M .* S', 2) + sum(V.^2, 1)');
                end
            end
        end

        function obj = train(obj, bounds)
            % maximize the bound over (rho, sigma2, tau2) in bounds (Nelder-Mead on their logarithms, clipped to the
            % bounds); rho is shared by the axes
            if nargin < 2, bounds = [1e-2, 1e2; 1e-4, 1e4; 1e-8, 1e2]; end
            lb = log(bounds(:, 1)); ub = log(bounds(:, 2));
            z0 = log([obj.sg.rho(1); obj.sigma2; obj.tau2]);
            z = fminsearch(@nelbo, z0, optimset('TolX', 1e-3, 'TolFun', 1e-3));
            nelbo(z);

            function v = nelbo(z)
                p = exp(min(max(z, lb), ub));
                obj.sg.rho = p(1); obj.sigma2 = p(2); obj.tau2 = p(3);
                obj.fit();
                v = -obj.elbo;
            end
        end
    end
end
