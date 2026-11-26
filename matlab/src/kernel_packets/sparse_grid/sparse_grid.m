classdef sparse_grid < handle
    % Sparse grid of level eta >= d in [lo,hi]^d from a nested 1-D design, the inducing points of d-dimensional
    % regression (C++ core: design, K\Y by the combination technique over tensor grids of 1-D kernel packets, log|K|):
    % K\Y is the sum over |j| = max(d,eta-d+1)..eta of (-1)^(eta-|j|)*nchoosek(d-1,eta-|j|) times the tensor-grid K\Y
    % of level j; points X with sorted rows
    properties (SetAccess = private)
        d
        eta
        k
        X
        h = uint64(0)
    end
    properties (Dependent)
        rho
    end
    properties (Access = private)
        rho_
    end

    methods
        function obj = sparse_grid(d, eta, k, rho, design, lo, hi)
            if nargin < 3, k = 5; end
            if nargin < 4 || isempty(rho), rho = sqrt(k-2); end
            if nargin < 5 || isempty(design), design = @hc; end
            if nargin < 6, lo = 0; end
            if nargin < 7, hi = 1; end
            if exist('kp_mex', 'file') ~= 3, kp_build; end
            obj.d = d; obj.eta = eta; obj.k = k;
            nodes = arrayfun(@(j) design(j, lo, hi), 0:eta-d+1, 'UniformOutput', false);
            obj.rho_ = rho .* ones(1, d);
            obj.h = kp_mex('sg_new', d, eta, [nodes{:}], cumsum([0, cellfun(@numel, nodes)]), k, obj.rho_);
            obj.X = kp_mex('sg_points', obj.h, d)';
        end

        function r = get.rho(obj)
            r = obj.rho_;
        end

        function set.rho(obj, rho)
            obj.rho_ = rho .* ones(1, obj.d);
            kp_mex('sg_set_rho', obj.h, obj.rho_);
        end

        function delete(obj)
            if obj.h, kp_mex('sg_free', obj.h); end
        end

        function W = solve(obj, Y)
            % K\Y for Y with N rows
            W = kp_mex('sg_solve', obj.h, Y);
        end

        function L = logdet(obj)
            L = kp_mex('sg_logdet', obj.h);
        end

        function b = nbytes(obj)
            b = kp_mex('sg_bytes', obj.h);
        end

        function t = trace_solve(obj, G)
            % trace(K\G) for an N*N matrix G (C++, O(N) memory per thread)
            t = kp_mex('sg_trace_solve', obj.h, G);
        end
    end

    methods (Static)
        function g = for_data(X, k, rho, tau2, n_max, kl, design, lo, hi)
            % The sparse grid of the smallest level eta with tr(K_XX - Q_XX) / tau2 <= kl for X (n*d), within n_max
            if nargin < 2 || isempty(k), k = 5; end
            if nargin < 3 || isempty(rho), rho = sqrt(k-2); end
            if nargin < 4 || isempty(tau2), tau2 = 1e-2; end
            if nargin < 5 || isempty(n_max), n_max = 4000; end
            if nargin < 6 || isempty(kl), kl = 1.0; end
            if nargin < 7 || isempty(design), design = @hc; end
            if nargin < 8, lo = 0; end
            if nargin < 9, hi = 1; end
            [n, d] = size(X);
            n_max = min(n_max, n);
            r = rho .* ones(1, d);
            L = max(1, floor(log2(n_max + 1)));
            sizes = arrayfun(@(l) numel(design(l, lo, hi)), 0:L);
            D = zeros(d, L, n);
            for j = 1:d
                v0 = ones(n, 1);
                for l = 1:L
                    [~, vl] = posterior(kernel_packet(design(l, lo, hi), k, r(j)), X(:, j), zeros(sizes(l+1), 1));
                    D(j, l, :) = v0 - vl;
                    v0 = vl;
                end
            end
            dn = diff(sizes);
            S = ones(1, n); C = 1;
            for j = 1:d
                S2 = zeros(size(S, 1) + L, n);
                C2 = zeros(numel(C) + L, 1);
                for l = 1:L
                    dj = reshape(D(j, l, :), 1, n);
                    S2(l+1:l+size(S, 1), :) = S2(l+1:l+size(S, 1), :) + S .* dj;
                    C2(l+1:l+numel(C)) = C2(l+1:l+numel(C)) + dn(l) * C;
                end
                S = S2; C = C2;
            end
            q = cumsum(sum(S, 2));
            N = cumsum(C);
            etas = find(N <= n_max) - 1;
            etas = etas(etas >= d);
            if isempty(etas), etas = d; end
            idx = find((n - q(etas + 1)) / tau2 <= kl, 1, 'first');
            if isempty(idx), eta = etas(end); else, eta = etas(idx); end
            g = sparse_grid(d, eta, k, r, design, lo, hi);
        end
    end
end
