classdef kernel_packet < handle
    % Kernel packets of the Matern-(k-2)/2 kernel with lengthscale rho on sorted 1-D nodes x, noise ratio tau2 (C++
    % core through kp_mex).  method 'stable' (default; k <= 7 and n >= k, else the original is taken): the stable KPs
    % (factored one-sided whitening, Green-form mean, selected-inverse variance); 'original' (noise-free):
    % K\Y = A*(Phi\Y), log|K| = log|Phi| - log|A|, K = Phi*inv(B)*Phi' with B = A'*Phi banded
    properties (SetAccess = private)
        x
        k
        n
        method = 'stable'
    end
    properties (Dependent)
        rho
        tau2
    end
    properties (Access = private)
        rho_
        tau2_ = 0
    end
    properties (SetAccess = private)
        h = uint64(0)
    end

    methods
        function obj = kernel_packet(x, k, rho, tau2, method)
            if nargin < 2, k = 5; end
            if nargin < 3, rho = sqrt(k-2); end
            if nargin >= 4, obj.tau2_ = tau2; end
            if nargin >= 5, obj.method = method; end
            if exist('kp_mex', 'file') ~= 3, kp_build; end
            obj.x = x(:);
            obj.k = k;
            obj.n = numel(x);
            obj.rho_ = rho;
            [obj.h, taken] = kp_mex('kp_new', obj.x, obj.k, obj.rho_, obj.tau2_, double(~strcmp(obj.method, 'stable')));
            names = {'stable', 'original'};
            obj.method = names{taken + 1};
        end

        function r = get.rho(obj), r = obj.rho_; end
        function set.rho(obj, rho)
            obj.rho_ = rho;
            if obj.h, kp_mex('kp_set_params', obj.h, obj.rho_, obj.tau2_); end
        end

        function t = get.tau2(obj), t = obj.tau2_; end
        function set.tau2(obj, tau2)
            obj.tau2_ = tau2;
            if obj.h, kp_mex('kp_set_params', obj.h, obj.rho_, obj.tau2_); end
        end

        function set_params(obj, rho, tau2)
            obj.rho_ = rho; obj.tau2_ = tau2;
            kp_mex('kp_set_params', obj.h, obj.rho_, obj.tau2_);
        end

        function b = nbytes(obj)
            b = kp_mex('kp_bytes', obj.h);
        end

        function delete(obj)
            obj.free();
        end

        function L = logdet(obj)
            L = kp_mex('kp_logdet', obj.h);
        end

        function W = solve(obj, Y, with_a)
            % K\Y (Phi\Y if ~with_a) for Y with n rows
            if nargin < 3, with_a = true; end
            W = kp_mex('kp_solve', obj.h, Y, with_a);
        end

        function F = sample_(obj, Z)
            % Phi*(L'\Z) (B = L*L') for Z with n rows: N(0,K) from N(0,I)
            F = kp_mex('kp_sample', obj.h, Z);
        end

        function P = basis(obj, xnew)
            % sparse m*n matrix of the KPs at xnew, P(i,j) = phi_j(xnew(i))
            [s, V] = kp_mex('kp_basis', obj.h, xnew(:));
            [w, m] = size(V);
            P = sparse(repmat(1:m, w, 1), s' + (0:w-1)', V, m, obj.n);
        end
    end

    methods (Access = private)
        function free(obj)
            if obj.h, kp_mex('kp_free', obj.h); obj.h = uint64(0); end
        end
    end
end
