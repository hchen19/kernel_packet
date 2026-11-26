function [theta, L_hat, L_init] = mle(model, y, F, bounds, tau2_bounds)
% Maximize the log-likelihood over a scalar rho in bounds (fminbnd on log rho), or, with tau2_bounds, over rho and
% tau2 (clamped fminsearch on their logarithms); theta = struct(rho, tau2, beta, sigma2), model.rho (and tau2) set
if nargin < 3 || isempty(F), F = ones(numel(y), 1); end
if nargin < 4 || isempty(bounds), bounds = [5e-2, 5]; end
if nargin < 5, tau2_bounds = []; end
L_init = loglike(model, y, F);
if isempty(tau2_bounds)
    z = fminbnd(@nll_rho, log(bounds(1)), log(bounds(2)), optimset('TolX', 1e-3));
    model.rho = exp(z);
else
    lb = [log(bounds(1)); log(tau2_bounds(1))];
    ub = [log(bounds(2)); log(tau2_bounds(2))];
    z = fminsearch(@nll_joint, 0.5 * (lb + ub), optimset('TolX', 1e-3, 'TolFun', 1e-6));
    p = exp(min(max(z(:), lb), ub));
    model.set_params(p(1), p(2));
end
[L_hat, beta, sigma2] = loglike(model, y, F);
t2 = 0; if isa(model, 'kernel_packet'), t2 = model.tau2; end
theta = struct('rho', model.rho, 'tau2', t2, 'beta', beta, 'sigma2', sigma2);

    function v = nll_rho(z)
        model.rho = exp(z);
        v = -loglike(model, y, F);
        if ~isreal(v) || ~isfinite(v), v = 1e30; end
    end

    function v = nll_joint(z)
        p = exp(min(max(z(:), lb), ub));
        model.set_params(p(1), p(2));
        v = -loglike(model, y, F);
        if ~isreal(v) || ~isfinite(v), v = 1e30; end
    end
end
