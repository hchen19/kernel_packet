function [L, beta, sigma2] = loglike(model, y, F)
% Log-likelihood at model.rho with mean F*beta and variance sigma2 at their MLEs (C++; model: kernel_packet or
% sparse_grid with y at its points)
if nargin < 3, F = ones(numel(y), 1); end
if isa(model, 'sparse_grid'), cmd = 'sg_loglike'; else, cmd = 'kp_loglike'; end
[L, beta, sigma2] = kp_mex(cmd, model.h, y(:), F);
end
