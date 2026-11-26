function [mu, v] = posterior(kp, xnew, y, cov)
% Posterior mean K(xnew,X)*((K + tau2*I)\y) of the kernel_packet kp (zero prior mean, unit variance, noise ratio
% kp.tau2) and the variance, or the covariance if cov (C++)
if nargin > 3 && cov
    mu = kp_mex('kp_posterior', kp.h, xnew(:), y(:));
    v = kp_mex('kp_posterior_cov', kp.h, xnew(:));
elseif nargout > 1
    [mu, v] = kp_mex('kp_posterior', kp.h, xnew(:), y(:));
else
    mu = kp_mex('kp_posterior', kp.h, xnew(:), y(:));
end
end
