function F = posterior_sample(model, xnew, y, p)
% p posterior draws at xnew (m*p).  kernel_packet model with data y at its nodes: Matheron's rule with noise ratio
% tau2 (C++).  sparse_grid_regression model: mean + sqrt(sigma2)*K(xnew,Z)*(L'\z).
if nargin < 4, p = 1; end
if isa(model, 'sparse_grid_regression')
    M = matern(xnew, model.sg.X, model.sg.k, model.sg.rho);
    F = model.predict(xnew) + sqrt(model.sigma2) * (M * (model.L' \ randn(size(M, 2), p)));
else
    F = kp_mex('kp_posterior_sample', model.h, xnew(:), y(:), randn(model.n + numel(xnew) + model.n * (model.tau2 > 0), p));
end
end
