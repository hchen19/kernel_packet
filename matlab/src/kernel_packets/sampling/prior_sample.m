function F = prior_sample(kp, p)
% p prior draws at the nodes of the kernel_packet kp (n*p): C~^{-1}*L_B*Z (stable) or Phi*(L'\Z) (original) (C++)
if nargin < 2, p = 1; end
F = kp.sample_(randn(kp.n, p));
end
