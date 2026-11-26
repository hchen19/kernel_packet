function y = predict(xnew, X, w, k, rho)
% Posterior mean K(xnew,X)*w, w = K\(y - mean), summed on the fly (no m*N matrix, C++)
y = kp_mex('kernel_predict', xnew', X', w, k, rho .* ones(1, size(X, 2)));
end
