function w = draws_wasserstein2(F, mu, C)
% W2 between the Gaussian of the draws F (m*p) (sample mean and covariance) and N(mu, C)
w = wasserstein2(cov(F'), C, mean(F, 2) - mu(:));
end
