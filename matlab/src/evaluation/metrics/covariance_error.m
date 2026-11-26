function e = covariance_error(L, K)
% max |L*L' - K| of a sampler's implied factor L (its draws from the unit vectors)
e = max_error(L * L', K);
end
