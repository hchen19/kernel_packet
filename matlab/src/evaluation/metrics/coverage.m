function c = coverage(y, mu, v, z)
% Fraction of y inside mu +- z sd
if nargin < 4, z = 1.959963984540054; end
c = mean(abs(y(:) - mu(:)) <= z * sqrt(max(v(:), 0)));
end
