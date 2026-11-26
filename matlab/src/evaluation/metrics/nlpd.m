function e = nlpd(y, mu, v)
% Mean negative log predictive density of y under N(mu, v)
e = mean(0.5 * log(2*pi*v(:)) + 0.5 * (y(:) - mu(:)).^2 ./ v(:));
end
