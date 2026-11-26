function e = rmse(a, b)
e = sqrt(mean((a(:) - b(:)).^2));
end
