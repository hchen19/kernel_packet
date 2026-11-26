function s = styles(name)
% Color, marker and line of each method in every figure, KP drawn thick and on top
switch name
    case 'KP', s = struct('Color', [0.110, 0.361, 0.671], 'Marker', 'o', 'LineStyle', '-', 'LineWidth', 2.0);
    case 'Cholesky', s = struct('Color', [0.878, 0.482, 0], 'Marker', 's', 'LineStyle', '--', 'LineWidth', 1.3);
end
end
