function draws_plot(ax, xs, draws, ttl, X, y, truth, band)
% draws: struct with fields KP, Cholesky of draws (m*p), one legend entry per method; optional band = {mean, var, label}
xs = xs(:);
hold(ax, 'on');
if nargin > 7 && ~isempty(band)
    bmu = band{1}(:); bsd = sqrt(max(band{2}(:), 0));
    fill(ax, [xs; flipud(xs)], [bmu - 1.96*bsd; flipud(bmu + 1.96*bsd)], [0.89, 0.89, 0.87], ...
        'FaceAlpha', 0.9, 'EdgeColor', 'none', 'DisplayName', band{3});
    plot(ax, xs, bmu, 'Color', [0.32, 0.32, 0.31], 'LineWidth', 0.8, 'HandleVisibility', 'off');
end
if nargin > 6 && ~isempty(truth)
    plot(ax, xs, truth, ':', 'Color', [0.04, 0.04, 0.04], 'LineWidth', 1.2, 'DisplayName', 'ground truth');
end
for name = string(fieldnames(draws))'
    st = styles(name); F = draws.(name);
    h = plot(ax, xs, F, 'Color', st.Color, 'LineStyle', st.LineStyle, 'LineWidth', 0.9);
    set(h(2:end), 'HandleVisibility', 'off');
    h(1).DisplayName = name + " draws";
end
if nargin > 4 && ~isempty(X)
    plot(ax, X, y, 'o', 'Color', [0.84, 0.15, 0.16], 'MarkerFaceColor', [0.84, 0.15, 0.16], 'MarkerSize', 3, ...
        'LineStyle', 'none', 'DisplayName', 'observations');
end
xlabel(ax, 'x');
title(ax, ttl);
legend(ax, 'FontSize', 7, 'Box', 'off');
grid(ax, 'on'); box(ax, 'off');
end
