function posterior_plot(ax, xs, truth, X, y, fits, ttl)
% fits: struct with fields KP, Cholesky, each {mean, var}: KP mean and 95% interval, Cholesky mean, truth and data
xs = xs(:);
hold(ax, 'on');
plot(ax, xs, truth, ':', 'Color', [0.04, 0.04, 0.04], 'LineWidth', 1.2, 'DisplayName', 'ground truth');
for name = string(fieldnames(fits))'
    st = styles(name); mu = fits.(name){1}(:); sd = sqrt(max(fits.(name){2}(:), 0));
    if name == "KP"
        fill(ax, [xs; flipud(xs)], [mu - 1.96*sd; flipud(mu + 1.96*sd)], st.Color, 'FaceAlpha', 0.18, ...
            'EdgeColor', 'none', 'DisplayName', 'KP 95% interval');
        plot(ax, xs, mu, 'Color', st.Color, 'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth, ...
            'DisplayName', 'KP mean');
    else
        plot(ax, xs, mu, 'Color', st.Color, 'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth, ...
            'DisplayName', name + " mean");
    end
end
plot(ax, X, y, 'o', 'Color', [0.84, 0.15, 0.16], 'MarkerFaceColor', [0.84, 0.15, 0.16], 'MarkerSize', 3, ...
    'LineStyle', 'none', 'DisplayName', 'observations');
xlabel(ax, 'x');
title(ax, ttl);
legend(ax, 'FontSize', 7, 'Box', 'off');
grid(ax, 'on'); box(ax, 'off');
end
