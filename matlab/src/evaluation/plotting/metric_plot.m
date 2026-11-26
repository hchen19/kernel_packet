function metric_plot(ax, x, series, xlab, ttl, slopes, logx, logy)
% series: struct of values or {mean, sd} at x per method (fields KP, Cholesky), the quantity named by the title;
% slopes: reference powers p drawn as c*x^p through the first finite point of the first method
if nargin < 6, slopes = []; end
if nargin < 7, logx = true; end
if nargin < 8, logy = true; end
x = double(x(:))';
hold(ax, 'on');
names = string(fieldnames(series))';
first_vals = [];
for name = names
    st = styles(name);
    v = series.(name);
    if iscell(v)
        mu = double(v{1}(:))'; sd = double(v{2}(:))';
    else
        mu = double(v(:))'; sd = [];
    end
    if isempty(first_vals), first_vals = mu; end
    mask = isfinite(mu) & (~logy | mu > 0);
    if ~any(mask), continue; end
    if isempty(sd)
        plot(ax, x(mask), mu(mask), 'Color', st.Color, 'Marker', st.Marker, 'LineStyle', st.LineStyle, ...
            'LineWidth', st.LineWidth, 'MarkerFaceColor', st.Color, 'MarkerSize', 5, 'DisplayName', name);
    else
        lo = sd(mask); if logy, lo = min(lo, 0.99 * mu(mask)); end
        errorbar(ax, x(mask), mu(mask), lo, sd(mask), 'Color', st.Color, 'Marker', st.Marker, ...
            'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth, 'MarkerFaceColor', st.Color, 'MarkerSize', 5, ...
            'CapSize', 2, 'DisplayName', name);
    end
end
guides = {':', '--', '-.'};
if ~isempty(first_vals)
    idx = find(isfinite(first_vals) & (~logy | first_vals > 0), 1, 'first');
    if ~isempty(idx)
        for p = slopes
            ls = ':'; if p >= 1 && p <= 3, ls = guides{p}; end
            lbl = 'O(n)'; if p ~= 1, lbl = sprintf('O(n^%d)', p); end
            plot(ax, x, first_vals(idx) * (x / x(idx)).^p, ls, 'Color', [0.32, 0.32, 0.31], ...
                'LineWidth', 0.9, 'DisplayName', lbl);
        end
    end
end
if logx, set(ax, 'XScale', 'log'); end
if logy, set(ax, 'YScale', 'log'); end
xlabel(ax, xlab);
title(ax, ttl);
legend(ax, 'FontSize', 7, 'Box', 'off');
grid(ax, 'on'); box(ax, 'off');
end
