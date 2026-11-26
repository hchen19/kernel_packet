function save_figure(fig, path)
exportgraphics(fig, path, 'Resolution', 200);
close(fig);
end
